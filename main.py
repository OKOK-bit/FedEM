import argparse

import torch.multiprocessing as tmx
from matplotlib import pyplot as plt
from torchvision.models import VGG16_Weights

from basic.dataset import *
from basic.utils import set_seed, AvgMeter
from attack.dlg_attack import perform_dlg
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter
import yaml
import time
import os
import pickle as pk
import torch
from fl.client import Client, params_norm, dp_scale_laplace, get_delta_norm_by_eps
from fl.server import Server
from basic.config import device, inversefed_setup, logger, args
import inversefed
from attack.dlg_utils import calculate_ssim
import traceback

from lpips import LPIPS
import torch.nn.functional as F
from torchvision import models
import numpy as np
from scipy.special import kl_div
import torch.nn as nn

import attacks
import utils

def fed_init(args, tb_writer, ds_name, shuffle):
    """new version based on InvGrad Data Processing"""
    # data_path = '/tmp/pycharm_project_455/fednfl-master/cifar10'
    # data_path = '/tmp/pycharm_project_196/fednfl-master/data/mnist'
    data_path = '/tmp/pycharm_project_344/fednfl-master/data/fmnist'

    # 1.init dataset
    if ds_name == "mnist":
        arch = 'LeNetZhuMNIST'
        trainset, testset = inversefed.build_mnist(data_path, False, True)
    elif ds_name == "fmnist":
        arch = 'LeNetZhuMNIST'
        trainset, testset = inversefed.build_fmnist(data_path, False, True)
    elif ds_name == "octmnist":
        arch = 'LeNetZhuMNIST'
        trainset, testset = inversefed.build_octmnist(data_path, False, True)
    elif ds_name == "organamnist":
        arch = 'LeNetZhuMNIST'
        trainset, testset = inversefed.build_organamnist(data_path, False, True)
    elif ds_name == "breastmnist":
        arch = 'LeNetZhuMNIST'
        trainset, testset = inversefed.build_breastmnist(data_path, False, True)
    elif ds_name == "cifar10":
        # arch = 'ConvNet64'
        arch = 'ResNet20'
        trainset, testset = inversefed.build_cifar10(data_path, False, True)
    else:
        raise Exception("Does not support for dataset:{args.dataset }.")

    # 2.client data partition
    trainsets, valsets, testsets = easy_data_partition(args.n_clients, trainset, testset, shuffle,
                                                           n_train_data_per_client = args.data_per_client)  # 1000


    # 3. init client & server object
    clients = []
    for i in range(args.n_clients):
        client = Client(i, ds_name, arch, trainsets[i], valsets[i], testsets[i], shuffle=args.shuffle,
                        apply_distortion=args.nfl.apply_distortion,
                        distortion_iter=args.nfl.distortion_iter, local_batch_iter=args.local_batch_iter,
                        model_optim=args.model_optim, zeta=args.nfl.zeta, lr=args.lr,
                        bs=args.batch_size, wd=args.weight_decay, le=args.local_epoch,
                        tb_writer=tb_writer)
        clients.append(client)
    server = Server(clients, ds_name, arch, args.checkpoint_dir)
    return clients, server


def get_nfl_bounds(args, model):
    if args.nfl.privacy == 'nfl':
        l = get_delta_norm_by_eps(args.nfl.eps, args.nfl.D, args.nfl.ca, args.nfl.c0, args.nfl.dlg_iter)
        # u = 2*clip
        u = 2*l
    else:
        sigma_dp = dp_scale_laplace(args.nfl.eps, args.nfl.clipDP, args.lr)
        dp_raw_l = args.lr * sigma_dp
        if args.nfl.opt_target == 'val':
            l = dp_raw_l
        else:
            total_norm, _ = params_norm([torch.ones_like(p) for p in model.parameters()], norm_type=1)
            logger.info('model params by L1 norm={}'.format(total_norm))
            l = dp_raw_l * total_norm.item()
        u = 2*l
    return l, u


def dlg_inv_grad(args, ground_truth, labels, original_model, input_grad, tb_writer, cid, bid, train_epoch_round, input_type='raw'):
    setup = inversefed_setup
    invgrad_config = dict(signed=True,
              boxed=True,
              cost_fn=args.nfl.cost_fn,  # sim
              indices='def',
              weights='equal',
              lr=args.nfl.dlg_lr,  # 0.1
              optim='adam',
              restarts=1,
              max_iterations=args.nfl.dlg_iter,  # 4000
              total_variation=args.nfl.tv_lambda,  # 1e-6
              init=args.nfl.dlg_img_init,  # randn / fname
              filter='none',
              lr_decay=True,
              scoring_choice='loss')

    dm = torch.as_tensor(eval(f'inversefed.consts.{args.dataset}_mean'), **setup)[:, None, None]
    ds = torch.as_tensor(eval(f'inversefed.consts.{args.dataset}_std'), **setup)[:, None, None]
    img_shape=(1, 28, 28) if 'mnist' in args.dataset else (3, 32, 32)

    if input_type == 'updates':
        rec_machine = inversefed.FedAvgReconstructor(original_model, (dm, ds), args.local_batch_iter, args.lr, invgrad_config,
                                                     use_updates=True, num_images=args.batch_size)
    else:  # raw, equiv
        rec_machine = inversefed.GradientReconstructor(original_model, (dm, ds), invgrad_config, num_images=args.batch_size)
    output, stats = rec_machine.reconstruct(input_grad, labels, img_shape=img_shape, tb_writer=tb_writer, cid=cid, bid=bid)

    test_mse = (output.detach() - ground_truth).pow(2).mean().item()
    feat_mse = (original_model(output.detach())- original_model(ground_truth)).pow(2).mean().item()
    test_psnr = float(inversefed.metrics.psnr(output, ground_truth, factor=1/ds))
    test_ssim = float(np.mean([calculate_ssim(torch.unsqueeze(x, 0).cpu(), torch.unsqueeze(x_rec, 0).cpu()).cpu().detach().item()
                        for x,x_rec in zip(ground_truth, output)]))

    # 确保设备一致性（GPU 或 CPU）
    device_dlg = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用VGG网络提取特征(KL散度）
    vgg_model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()

    # 确保修改后的模型仍然在正确的设备上
    vgg_model = vgg_model.to(device_dlg)

    # 将模型和输入张量移动到相同设备
    vgg_model = vgg_model.to(device_dlg)  # 将VGG模型移到同一设备
    lpips_model = LPIPS(net='vgg').to(device_dlg)  # 将LPIPS模型移到同一设备
    output = output.to(device_dlg)  # 将output张量移到同一设备
    ground_truth = ground_truth.to(device_dlg)  # 将ground_truth张量移到同一设备

    # -------------------- 添加 LPIPS 计算 ---------------------
    lpips_score = lpips_model(output, ground_truth)  # 计算恢复图像与真实图像之间的感知差异
    # 计算整个批次的平均LPIPS
    mean_lpips_score = lpips_score.mean().item()  # 获取整个批次的平均LPIPS分数

    # -------------------- 添加 KL散度计算 ---------------------
    def extract_features(model, image):
        features = []
        x = image  # 这里 x 就是批次的数据，4D 张量
        for layer in model:
            x = layer(x)
            features.append(x)

        # 选择最后一层的特征，或选择你感兴趣的某一层的特征
        return features[-1]  # 返回最后一层的特征图

    ground_truth_kl = ground_truth.squeeze(0)  # 将形状变为[8, 1, 28, 28]
    output_kl = output.squeeze(0)  # 如果 output 的形状也是这样，执行相同操作
    # 调整图像大小
    ground_truth_kl = F.interpolate(ground_truth_kl, size=(224, 224), mode='bilinear', align_corners=False)
    output_kl = F.interpolate(output_kl, size=(224, 224), mode='bilinear', align_corners=False)
    ground_truth_kl = ground_truth_kl.to(device_dlg)  # 将ground_truth_kl张量移到同一设备
    output_kl = output_kl.to(device_dlg)  # 将output_kl张量移到同一设备

    # 将单通道图像转换为三通道图像
    if ground_truth_kl.shape[1] == 1:  # 如果是单通道灰度图像
        ground_truth_kl = ground_truth_kl.repeat(1, 3, 1, 1)  # 复制单通道为3通道

    if output_kl.shape[1] == 1:  # 如果是单通道灰度图像
        output_kl = output_kl.repeat(1, 3, 1, 1)  # 复制单通道为3通道

    # 提取恢复图像和真实图像的VGG特征
    original_features = extract_features(vgg_model, ground_truth_kl)
    recovered_features = extract_features(vgg_model, output_kl)

    def compute_kl_divergence(features1, features2):
        # 将特征从Tensor转换为概率分布
        features1 = F.softmax(features1.view(features1.size(0), -1), dim=1)
        features2 = F.softmax(features2.view(features2.size(0), -1), dim=1)

        kl_scores = []
        for f1, f2 in zip(features1, features2):
            kl_score = kl_div(f1.cpu().detach().numpy(), f2.cpu().detach().numpy()).sum()
            kl_scores.append(kl_score)
        return np.mean(kl_scores)

    # 计算KL散度
    kl_divergence_score = compute_kl_divergence(original_features, recovered_features)

    output = output.clone().detach()
    output.mul_(ds).add_(dm).clamp_(0, 1)
    output = output.cpu()  # permute(0, 2, 3, 1)

    gt_img = ground_truth.detach().mul_(ds).add_(dm).clamp_(0, 1).cpu()

    # -------------------- 保存重建结果 ---------------------
    dlg_result_dict = dict(
        test_mse=test_mse, feat_mse=feat_mse, test_psnr=test_psnr, test_ssim=test_ssim,
        lpips_score=mean_lpips_score,kl_divergence_score = kl_divergence_score ,  # 新增的LPIPS和KL散度指标
        rec_img=output.numpy(), gt=gt_img.numpy()
    )
    # kl_divergence_score = kl_divergence_score   指标暂时不用

    pk.dump(dlg_result_dict, open(os.path.join(args.checkpoint_dir, f'dlg_result_E{train_epoch_round}.pkl'), 'wb'))

    # -------------------- TensorBoard记录 ---------------------
    tb_writer.add_scalar(f'C{cid}/dlg_B{bid}_mse', test_mse, train_epoch_round)
    tb_writer.add_scalar(f'C{cid}/dlg_B{bid}_feat_mse', feat_mse, train_epoch_round)
    tb_writer.add_scalar(f'C{cid}/dlg_B{bid}_psnr', test_psnr, train_epoch_round)
    tb_writer.add_scalar(f'C{cid}/dlg_B{bid}_ssim', test_ssim, train_epoch_round)
    tb_writer.add_scalar(f'C{cid}/dlg_B{bid}_lpips', mean_lpips_score, train_epoch_round)  # 记录LPIPS
    # tb_writer.add_scalar(f'C{cid}/dlg_B{bid}_kl_divergence', kl_divergence_score, train_epoch_round)  # 记录KL散度
    tb_writer.add_images(f'C{cid}/dlg_B{bid}_imgs', output, train_epoch_round)
    tb_writer.add_images(f'C{cid}/dlg_B{bid}_imgs_raw', gt_img, train_epoch_round)


def fed_train():
    # ===0.0 config
    logger.info("#" * 100)
    logger.info(str(args))
    set_seed(args.seed)

    # ===0.1 init tb writer
    tb_writer = SummaryWriter(args.checkpoint_dir)


    # ===1.create clients and server
    clients, server = fed_init(args, tb_writer, args.dataset, args.shuffle)
    logger.info([(k,p.detach().norm().cpu().item()) for k, p in clients[0].model.named_parameters()])

    if args.nfl.apply_distortion == 'nfl':
        l, u = get_nfl_bounds(args, clients[0].model)
    else:
        l, u = -1, -1
    args.nfl.l, args.nfl.u = l, u
    tb_writer.add_text('config', str(args.nfl))

    # ===2. FL process
    best_val_accs = [0.] * args.n_clients
    test_accs = [0.] * args.n_clients
    best_rounds = [-1] * args.n_clients
    client_loss_meter_list = [AvgMeter() for _ in range(args.n_clients)]
    comm_R = 0

    for train_epoch_round in range(args.global_epoch):
        # ===2.1 init for one global epoch
        logger.info("** Training Epoch {}, Communication Round:{} Start! **".format(train_epoch_round, comm_R))
        train_loader_list = list()
        for i in range(args.n_clients):
            train_loader_list.append(clients[i].trainloader)
            client_loss_meter_list[i].reset()


        # ===2.2 start one global epoch (n_batch)
        best_val_acc_global, best_te_acc_global = 0,0
        for batch_idx, clients_batch_data in enumerate(zip(*train_loader_list)):
            # ===2.2.1: client sequential train
            for client_idx, c_batch_data in enumerate(clients_batch_data):
                # === before
                if args.nfl.apply_dlg:
                    clients[client_idx].frozen_net(False)
                    original_model = clients[client_idx].get_copied_model()
                    original_model.train()

                # def visualize_images(original):
                #     # 假设original和perturbed是张图片，转换为numpy格式并归一化到0-1之间
                #     original = original.cpu().numpy().transpose(0, 2, 3, 1)[1]  # 仅选择第一张图像
                #
                #     # 对图片做裁剪处理，确保数值在[0,1]区间内
                #     original = np.clip(original, 0, 1)
                #
                #     # 绘制原始图像和扰动后图像
                #     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                #     axes[0].imshow(original)
                #     axes[0].set_title("Original Image")
                #     axes[0].axis('off')
                #     plt.show()

                # === local train @ client idx
                x, y = c_batch_data[0].to(device), c_batch_data[1].to(device)

                # # 可视化原始图像和添加扰动后的图像
                # visualize_images(x)  # 仅选择第一张图像进行展示

                if train_epoch_round <= args.nfl.warm_up_rounds - 1:
                    loss_list, acc_list, grad_list, noisy_grad_list = clients[client_idx]. \
                        perform_nfl_train(x, y, comm_R, l, u, warming_up=True,
                                          u_loss_type=args.nfl.u_loss_type,
                                          nfl_lba=args.nfl.lba, clipDP=args.nfl.clipDP,
                                          privacy_measure=args.nfl.privacy, optimized_target=args.nfl.opt_target)
                else:
                    if 'dp' in args.nfl.apply_distortion:
                        mecha = args.nfl.apply_distortion[3:]
                        loss_list, acc_list, grad_list, noisy_grad_list = clients[client_idx].\
                            perform_dp_train(x, y, comm_R,
                                            clip=args.nfl.clip, mechanism=mecha,
                                            eps=args.nfl.eps, clip_level=args.nfl.clipL,
                                            element_wise_rand=args.nfl.element_wise_rand)
                    else:
                        loss_list, acc_list, grad_list, noisy_grad_list = clients[client_idx]. \
                            perform_nfl_train(x, y, comm_R, l, u, warming_up=False,
                                            u_loss_type=args.nfl.u_loss_type,
                                            nfl_lba=args.nfl.lba, clipDP=args.nfl.clipDP,
                                            privacy_measure=args.nfl.privacy, optimized_target=args.nfl.opt_target,
                                            element_wise_rand=args.nfl.element_wise_rand,
                                            dp_upratio=args.nfl.dp_upratio)

                for loss, acc in zip(loss_list, acc_list):
                    client_loss_meter_list[client_idx].update(loss)
                # record best val
                if batch_idx % 5 == 0:
                    val_acc = clients[client_idx].local_val()
                    tb_writer.add_scalar(f'C{client_idx}/val_acc', val_acc, comm_R)
                    if val_acc > best_val_accs[client_idx]:
                        best_val_accs[client_idx] = val_acc
                        test_accs[client_idx] = clients[client_idx].local_test()
                        best_rounds[client_idx] = train_epoch_round

                # === after (挑选某些batch的图片进行DLG攻击)
                if args.nfl.apply_dlg and batch_idx == 0 and client_idx == 0 and train_epoch_round in args.nfl.dlg_attack_epochs:
                    if args.nfl.dlg_know_grad == 'equiv':  # W_{t-1} - Wt / lr
                        updated_model = clients[client_idx].get_copied_model()
                        equiv_raw_grad = list((om.detach().clone() - tm.detach().clone()) / args.lr for om, tm in
                                    zip(original_model.parameters(), updated_model.parameters()))
                    elif args.nfl.dlg_know_grad == 'raw':
                        equiv_raw_grad = noisy_grad_list[0] if args.nfl.apply_distortion != 'no' else grad_list[0]
                        # equiv_raw_grad = noisy_grad_list[0]
                    elif args.nfl.dlg_know_grad == 'updates':  # # W_t - W_{t-1}
                        updated_model = clients[client_idx].get_copied_model()
                        equiv_raw_grad = list((um.detach().clone() - om.detach().clone()) for om, um in
                                              zip(original_model.parameters(), updated_model.parameters()))
                    else:
                        raise NotImplementedError('{} not valid'.format(args.nfl.dlg_know_grad))

                    dlg_inv_grad(args, x, y, original_model, equiv_raw_grad,
                                 tb_writer, client_idx, batch_idx, train_epoch_round,
                                 args.nfl.dlg_know_grad)
                    # # 保存dlg的现场
                    # dlg_save_dict = dict(
                    #     x = x, y = y,
                    #     ori_model = original_model.state_dict(),
                    #     equiv_raw_grad = equiv_raw_grad,
                    #     local_model_lr=args.lr,
                    #     cid=client_idx, batch_idx=batch_idx,
                    #     round=comm_R,
                    #     gE=train_epoch_round,
                    # )
                    # save_path = os.path.join(args.checkpoint_dir, 'dlg')
                    # if not os.path.exists(save_path):
                    #     os.makedirs(save_path)
                    # torch.save(dlg_save_dict, os.path.join(save_path, f'C{client_idx}_E{train_epoch_round}_B{batch_idx}.pkl'))

            # ===2.2.2 server aggregation
            server.receive()
            server.send()
            comm_R += 1

        val_acc_global, te_acc_global = server.eval_global('val'), server.eval_global('test')
        tb_writer.add_scalar(f'global/val_acc', val_acc_global, train_epoch_round)
        tb_writer.add_scalar(f'global/test_acc', te_acc_global, train_epoch_round)

        # 将每轮的验证和测试准确率记录到日志文件
        metrics_file_path = os.path.join(args.checkpoint_dir, 'all_metrics_log.txt')
        with open(metrics_file_path, 'a') as metrics_file:
            metrics_file.write(f"{train_epoch_round},{val_acc_global},{te_acc_global}\n")

        if val_acc_global > best_val_acc_global:
            best_val_acc_global, best_te_acc_global = val_acc_global, te_acc_global
            with open(os.path.join(args.checkpoint_dir, 'best_metric.txt'), 'w') as f:
                f.writelines(['{},{},{}'.format(train_epoch_round, best_val_acc_global, best_te_acc_global)])
            torch.save(server.global_net, os.path.join(args.checkpoint_dir, 'server_best_global.pkl'))

        # ===2.3 end one global epoch
        # ===2.3.1 check early stop
        early_stop = True
        for i in range(args.n_clients):
            if train_epoch_round <= best_rounds[i] + args.early_stop_rounds:
                early_stop = False
                break
        if early_stop:
            logger.info("early stoped at epoch{}, communication round:{}".format(train_epoch_round, comm_R))
            break
        # ===2.3.2 report metric for each global epoch
        logger.info("train epoch round:{}, communication round:{}".format(train_epoch_round, comm_R))
        for i in range(args.n_clients):
            loss_avg = client_loss_meter_list[i].get()
            logger.info("client:%2d, test acc:%2.6f, best epoch:%2d, loss:%2.6f" % (i, test_accs[i], best_rounds[i], loss_avg))

    # ===3. End of FL
    logger.info("** Federated Learning Finish! **")
    for i in range(args.n_clients):
        logger.info("client:%2d, test acc:%2.6f, best epoch:%2d" % (i, test_accs[i], best_rounds[i]))


if __name__ == "__main__":
    try:
        fed_train()
    except Exception as e:
        msg = traceback.format_exc()
        logger.exception(msg)
