"""Repeatable code parts concerning data loading."""

import os.path

import numpy as np
from matplotlib import pyplot as plt
from torchvision.datasets.mnist import read_label_file, read_image_file


from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, Optional, Union, Tuple, Dict
from pathlib import Path
import warnings


import torch
import torchvision
import torchvision.transforms as transforms

import os

from ..consts import *

from .data import _build_bsds_sr, _build_bsds_dn
from .loss import Classification, PSNR

class OCTMNIST(Dataset):

    # 需要根据 octmnist 数据集的来源修改下载链接
    mirrors = [
        "http://example.com/octmnist/ ",
        "https://zenodo.org/record/4269852/files/octmnist.npz?download=1"  # <-- 修改为实际的 octmnist 数据集下载链接
    ]

    # 需要根据 octmnist 数据集的文件名修改
    resources = [
        ("train-images-idx3-ubyte.gz", "c68d92d5b585d8d81f7112f81e2d0842"),  # <-- 修改为 octmnist 的文件和对应的 MD5 值
        ("train-labels-idx1-ubyte.gz", "c68d92d5b585d8d81f7112f81e2d0842"),
        ("test-images-idx3-ubyte.gz", "c68d92d5b585d8d81f7112f81e2d0842"),
        ("test-labels-idx1-ubyte.gz", "c68d92d5b585d8d81f7112f81e2d0842"),
    ]

    training_file = "training.pt"
    test_file = "test.pt"

    # 根据 octmnist 数据集的类别修改
    classes = [
        "0 - choroidal neovascularization",  # <-- 修改为 octmnist 数据集的实际类别
        "1 - diabetic macular edema",
        "2 - drusen",
        "3 - normal",
    ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
            self,
            root: Union[str, Path],
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        # **高亮**：正确调用父类的构造函数，初始化父类
        super().__init__()

        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.target_transform = target_transform

        # 检查原始数据是否已经存在
        if not self._check_exists():
            raise RuntimeError("Dataset not found. Please make sure the dataset is placed in the correct directory")

        self.data, self.targets = self._load_data()

    def _check_exists(self):
        """检查原始数据是否存在"""
        return all(
            os.path.exists(os.path.join(self.raw_folder, filename))
            for filename, _ in self.resources
        )

    def _load_data(self):
        """加载数据"""
        image_file = f"{'train' if self.train else 'test'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))  # <-- 确保这个函数读取的是正确格式的数据

        label_file = f"{'train' if self.train else 'test'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))  # <-- 确保这个函数正确读取标签

        return data, targets

    def __getitem__(self, index: int) -> Tuple:
        img, target = self.data[index], int(self.targets[index])

        # Ensure it's a PIL Image for consistency with other datasets
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root,  "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

class ORGANAMNIST(Dataset):

    # 需要根据 organamnist 数据集的来源修改下载链接
    mirrors = [
        "http://example.com/organamnist/ ",
        "https://zenodo.org/record/4269852/files/organmnist_axial.npz?download=1"  # <-- 修改为实际的 organamnist 数据集下载链接
    ]

    # 需要根据 octmnist 数据集的文件名修改
    resources = [
        ("train-images-idx3-ubyte.gz", "866b832ed4eeba67bfb9edee1d5544e6"),  # <-- 修改为 octmnist 的文件和对应的 MD5 值
        ("train-labels-idx1-ubyte.gz", "866b832ed4eeba67bfb9edee1d5544e6"),
        ("test-images-idx3-ubyte.gz", "866b832ed4eeba67bfb9edee1d5544e6"),
        ("test-labels-idx1-ubyte.gz", "866b832ed4eeba67bfb9edee1d5544e6"),
    ]

    training_file = "training.pt"
    test_file = "test.pt"

    # 根据 organamnist 数据集的类别修改
    classes = [
        "0 - normal",
        "1 - pneumonia",
    ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
            self,
            root: Union[str, Path],
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        # **高亮**：正确调用父类的构造函数，初始化父类
        super().__init__()

        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.target_transform = target_transform

        # 检查原始数据是否已经存在
        if not self._check_exists():
            raise RuntimeError("Dataset not found. Please make sure the dataset is placed in the correct directory")

        self.data, self.targets = self._load_data()

    def _check_exists(self):
        """检查原始数据是否存在"""
        return all(
            os.path.exists(os.path.join(self.raw_folder, filename))
            for filename, _ in self.resources
        )

    def _load_data(self):
        """加载数据"""
        image_file = f"{'train' if self.train else 'test'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))  # <-- 确保这个函数读取的是正确格式的数据

        label_file = f"{'train' if self.train else 'test'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))  # <-- 确保这个函数正确读取标签

        return data, targets

    def __getitem__(self, index: int) -> Tuple:
        img, target = self.data[index], int(self.targets[index])

        # Ensure it's a PIL Image for consistency with other datasets
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root,  "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

class BREASTMNIST(Dataset):

    # 需要根据 breastmnist 数据集的来源修改下载链接
    mirrors = [
        "http://example.com/breastmnist/ ",
        "https://zenodo.org/record/4269852/files/breastmnist.npz?download=1"  # <-- 修改为实际的 breastmnist 数据集下载链接
    ]

    # 需要根据 breastmnist 数据集的文件名修改
    resources = [
        ("train-images-idx3-ubyte.gz", "750601b1f35ba3300ea97c75c52ff8f6"),  # <-- 修改为 octmnist 的文件和对应的 MD5 值
        ("train-labels-idx1-ubyte.gz", "750601b1f35ba3300ea97c75c52ff8f6"),
        ("test-images-idx3-ubyte.gz", "750601b1f35ba3300ea97c75c52ff8f6"),
        ("test-labels-idx1-ubyte.gz", "750601b1f35ba3300ea97c75c52ff8f6"),
    ]

    training_file = "training.pt"
    test_file = "test.pt"

    # 根据 breastmnist 数据集的类别修改
    classes = [
        "0 - malignant",
        "1 - normal",
    ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
            self,
            root: Union[str, Path],
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        # **高亮**：正确调用父类的构造函数，初始化父类
        super().__init__()

        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.target_transform = target_transform

        # 检查原始数据是否已经存在
        if not self._check_exists():
            raise RuntimeError("Dataset not found. Please make sure the dataset is placed in the correct directory")

        self.data, self.targets = self._load_data()

    def _check_exists(self):
        """检查原始数据是否存在"""
        return all(
            os.path.exists(os.path.join(self.raw_folder, filename))
            for filename, _ in self.resources
        )

    def _load_data(self):
        """加载数据"""
        image_file = f"{'train' if self.train else 'test'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))  # <-- 确保这个函数读取的是正确格式的数据

        label_file = f"{'train' if self.train else 'test'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))  # <-- 确保这个函数正确读取标签

        return data, targets

    def __getitem__(self, index: int) -> Tuple:
        img, target = self.data[index], int(self.targets[index])

        # Ensure it's a PIL Image for consistency with other datasets
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root,  "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

def construct_dataloaders(dataset, defs, shuffle=True, normalize=True):
    """Return a dataloader with given dataset and augmentation, normalize data?."""
    # data_path='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-adrec/liwenjie37/others/FedNFL/data'
    data_path = '/tmp/pycharm_project_344/fednfl-master/data/fmnist'            # 医学数据集
    path = os.path.expanduser(data_path)

    if dataset == 'CIFAR10':
        trainset, validset = _build_cifar10(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'CIFAR100':
        trainset, validset = _build_cifar100(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'MNIST':
        trainset, validset = _build_mnist(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'FMNIST':
        trainset, validset = _build_fmnist(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'MNIST_GRAY':
        trainset, validset = _build_mnist_gray(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'ImageNet':
        trainset, validset = _build_imagenet(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'BSDS-SR':
        trainset, validset = _build_bsds_sr(path, defs.augmentations, normalize, upscale_factor=3, RGB=True)
        loss_fn = PSNR()
    elif dataset == 'BSDS-DN':
        trainset, validset = _build_bsds_dn(path, defs.augmentations, normalize, noise_level=25 / 255, RGB=False)
        loss_fn = PSNR()
    elif dataset == 'BSDS-RGB':
        trainset, validset = _build_bsds_dn(path, defs.augmentations, normalize, noise_level=25 / 255, RGB=True)
        loss_fn = PSNR()
    elif dataset == 'octmnist':
        trainset, validset = _build_octmnist(path, defs.augmentations, normalize)
        # loss_fn = Classification()
    elif dataset == 'organamnist':
        trainset, validset = _build_organamnist(path, defs.augmentations, normalize)
        # loss_fn = Classification()
    elif dataset == 'breastmnist':
        trainset, validset = _build_breastmnist(path, defs.augmentations, normalize)
        # loss_fn = Classification()

    if MULTITHREAD_DATAPROCESSING:
        num_workers = min(torch.get_num_threads(), MULTITHREAD_DATAPROCESSING) if torch.get_num_threads() > 1 else 0
    else:
        num_workers = 0

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(defs.batch_size, len(trainset)),
                                              shuffle=shuffle, drop_last=True, num_workers=num_workers, pin_memory=PIN_MEMORY)
    validloader = torch.utils.data.DataLoader(validset, batch_size=min(defs.batch_size, len(trainset)),
                                              shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)

    return loss_fn, trainloader, validloader


def _build_cifar10(data_path, augmentations=True, normalize=True):
    """Define CIFAR-10 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if cifar10_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar10_mean, cifar10_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_cifar100(data_path, augmentations=True, normalize=True):
    """Define CIFAR-100 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if cifar100_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar100_mean, cifar100_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_mnist(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if mnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = mnist_mean, mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_breastmnist(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = BREASTMNIST(root=data_path, train=True, download=False, transform=transforms.ToTensor())
    validset = BREASTMNIST(root=data_path, train=False, download=False, transform=transforms.ToTensor())

    # # Step 2: 可视化原始图像（仅 ToTensor，未 Normalize）
    # loader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=False)
    # images, labels = next(iter(loader))  # [B, 1, 28, 28]
    # grid = torchvision.utils.make_grid(images, nrow=4, normalize=False, pad_value=1)
    #
    # plt.figure(figsize=(10, 3))
    # plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    # plt.title("Raw OCTMNIST Images (Before Normalize and Augmentation)")
    # plt.axis('off')
    # plt.show()

    if breastmnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = breastmnist_mean, breastmnist_std

    print(data_mean,data_std)   #打印均值和方差

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_octmnist(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = OCTMNIST(root=data_path, train=True, download=False, transform=transforms.ToTensor())
    validset = OCTMNIST(root=data_path, train=False, download=False, transform=transforms.ToTensor())

    # # Step 2: 可视化原始图像（仅 ToTensor，未 Normalize）
    # loader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=False)
    # images, labels = next(iter(loader))  # [B, 1, 28, 28]
    # grid = torchvision.utils.make_grid(images, nrow=4, normalize=False, pad_value=1)
    #
    # plt.figure(figsize=(10, 3))
    # plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    # plt.title("Raw OCTMNIST Images (Before Normalize and Augmentation)")
    # plt.axis('off')
    # plt.show()

    if octmnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = octmnist_mean, octmnist_std

    # print(data_mean,data_std)   #打印均值和方差

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_organamnist(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = ORGANAMNIST(root=data_path, train=True, download=False, transform=transforms.ToTensor())
    validset = ORGANAMNIST(root=data_path, train=False, download=False, transform=transforms.ToTensor())

    # # Step 2: 可视化原始图像（仅 ToTensor，未 Normalize）
    # loader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=False)
    # images, labels = next(iter(loader))  # [B, 1, 28, 28]
    # grid = torchvision.utils.make_grid(images, nrow=4, normalize=False, pad_value=1)
    #
    # plt.figure(figsize=(10, 3))
    # plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    # plt.title("Raw OCTMNIST Images (Before Normalize and Augmentation)")
    # plt.axis('off')
    # plt.show()

    if organamnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = organamnist_mean, organamnist_std

    print(data_mean,data_std)   #打印均值和方差

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_mnist_gray(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if mnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = mnist_mean, mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_fmnist(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if fmnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = fmnist_mean, fmnist_std
    print(data_mean, data_std)
    
    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_imagenet(data_path, augmentations=True, normalize=True):
    """Define ImageNet with everything considered."""
    # Load data
    trainset = torchvision.datasets.ImageNet(root=data_path, split='train', transform=transforms.ToTensor())
    validset = torchvision.datasets.ImageNet(root=data_path, split='val', transform=transforms.ToTensor())

    if imagenet_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = imagenet_mean, imagenet_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _get_meanstd(dataset):
    cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()
    return data_mean, data_std
