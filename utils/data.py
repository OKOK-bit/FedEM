import warnings

import numpy as np
import os
import pickle
import os.path
import torchvision.transforms as transforms
from torchvision.datasets.mnist import read_label_file, read_image_file

from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, Optional, Union, Tuple, Dict
from pathlib import Path

import torch
import torchvision




class ElementWiseTransform():
    def __init__(self, trans=None):
        self.trans = trans

    def __call__(self, x):
        if self.trans is None: return x
        # print(x.shape)  # 检查图像批次的形状，应该是 [batch_size, 3, 32, 32]
        return torch.cat([self.trans(xx.view(1, *xx.shape)) for xx in x] )


class IndexedTensorDataset():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        ''' transform HWC pic to CWH pic '''
        x = torch.tensor(x, dtype=torch.float32).permute(2,0,1)
        return x, y, idx

    def __len__(self):
        return len(self.x)


class Dataset():
    def __init__(self, x, y, transform=None, fitr=None):
        self.x = x
        self.y = y
        self.transform = transform
        self.fitr = fitr

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        ''' low pass filtering '''
        if self.fitr is not None:
            x = self.fitr(x)

        ''' data augmentation '''
        if self.transform is not None:
            x = self.transform( Image.fromarray(x) )

        return x, y

    def __len__(self):
        return len(self.x)


class IndexedDataset():
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.ii = np.array( range(len(x)), dtype=np.int64 )
        self.transform = transform

    def __getitem__(self, idx):
        x, y, ii = Image.fromarray(self.x[idx]), self.y[idx], self.ii[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y, ii

    def __len__(self):
        return len(self.x)


def datasetCIFAR10(root='./path', train=True, transform=None):
    return torchvision.datasets.CIFAR10(root=root, train=train,
                        transform=transform, download=True)

def datasetCIFAR100(root='./path', train=True, transform=None):
    return torchvision.datasets.CIFAR100(root=root, train=train,
                        transform=transform, download=True)

def datasetTinyImageNet(root='./path', train=True, transform=None):
    if train: root = os.path.join(root, 'tiny-imagenet_train.pkl')
    else: root = os.path.join(root, 'tiny-imagenet_val.pkl')
    with open(root, 'rb') as f:
        dat = pickle.load(f)
    return Dataset(dat['data'], dat['targets'], transform)


def datasetoctmnist(root, train=True, transform=None):
    return OCTMNIST(root=root, train=train, download=False, transform=transforms.ToTensor())

class Loader():
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False, num_workers=4):
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        self.iterator = None

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)

        try:
            samples = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            samples = next(self.iterator)

        return samples


#下面为OCTMNIST类相关前置类以及声明

class OCTMNIST(torch.utils.data.Dataset):
    """OctMNIST Dataset (custom dataset for octmnist classification task).

    Args:
        root (str or `pathlib.Path`): Root directory of dataset where octmnist/raw/train-images-idx3-ubyte
            and octmnist/raw/t10k-images-idx3-ubyte exist.
        train (bool, optional): If True, creates dataset from `train-images-idx3-ubyte`,
            otherwise from `t10k-images-idx3-ubyte.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, `transforms.RandomCrop`
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

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
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
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



