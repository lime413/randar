import os
from torch.utils import data
from torchvision import datasets, transforms
import torch
import numpy as np

class CIFAR10WithIndex(data.Dataset):
    """
    Wraps torchvision.datasets.CIFAR10 to return (image, label, index)
    """
    def __init__(self, root, train=True, transform=None, download=False):
        self.ds = datasets.CIFAR10(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )
        self.nb_classes = 10

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        img, label = self.ds[index]
        return img, label, index