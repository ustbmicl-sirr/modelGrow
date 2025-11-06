import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def random_subset(data, nsamples, seed):
    set_seed(seed)
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    return Subset(data, idx[:nsamples])


def get_cifar10():
    img_size = 32
    train_transform = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    ])
    test_transform = transforms.Compose([
        transforms.RandomCrop(img_size, padding=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    return train_dataset, test_dataset

DEFAULT_PATHS = {
    'cifar10': [
        './data'  # Assuming CIFAR10 is stored in ./data
    ]
}

def get_loaders(
    name, path='', batchsize=-1, workers=8, nsamples=1024, seed=0,
    noaug=False
):

    if not path:
        for path in DEFAULT_PATHS[name]:
            if os.path.exists(path):
                break

    if name == 'cifar10':
        if batchsize == -1:
            batchsize = 128
        train_data, test_data = get_cifar10()
        train_data_sample = random_subset(train_data, nsamples, seed)
    trainloader = DataLoader(train_data, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=True)
    trainloader_sample = DataLoader(train_data_sample, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=False)

    return trainloader, trainloader_sample, testloader
