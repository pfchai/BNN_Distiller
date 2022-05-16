# -*- coding: utf-8 -*-

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image


class CIFAR10Instance(datasets.CIFAR10):
    """
    CIFAR10 Instance Dataset.
    """
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CIFAR100Instance(datasets.CIFAR100):
    """
    CIFAR100 Instance Dataset.
    """
    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def load_data(dataset='cifar10', data_path='/data', batch_size=256, batch_size_test=256, num_workers=0, is_instance=False):
    # load data
    param = {
        'cifar10': {'name': datasets.CIFAR10, 'size': 32, 'normalize': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]},
        'cifar100': {'name': datasets.CIFAR100, 'size': 32, 'normalize': [(0.507, 0.487, 0.441), (0.267, 0.256, 0.276)]},
    }
    data = param[dataset]

    transform1 = transforms.Compose([
        transforms.RandomCrop(data['size'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*data['normalize']),
    ])

    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*data['normalize']),
    ])

    if is_instance:
        if dataset == 'cifar100':
            trainset = CIFAR100Instance(root=data_path, train=True, transform=transform1)
        else:
            trainset = CIFAR10Instance(root=data_path, train=True, transform=transform1)
        n_data = len(trainset)
    else:
        trainset = data['name'](root=data_path, train=True, download=False, transform=transform1)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = data['name'](root=data_path, train=False, download=False, transform=transform2)
    testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=num_workers, pin_memory=True)

    if is_instance:
        return trainloader, testloader, n_data
    else:
        return trainloader, testloader
