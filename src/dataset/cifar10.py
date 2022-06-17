import os

import torchvision.datasets
import torchvision.transforms as T


def get_cifar10_data(conf, data_dir):
    transform_train = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    transform_test = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        os.path.join(data_dir, conf.dataset.path),
        train=True,
        transform=transform_train,
        download=True,
    )
    valid_set = torchvision.datasets.CIFAR10(
        os.path.join(data_dir, conf.dataset.path),
        train=False,
        transform=transform_test,
        download=True,
    )

    return train_set, valid_set