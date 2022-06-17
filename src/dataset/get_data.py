from src.dataset.cifar10 import get_cifar10_data


def get_train_data(conf, data_dir):
    assert conf.dataset.name == 'cifar10'
    assert conf.dataset.resolution == 32

    if conf.dataset.name == "cifar10":
        train_set, valid_set = get_cifar10_data(conf, data_dir)

    return train_set, valid_set
