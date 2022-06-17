import torch


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        x = x.to(device)
    elif isinstance(x, list):
        x = [to_device(xi, device) for xi in x]
    elif isinstance(x, tuple):
        x = tuple([to_device(xi, device) for xi in x])
    elif isinstance(x, dict):
        x = {k: to_device(v, device) for k,v in x.items()}
    elif isinstance(x, (bool, int, float, str)):
        pass
    elif x is None:
        pass
    else:
        raise ValueError(f'to_device does not support type {type(x)}')

    return x


def to_cpu(x):
    return to_device(x, 'cpu')


def set_requires_grad(model, flag: bool):
    assert isinstance(flag, bool)

    for param in model.parameters():
        param.requires_grad = flag


def trainable_parameters(model):
    return iter(p for p in model.parameters() if p.requires_grad)


def count_trainable_parameters(model):
    params = trainable_parameters(model)

    count = 0
    for p in params:
        count += p.numel()
    return count
