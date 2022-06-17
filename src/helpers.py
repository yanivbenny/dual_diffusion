def exists(val):
    return val is not None


def default(val, x):
    return val if exists(val) else x


def maybe_str2bool(x: str):
    assert isinstance(x, str), f'str2bool can only interpret strings (got {type(x)})'
    if x.lower() in ('1', 'y', 'true'):
        return True
    if x.lower() in ('0', 'n', 'false'):
        return False
    else:
        return x


def str2bool(x: str):
    res = maybe_str2bool(x)
    if not isinstance(res, bool):
        raise ValueError(f'str2bool cannot booleanize value "{x}"')
    return res