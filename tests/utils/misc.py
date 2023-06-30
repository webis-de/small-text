import os

import numpy as np


def increase_dense_labels_safe(ds):
    """Increase the labels without leaving the range of the target_labels property.
    The only purpose of this operation is to alter the labels so we can check for a change later."""

    # modulo needs not be used when single index result is 0
    if ds.y.max() == 0:
        ds.y = ds.y + 1
    else:
        ds.y = (ds.y + 1) % (ds.y.max() + 1)
    return ds


def random_seed(func_or_class=None, seed=42, set_torch_seed=False):
    import inspect
    from functools import wraps

    assert 0 <= seed <= 2**32-1

    if func_or_class is not None:
        if not inspect.isclass(func_or_class) and not inspect.isfunction(func_or_class):
            raise ValueError('The @random_seed decorator requires a function or class')

    def _decorator(func_or_class):
        @wraps(func_or_class)
        def wrapper(*args, **kwargs):
            os.environ['PYTHONHASHSEED'] = str(seed)
            if set_torch_seed:
                import torch
                torch.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            np.random.seed(seed)

            return func_or_class(*args, **kwargs)
        return wrapper
    return _decorator(func_or_class) if callable(func_or_class) else _decorator
