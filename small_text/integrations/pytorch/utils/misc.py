import warnings
import torch

from contextlib import contextmanager


@contextmanager
def default_tensor_type(tensor_type):
    warnings.warn('default_tensor_type() is deprecated since 1.1.0 and will be removed in 2.0.0',
                  DeprecationWarning,
                  stacklevel=2)
    default_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    torch.set_default_tensor_type(tensor_type)
    yield
    torch.set_default_tensor_type(default_type)


def assert_layer_exists(module, layer_name):
    if layer_name not in dict(module.named_modules()):
        raise ValueError(f'Given layer "{layer_name}" does not exist in the model!')


def early_stopping_deprecation_warning(early_stopping_no_improvement, early_stopping_acc):
    if early_stopping_no_improvement != 5:
        warnings.warn(
            'early_stopping_no_improvement class is deprecated since 1.1.0 '
            'and will be removed in 2.0.0. Please use the "early_stopping" kwarg in fit () '
            'instead.',
            DeprecationWarning,
            stacklevel=2)
    if early_stopping_acc != -1:
        warnings.warn(
            'early_stopping_acc class is deprecated since 1.1.0 '
            'and will be removed in 2.0.0. Please use the "early_stopping" kwarg in fit () '
            'instead.',
            DeprecationWarning,
            stacklevel=2)
