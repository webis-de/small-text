import torch

from contextlib import contextmanager


@contextmanager
def default_tensor_type(tensor_type):
    default_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    torch.set_default_tensor_type(tensor_type)
    yield
    torch.set_default_tensor_type(default_type)


def assert_layer_exists(module, layer_name):
    if layer_name not in dict(module.named_modules()):
        raise ValueError(f'Given layer "{layer_name}" does not exist in the model!')
