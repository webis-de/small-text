import torch

from contextlib import contextmanager
from small_text.utils.annotations import deprecated


@deprecated(deprecated_in='1.1.0', to_be_removed_in='2.0.0')
@contextmanager
def default_tensor_type(tensor_type):
    default_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    torch.set_default_tensor_type(tensor_type)
    yield
    torch.set_default_tensor_type(default_type)


def _assert_layer_exists(module, layer_name):
    if layer_name not in dict(module.named_modules()):
        raise ValueError(f'Given layer "{layer_name}" does not exist in the model!')
