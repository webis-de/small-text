from contextlib import contextmanager
from packaging.version import parse, Version

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


@contextmanager
def inference_mode():
    inference_mode_decorator = torch.inference_mode \
        if parse(torch.__version__) >= Version('1.9.0') else torch.no_grad

    with inference_mode_decorator():
        yield
