import torch
import torch.nn as nn

from contextlib import contextmanager
from packaging.version import parse, Version


DROPOUT_MODULES = (
    nn.Dropout,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


@contextmanager
def enable_dropout(model):
    modules = dict({name: module for name, module in model.named_modules()})
    altered_modules = []

    for name, mod in modules.items():
        if isinstance(mod, DROPOUT_MODULES) and mod.training is False:
            mod.train()
            altered_modules.append(name)
    yield
    for name in altered_modules:
        modules[name].eval()


def _assert_layer_exists(module, layer_name):
    if layer_name not in dict(module.named_modules()):
        raise ValueError(f'Given layer "{layer_name}" does not exist in the model!')


def _compile_if_possible(model, compile_model: bool = True):
    if compile_model and parse(torch.__version__) >= Version('2.0.0'):
        model = torch.compile(model)
    return model
