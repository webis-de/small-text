from enum import Enum
from typing import NamedTuple

from small_text.utils.system import get_offline_mode


class ModelLoadingStrategy(Enum):
    DEFAULT = 'default'
    ALWAYS_LOCAL = 'always-local'
    ALWAYS_DOWNLOAD = 'always-download'


class PretrainedModelLoadingArguments(NamedTuple):
    force_download: bool = False
    local_files_only: bool = False


def get_default_model_loading_strategy() -> ModelLoadingStrategy:
    if get_offline_mode() is True:
        return ModelLoadingStrategy.ALWAYS_LOCAL

    return ModelLoadingStrategy.DEFAULT
