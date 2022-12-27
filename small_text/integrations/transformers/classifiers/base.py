from enum import Enum
from typing import NamedTuple


class ModelLoadingStrategy(Enum):
    DEFAULT = 'default'
    ALWAYS_LOCAL = 'always-local'
    ALWAYS_DOWNLOAD = 'always-download'


class PretrainedModelLoadingArguments(NamedTuple):
    force_download: bool = False
    local_files_only: bool = False
