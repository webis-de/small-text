import importlib
import os

from typing import Union


OFFLINE_MODE_VARIABLE = 'SMALL_TEXT_OFFLINE'


PROGRESS_BARS_VARIABLE = 'SMALL_TEXT_PROGRESS_BARS'


TMP_DIR_VARIABLE = 'SMALL_TEXT_TEMP'


def get_offline_mode() -> bool:

    if OFFLINE_MODE_VARIABLE in os.environ:
        return True

    return False


def get_show_progress_bar_default() -> bool:
    if PROGRESS_BARS_VARIABLE in os.environ:
        if os.environ[PROGRESS_BARS_VARIABLE] == '0' or os.environ[PROGRESS_BARS_VARIABLE].lower() == 'false':
            return False

    return True


def get_tmp_dir_base() -> Union[str, None]:

    if TMP_DIR_VARIABLE in os.environ:
        return os.environ[TMP_DIR_VARIABLE]

    return None


def is_pytorch_available() -> bool:
    try:
        importlib.import_module('torch')
        return True
    except ImportError:
        return False


def is_transformers_available() -> bool:
    try:
        importlib.import_module('transformers')
        return True
    except ImportError:
        return False
