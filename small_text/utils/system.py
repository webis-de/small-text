import importlib
import os

TMP_DIR_VARIABLE = 'SMALL_TEXT_TEMP'


def get_tmp_dir_base():

    if TMP_DIR_VARIABLE in os.environ:
        return os.environ[TMP_DIR_VARIABLE]

    return None


def is_pytorch_available():
    try:
        importlib.import_module('torch')
        return True
    except ImportError:
        return False


def is_transformers_available():
    try:
        importlib.import_module('transformers')
        return True
    except ImportError:
        return False
