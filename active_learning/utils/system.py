import os

TMP_DIR_VARIABLE = 'ALL_TMP'


def get_tmp_dir_base():

    if TMP_DIR_VARIABLE in os.environ:
        return os.environ[TMP_DIR_VARIABLE]

    return None
