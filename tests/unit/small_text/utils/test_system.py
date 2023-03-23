import os
import unittest

from small_text.utils.system import (
    OFFLINE_MODE_VARIABLE,
    PROGRESS_BARS_VARIABLE,
    TMP_DIR_VARIABLE,
    get_offline_mode,
    get_progress_bars_default,
    get_tmp_dir_base
)


class SystemUtilsTest(unittest.TestCase):

    def test_get_offline_mode(self):
        self.assertFalse(get_offline_mode())

    def test_get_offline_mode_true(self):
        os.environ[OFFLINE_MODE_VARIABLE] = '1'
        self.assertTrue(get_offline_mode())

    def test_get_progress_bars_default(self):
        self.assertEqual('tqdm', get_progress_bars_default())

    def test_get_progress_bars_default_custom(self):
        custom_pbars = '0'
        os.environ[PROGRESS_BARS_VARIABLE] = custom_pbars
        self.assertEqual(None, get_progress_bars_default())

    def test_get_tmp_dir_base(self):
        self.assertIsNone(get_tmp_dir_base())

    def test_get_tmp_dir_base_custom(self):
        custom_tmp_dir = '/tmp'
        os.environ[TMP_DIR_VARIABLE] = custom_tmp_dir
        self.assertEqual(custom_tmp_dir, get_tmp_dir_base())
