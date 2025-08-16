import os
import unittest

from unittest.mock import patch

from small_text.utils.system import (
    OFFLINE_MODE_VARIABLE,
    PROGRESS_BARS_VARIABLE,
    TMP_DIR_VARIABLE,
    get_offline_mode,
    get_show_progress_bar_default,
    get_tmp_dir_base
)


class SystemUtilsTest(unittest.TestCase):

    def test_get_offline_mode(self):
        self.assertFalse(get_offline_mode())

    def test_get_offline_mode_true(self):
        with patch.dict(os.environ, {OFFLINE_MODE_VARIABLE: '1'}):
            self.assertTrue(get_offline_mode())

    def test_get_show_progress_bar_default(self):
        self.assertTrue(get_show_progress_bar_default())

    def test_get_show_progress_bar_default_with_env_false(self):
        with patch.dict(os.environ, {PROGRESS_BARS_VARIABLE: '0'}):
            self.assertFalse(get_show_progress_bar_default())
        with patch.dict(os.environ, {PROGRESS_BARS_VARIABLE: 'false'}):
            self.assertFalse(get_show_progress_bar_default())

    def test_get_show_progress_bars_default_with_env_true(self):
        with patch.dict(os.environ, {PROGRESS_BARS_VARIABLE: '1'}):
            self.assertTrue(get_show_progress_bar_default())
        with patch.dict(os.environ, {PROGRESS_BARS_VARIABLE: 'true'}):
            self.assertTrue(get_show_progress_bar_default())

    def test_get_tmp_dir_base(self):
        self.assertIsNone(get_tmp_dir_base())

    def test_get_tmp_dir_base_custom(self):
        custom_tmp_dir = '/tmp'
        with patch.dict(os.environ, {TMP_DIR_VARIABLE: custom_tmp_dir}):
            self.assertEqual(custom_tmp_dir, get_tmp_dir_base())
