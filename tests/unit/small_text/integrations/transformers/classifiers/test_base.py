import unittest

from unittest.mock import patch
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from small_text.integrations.transformers import ModelLoadingStrategy, get_default_model_loading_strategy
except (ModuleNotFoundError, PytorchNotFoundError):
    pass


class BaseTest(unittest.TestCase):

    def test_get_default_model_loading_strategy(self):
        self.assertEqual(ModelLoadingStrategy.DEFAULT, get_default_model_loading_strategy())

    def test_get_default_model_loading_strategy_with_offline_true(self):
        with patch('small_text.integrations.transformers.classifiers.base.get_offline_mode') as offline_mock:
            offline_mock.return_value = True
            self.assertEqual(ModelLoadingStrategy.ALWAYS_LOCAL, get_default_model_loading_strategy())
