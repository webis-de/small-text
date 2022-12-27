import unittest
import pytest

from unittest.mock import patch, MagicMock
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
    from small_text.integrations.transformers.classifiers.base import (
        ModelLoadingStrategy
    )
    from small_text.integrations.transformers.classifiers.classification import (
        TransformerModelArguments
    )
    from small_text.integrations.transformers.utils.classification import (
        _get_arguments_for_from_pretrained_model,
        _initialize_transformer_components
    )
except (PytorchNotFoundError, ModuleNotFoundError):
    pass


@pytest.mark.pytorch
class ClassificationUtilsTest(unittest.TestCase):

    def test_get_arguments_for_from_pretrained_model_strategy_default(self):
        args = _get_arguments_for_from_pretrained_model(ModelLoadingStrategy.DEFAULT)
        self.assertFalse(args.force_download)
        self.assertFalse(args.local_files_only)

    def test_get_arguments_for_from_pretrained_model_strategy_always_local(self):
        args = _get_arguments_for_from_pretrained_model(ModelLoadingStrategy.ALWAYS_LOCAL)
        self.assertFalse(args.force_download)
        self.assertTrue(args.local_files_only)

    def test_get_arguments_for_from_pretrained_model_strategy_always_download(self):
        args = _get_arguments_for_from_pretrained_model(ModelLoadingStrategy.ALWAYS_DOWNLOAD)
        self.assertTrue(args.force_download)
        self.assertFalse(args.local_files_only)

    def test_initialize_transformer_components(self):
        transformer_model = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        num_labels = 3
        cache_dir = 'tmp/'

        def config_fake_from_pretrained(self, *_args, **_kwargs):
            return MagicMock(), {}

        with patch.object(AutoConfig, 'from_pretrained') as config_mock, \
            patch.object(AutoTokenizer, 'from_pretrained') as tokenizer_mock, \
                patch.object(AutoModelForSequenceClassification, 'from_pretrained') as model_mock:

            config_mock.side_effect = config_fake_from_pretrained

            config, tokenizer, model = _initialize_transformer_components(transformer_model,
                                                                          num_labels,
                                                                          cache_dir)

            self.assertIsNotNone(config)
            self.assertIsNotNone(tokenizer)
            self.assertIsNotNone(model)

            # this mock can be called multiple times
            for i in range(config_mock.call_count):
                kwargs = config_mock.call_args_list[i].kwargs
                self.assertEqual(transformer_model.config, config_mock.call_args_list[i][0][0])
                self.assertEqual(num_labels, kwargs['num_labels'])
                self.assertEqual(cache_dir, kwargs['cache_dir'])
                self.assertFalse(kwargs['force_download'])

            for i in range(tokenizer_mock.call_count):
                kwargs = tokenizer_mock.call_args_list[i].kwargs
                self.assertEqual(transformer_model.tokenizer, tokenizer_mock.call_args_list[i][0][0])
                self.assertEqual(cache_dir, kwargs['cache_dir'])
                self.assertFalse(kwargs['force_download'])

            for i in range(model_mock.call_count):
                kwargs = model_mock.call_args_list[i].kwargs
                self.assertEqual(transformer_model.model, model_mock.call_args_list[i][0][0])
                self.assertEqual(cache_dir, kwargs['cache_dir'])
                self.assertEqual(config, kwargs['config'])
                self.assertFalse(kwargs['force_download'])
                self.assertFalse(kwargs['local_files_only'])

    def test_initialize_transformer_components_with_model_loading_strategy_always_local(self):
        transformer_model = TransformerModelArguments(
            'sshleifer/tiny-distilroberta-base',
            model_loading_strategy=ModelLoadingStrategy.ALWAYS_LOCAL)
        num_labels = 3
        cache_dir = 'tmp/'

        def config_fake_from_pretrained(_self, *_args, **_kwargs):
            return MagicMock(), {}

        with patch.object(AutoConfig,
                          'from_pretrained') as config_mock, \
            patch.object(AutoTokenizer, 'from_pretrained') as tokenizer_mock, \
                patch.object(AutoModelForSequenceClassification, 'from_pretrained') as model_mock:

            config_mock.side_effect = config_fake_from_pretrained

            config, tokenizer, model = _initialize_transformer_components(transformer_model,
                                                                          num_labels,
                                                                          cache_dir)

            self.assertIsNotNone(config)
            self.assertIsNotNone(tokenizer)
            self.assertIsNotNone(model)

            # this mock can be called multiple times
            for i in range(config_mock.call_count):
                kwargs = config_mock.call_args_list[i].kwargs
                self.assertEqual(transformer_model.config, config_mock.call_args_list[i][0][0])
                self.assertEqual(num_labels, kwargs['num_labels'])
                self.assertEqual(cache_dir, kwargs['cache_dir'])
                self.assertFalse(kwargs['force_download'])

            for i in range(tokenizer_mock.call_count):
                kwargs = tokenizer_mock.call_args_list[i].kwargs
                self.assertEqual(transformer_model.tokenizer, tokenizer_mock.call_args_list[i][0][0])
                self.assertEqual(cache_dir, kwargs['cache_dir'])
                self.assertFalse(kwargs['force_download'])

            for i in range(model_mock.call_count):
                kwargs = model_mock.call_args_list[i].kwargs
                self.assertEqual(transformer_model.model, model_mock.call_args_list[i][0][0])
                self.assertEqual(cache_dir, kwargs['cache_dir'])
                self.assertEqual(config, kwargs['config'])
                self.assertFalse(kwargs['force_download'])
                self.assertTrue(kwargs['local_files_only'])
