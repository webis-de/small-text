import os
import unittest
import pytest
import small_text

import numpy as np

from importlib import reload
from unittest.mock import patch

from small_text.base import LABEL_UNLABELED
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.utils.logging import VERBOSITY_MORE_VERBOSE
from small_text.utils.system import OFFLINE_MODE_VARIABLE, PROGRESS_BARS_VARIABLE

try:
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    from small_text.integrations.transformers.classifiers.base import (
        ModelLoadingStrategy
    )
    from small_text.integrations.transformers.classifiers.classification import \
        FineTuningArguments, TransformerModelArguments, TransformerBasedClassification
    from small_text.integrations.pytorch.datasets import PytorchDatasetView
    from small_text.integrations.transformers.datasets import TransformersDataset
    from tests.utils.datasets import random_transformer_dataset
except (ModuleNotFoundError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class TestFineTuningArguments(unittest.TestCase):

    def test_fine_tuning_arguments_init(self):
        FineTuningArguments(0.2, 0.9)
        FineTuningArguments(0.2, -1)

    def test_fine_tuning_arguments_invalid_lr(self):
        with self.assertRaises(ValueError):
            FineTuningArguments(0, 0)

    def test_fine_tuning_arguments_invalid_decay_factor(self):
        for invalid_decay_factor in [0, 1]:
            with self.assertRaises(ValueError):
                FineTuningArguments(0, invalid_decay_factor)


@pytest.mark.pytorch
class TestTransformerModelArguments(unittest.TestCase):

    def test_transformer_model_arguments_init(self):
        model_args = TransformerModelArguments('bert-base-uncased')
        self.assertEqual('bert-base-uncased', model_args.model)
        self.assertEqual('bert-base-uncased', model_args.config)
        self.assertEqual('bert-base-uncased', model_args.tokenizer)
        self.assertIsNotNone(model_args.model_loading_strategy)
        self.assertTrue(model_args.show_progress_bar)
        self.assertEqual(ModelLoadingStrategy.DEFAULT, model_args.model_loading_strategy)
        self.assertFalse(model_args.compile_model)

    def test_transformer_model_arguments_init_with_paths(self):
        tokenizer = '/path/to/tokenizer/'
        config = '/path/to/config/'
        model_args = TransformerModelArguments('bert-base-uncased',
                                               tokenizer=tokenizer,
                                               config=config)
        self.assertEqual('bert-base-uncased', model_args.model)
        self.assertEqual(config, model_args.config)
        self.assertEqual(tokenizer, model_args.tokenizer)
        self._assert_empty_kwargs(model_args)
        self.assertIsNotNone(model_args.model_loading_strategy)
        self.assertTrue(model_args.show_progress_bar)
        self.assertEqual(ModelLoadingStrategy.DEFAULT, model_args.model_loading_strategy)
        self.assertFalse(model_args.compile_model)

    def _assert_empty_kwargs(self, model_args):
        self._assert_dict_not_none_and_empty(model_args.model_kwargs)
        self._assert_dict_not_none_and_empty(model_args.tokenizer_kwargs)
        self._assert_dict_not_none_and_empty(model_args.config_kwargs)

    def _assert_dict_not_none_and_empty(self, dict_to_check):
        self.assertIsNotNone(dict_to_check)
        self.assertEqual(0, len(dict_to_check))

    def test_transformer_model_arguments_init_show_progress_bar(self):
        tokenizer = '/path/to/tokenizer/'
        config = '/path/to/config/'
        model_args = TransformerModelArguments('bert-base-uncased',
                                               tokenizer=tokenizer,
                                               config=config,
                                               show_progress_bar=False)
        self.assertEqual('bert-base-uncased', model_args.model)
        self.assertEqual(config, model_args.config)
        self.assertEqual(tokenizer, model_args.tokenizer)
        self._assert_empty_kwargs(model_args)
        self.assertIsNotNone(model_args.model_loading_strategy)
        self.assertFalse(model_args.show_progress_bar)
        self.assertEqual(ModelLoadingStrategy.DEFAULT, model_args.model_loading_strategy)
        self.assertFalse(model_args.compile_model)

    def test_transformer_model_arguments_init_show_progress_bar_env_override(self):
        with patch.dict(os.environ, {PROGRESS_BARS_VARIABLE: '0'}):
            tokenizer = '/path/to/tokenizer/'
            config = '/path/to/config/'
            model_args = TransformerModelArguments('bert-base-uncased',
                                                   tokenizer=tokenizer,
                                                   config=config)
            self.assertEqual('bert-base-uncased', model_args.model)
            self.assertEqual(config, model_args.config)
            self.assertEqual(tokenizer, model_args.tokenizer)
            self._assert_empty_kwargs(model_args)
            self.assertIsNotNone(model_args.model_loading_strategy)
            self.assertFalse(model_args.show_progress_bar)
            self.assertEqual(ModelLoadingStrategy.DEFAULT, model_args.model_loading_strategy)
            self.assertFalse(model_args.compile_model)

    def test_transformer_model_arguments_init_model_loading_strategy(self):
        tokenizer = '/path/to/tokenizer/'
        config = '/path/to/config/'
        model_loading_strategy = ModelLoadingStrategy.ALWAYS_LOCAL
        model_args = TransformerModelArguments('bert-base-uncased',
                                               tokenizer=tokenizer,
                                               config=config,
                                               model_loading_strategy=model_loading_strategy)
        self.assertEqual('bert-base-uncased', model_args.model)
        self.assertEqual(config, model_args.config)
        self.assertEqual(tokenizer, model_args.tokenizer)
        self._assert_empty_kwargs(model_args)
        self.assertIsNotNone(model_args.model_loading_strategy)
        self.assertTrue(model_args.show_progress_bar)
        self.assertEqual(model_loading_strategy, model_args.model_loading_strategy)
        self.assertFalse(model_args.compile_model)

    def test_transformer_model_arguments_init_model_loading_strategy_with_env_override(self):
        with patch.dict(os.environ, {OFFLINE_MODE_VARIABLE: '1'}):
            # reload TransformerModelArguments so that updated environment variables are read
            reload(small_text.integrations.transformers.classifiers.classification)
            from small_text.integrations.transformers.classifiers.classification import TransformerModelArguments

            tokenizer = '/path/to/tokenizer/'
            config = '/path/to/config/'
            model_args = TransformerModelArguments('bert-base-uncased',
                                                   tokenizer=tokenizer,
                                                   config=config)
            self.assertEqual('bert-base-uncased', model_args.model)
            self.assertEqual(config, model_args.config)
            self.assertEqual(tokenizer, model_args.tokenizer)
            self._assert_empty_kwargs(model_args)
            self.assertIsNotNone(model_args.model_loading_strategy)
            self.assertTrue(model_args.show_progress_bar)
            self.assertEqual(ModelLoadingStrategy.ALWAYS_LOCAL, model_args.model_loading_strategy)
            self.assertFalse(model_args.compile_model)

    def test_transformer_model_arguments_init_with_compile(self):
        tokenizer = '/path/to/tokenizer/'
        config = '/path/to/config/'
        model_args = TransformerModelArguments('bert-base-uncased',
                                               tokenizer=tokenizer,
                                               config=config,
                                               compile_model=False)
        self.assertEqual('bert-base-uncased', model_args.model)
        self.assertEqual(config, model_args.config)
        self.assertEqual(tokenizer, model_args.tokenizer)
        self._assert_empty_kwargs(model_args)
        self.assertIsNotNone(model_args.model_loading_strategy)
        self.assertTrue(model_args.show_progress_bar)
        self.assertEqual(ModelLoadingStrategy.DEFAULT, model_args.model_loading_strategy)
        self.assertFalse(model_args.compile_model)

    def test_transformer_model_arguments_init_model_kwargs(self):
        tokenizer = '/path/to/tokenizer/'
        config = '/path/to/config/'
        model_kwargs = {'torch_dtype': 'bfloat16'}
        model_args = TransformerModelArguments('bert-base-uncased',
                                               tokenizer=tokenizer,
                                               config=config,
                                               model_kwargs=model_kwargs)
        self.assertEqual('bert-base-uncased', model_args.model)
        self.assertEqual(config, model_args.config)
        self.assertEqual(tokenizer, model_args.tokenizer)
        self.assertEqual(model_kwargs, model_args.model_kwargs)
        self._assert_dict_not_none_and_empty(model_args.tokenizer_kwargs)
        self._assert_dict_not_none_and_empty(model_args.config_kwargs)
        self.assertIsNotNone(model_args.model_loading_strategy)
        self.assertTrue(model_args.show_progress_bar)
        self.assertEqual(ModelLoadingStrategy.DEFAULT, model_args.model_loading_strategy)
        self.assertFalse(model_args.compile_model)

    def test_transformer_model_arguments_init_model_kwargs_invalid(self):
        tokenizer = '/path/to/tokenizer/'
        config = '/path/to/config/'
        model_kwargs = {'cache_dir': '/path/to/cache'}

        with self.assertRaisesRegex(ValueError, 'Cannot override managed keyword argument in model_kwargs'):
            TransformerModelArguments('bert-base-uncased',
                                      tokenizer=tokenizer,
                                      config=config,
                                      model_kwargs=model_kwargs)

    def test_transformer_model_arguments_init_tokenizer_kwargs(self):
        tokenizer = '/path/to/tokenizer/'
        config = '/path/to/config/'
        tokenizer_kwargs = {'eos_token': '</s>'}
        model_args = TransformerModelArguments('bert-base-uncased',
                                               tokenizer=tokenizer,
                                               config=config,
                                               tokenizer_kwargs=tokenizer_kwargs)
        self.assertEqual('bert-base-uncased', model_args.model)
        self.assertEqual(config, model_args.config)
        self.assertEqual(tokenizer, model_args.tokenizer)
        self._assert_dict_not_none_and_empty(model_args.model_kwargs)
        self.assertEqual(tokenizer_kwargs, model_args.tokenizer_kwargs)
        self._assert_dict_not_none_and_empty(model_args.config_kwargs)
        self.assertIsNotNone(model_args.model_loading_strategy)
        self.assertTrue(model_args.show_progress_bar)
        self.assertEqual(ModelLoadingStrategy.DEFAULT, model_args.model_loading_strategy)
        self.assertFalse(model_args.compile_model)

    def test_transformer_model_arguments_init_tokenizer_kwargs_invalid(self):
        tokenizer = '/path/to/tokenizer/'
        config = '/path/to/config/'
        tokenizer_kwargs = {'cache_dir': '/path/to/cache'}

        with self.assertRaisesRegex(ValueError, 'Cannot override managed keyword argument in tokenizer_kwargs'):
            TransformerModelArguments('bert-base-uncased',
                                      tokenizer=tokenizer,
                                      config=config,
                                      tokenizer_kwargs=tokenizer_kwargs)

    def test_transformer_model_arguments_init_config_kwargs(self):
        tokenizer = '/path/to/tokenizer/'
        config = '/path/to/config/'
        config_kwargs = {'output_attentions': True}
        model_args = TransformerModelArguments('bert-base-uncased',
                                               tokenizer=tokenizer,
                                               config=config,
                                               config_kwargs=config_kwargs)
        self.assertEqual('bert-base-uncased', model_args.model)
        self.assertEqual(config, model_args.config)
        self.assertEqual(tokenizer, model_args.tokenizer)
        self._assert_dict_not_none_and_empty(model_args.model_kwargs)
        self._assert_dict_not_none_and_empty(model_args.tokenizer_kwargs)
        self.assertEqual(config_kwargs, model_args.config_kwargs)
        self.assertIsNotNone(model_args.model_loading_strategy)
        self.assertTrue(model_args.show_progress_bar)
        self.assertEqual(ModelLoadingStrategy.DEFAULT, model_args.model_loading_strategy)
        self.assertFalse(model_args.compile_model)

    def test_transformer_model_arguments_init_config_kwargs_invalid(self):
        tokenizer = '/path/to/tokenizer/'
        config = '/path/to/config/'
        config_kwargs = {'cache_dir': '/path/to/cache'}

        with self.assertRaisesRegex(ValueError, 'Cannot override managed keyword argument in config_kwargs'):
            TransformerModelArguments('bert-base-uncased',
                                      tokenizer=tokenizer,
                                      config=config,
                                      config_kwargs=config_kwargs)


@pytest.mark.pytorch
class TestTransformerBasedClassification(unittest.TestCase):

    def test_init(self):
        model_args = TransformerModelArguments('bert-base-uncased')
        num_classes = 2
        classifier = TransformerBasedClassification(model_args, num_classes)
        self.assertEqual(num_classes, classifier.num_classes)
        self.assertFalse(classifier.multi_label)
        self.assertEqual(10, classifier.num_epochs)
        self.assertEqual(2e-5, classifier.lr)
        self.assertEqual(12, classifier.mini_batch_size)
        self.assertIsNone(classifier.criterion)
        self.assertEqual(0.1, classifier.validation_set_size)
        self.assertEqual(1, classifier.validations_per_epoch)
        self.assertIsNone(classifier.fine_tuning_arguments)
        self.assertIsNotNone(classifier.device)
        self.assertEqual(1, classifier.memory_fix)
        self.assertIsNone(classifier.class_weight)
        self.assertEqual(VERBOSITY_MORE_VERBOSE, classifier.verbosity)
        self.assertEqual('.active_learning_lib_cache/', classifier.cache_dir)

    def test_init_parameters(self):
        model_args = TransformerModelArguments('bert-base-uncased')
        num_classes = 3
        multi_label = False
        num_epochs = 20
        lr = 1e-5
        mini_batch_size = 24
        validation_set_size = 0.05
        validations_per_epoch = 5
        fine_tuning_arguments = FineTuningArguments(5e-5, 0.99)
        device = 'cuda'
        memory_fix = 1
        class_weight = 'balanced'
        verbosity = VERBOSITY_MORE_VERBOSE,
        cache_dir = '.active_learning_lib_cache/'

        classifier = TransformerBasedClassification(
            model_args,
            num_classes,
            num_epochs=num_epochs,
            lr=lr,
            mini_batch_size=mini_batch_size,
            validation_set_size=validation_set_size,
            validations_per_epoch=validations_per_epoch,
            fine_tuning_arguments=fine_tuning_arguments,
            device=device,
            memory_fix=memory_fix,
            class_weight=class_weight,
            verbosity=verbosity,
            cache_dir=cache_dir
        )

        self.assertEqual(num_classes, classifier.num_classes)
        self.assertEqual(multi_label, classifier.multi_label)
        self.assertEqual(num_epochs, classifier.num_epochs)
        self.assertEqual(lr, classifier.lr)
        self.assertEqual(mini_batch_size, classifier.mini_batch_size)
        self.assertEqual(validation_set_size, classifier.validation_set_size)
        self.assertEqual(validations_per_epoch, classifier.validations_per_epoch)
        self.assertEqual(fine_tuning_arguments, classifier.fine_tuning_arguments)
        self.assertEqual(device, classifier.device)
        self.assertEqual(memory_fix, classifier.memory_fix)
        self.assertEqual(class_weight, classifier.class_weight)
        self.assertEqual(verbosity, classifier.verbosity)
        self.assertEqual(cache_dir, classifier.cache_dir)

    def test_fit_where_y_train_contains_unlabeled(self):
        train_set = random_transformer_dataset(10)
        train_set.y = np.array([LABEL_UNLABELED] * 10)

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args, 2)
        with self.assertRaisesRegex(ValueError, 'Training set labels must be labeled'):
            classifier.fit(train_set)

    def test_fit_where_y_valid_contains_unlabeled(self):
        train_set = random_transformer_dataset(8)
        validation_set = random_transformer_dataset(2)
        validation_set.y = np.array([LABEL_UNLABELED] * 2)

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args, 2)
        with self.assertRaisesRegex(ValueError, 'Validation set labels must be labeled'):
            classifier.fit(train_set, validation_set=validation_set)

    def test_fit_with_train_set_mismatch_single_and_multi(self):
        train_set = random_transformer_dataset(10)

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args, 2, multi_label=True)

        with self.assertRaisesRegex(ValueError, 'The classifier is configured for single-label classification'):
            classifier.fit(train_set)

    def test_fit_with_train_set_mismatch_multi_and_single(self):
        train_set = random_transformer_dataset(10, num_classes=3, multi_label=True)

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args, 3)

        with self.assertRaisesRegex(ValueError, 'The classifier is configured for single-label classification'):
            classifier.fit(train_set)

    def test_fit_with_validation_set_mismatch_single_and_multi(self):
        train_set = random_transformer_dataset(10, num_classes=3, multi_label=True)
        validation_set = random_transformer_dataset(2)

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args, 2, multi_label=True)

        with self.assertRaisesRegex(ValueError, 'The classifier is configured for single-label classification'):
            classifier.fit(train_set, validation_set=validation_set)

    def test_fit_with_validation_set_mismatch_multi_and_single(self):
        train_set = random_transformer_dataset(10)
        validation_set = random_transformer_dataset(2, num_classes=3, multi_label=True)

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args, 3)

        with self.assertRaisesRegex(ValueError, 'The classifier is configured for single-label classification'):
            classifier.fit(train_set, validation_set=validation_set)

    def test_fit_without_validation_set(self):
        dataset = random_transformer_dataset(10)

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args, 2)
        with patch.object(classifier, '_fit_main') as fit_main_mock:
            classifier.fit(dataset)
            fit_main_mock.assert_called()
            self.assertIsNone(classifier.class_weights_)

            call_args = fit_main_mock.call_args[0]
            self.assertTrue(isinstance(call_args[0], PytorchDatasetView))
            self.assertTrue(isinstance(call_args[1], PytorchDatasetView))

            self.assertEqual(len(dataset), len(call_args[0]) + len(call_args[1]))

    def test_fit_with_validation_set(self):
        train = random_transformer_dataset(8)
        valid = random_transformer_dataset(2)

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args, 2)
        with patch.object(classifier, '_fit_main') as fit_main_mock:
            classifier.fit(train, validation_set=valid)
            fit_main_mock.assert_called()
            self.assertIsNone(classifier.class_weights_)

            call_args = fit_main_mock.call_args[0]
            self.assertTrue(isinstance(call_args[0], TransformersDataset))
            self.assertTrue(isinstance(call_args[1], TransformersDataset))

            self.assertEqual(len(train), len(call_args[0]))
            self.assertEqual(len(valid), len(call_args[1]))

    def test_fit_with_class_weighting(self):
        dataset = random_transformer_dataset(10)

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args, 2,
                                                    class_weight='balanced')
        with patch.object(classifier, '_fit_main') as fit_main_mock:
            classifier.fit(dataset)
            fit_main_mock.assert_called()
            self.assertIsNotNone(classifier.class_weights_)

    def test_fit_with_invalid_sample_weights(self):
        dataset = random_transformer_dataset(10)

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args, 2)

        weights = np.random.randn(len(dataset))
        weights[0] = -1

        with self.assertRaisesRegex(ValueError, 'Weights must be greater zero.'):
            classifier.fit(dataset, weights=weights)

    def test_fit_with_optimizer_and_scheduler(self):
        dataset = random_transformer_dataset(10)

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args, 2)

        classifier.fit(dataset)

        params = [param for param in classifier.model.parameters()
                  if param.requires_grad]

        optimizer = AdamW(params, lr=5e-5)
        steps = (len(dataset) // classifier.mini_batch_size) \
            + int(len(dataset) % classifier.mini_batch_size != 0)

        scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * steps, steps)

        with patch.object(classifier, '_fit_main') as fit_main_mock, \
                classifier.fit(dataset, optimizer=optimizer, scheduler=scheduler):

            fit_main_mock.assert_called()

            call_args = fit_main_mock.call_args[0]
            self.assertEqual(1, fit_main_mock.call_count)

            self.assertEqual(optimizer, call_args[5])
            self.assertEqual(scheduler, call_args[6])

    def test_predict_on_empty_data(self):
        test_set = TransformersDataset([], None)

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args, 2)
        # here would be a clf.fit call, which omit due to the runtime costs

        predictions = clf.predict(test_set)
        self.assertEqual(0, predictions.shape[0])
        self.assertTrue(np.issubdtype(predictions.dtype, np.integer))

    def test_predict_proba_on_empty_data(self):
        test_set = TransformersDataset([], None)

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args, 2)
        # here would be a clf.fit call, which omit due to the runtime costs

        proba = clf.predict_proba(test_set)
        self.assertEqual(0, proba.shape[0])
        self.assertTrue(np.issubdtype(proba.dtype, float))
