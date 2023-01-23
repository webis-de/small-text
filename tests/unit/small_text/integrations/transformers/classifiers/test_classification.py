import unittest
import pytest

import numpy as np

from unittest.mock import patch

from small_text.base import LABEL_UNLABELED
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.training.early_stopping import EarlyStopping, EarlyStoppingOrCondition
from small_text.training.metrics import Metric
from small_text.utils.logging import VERBOSITY_MORE_VERBOSE

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
        self.assertEqual(ModelLoadingStrategy.DEFAULT, model_args.model_loading_strategy)

    def test_transformer_model_arguments_init_with_paths(self):
        tokenizer = '/path/to/tokenizer/'
        config = '/path/to/config/'
        model_args = TransformerModelArguments('bert-base-uncased',
                                               tokenizer=tokenizer,
                                               config=config)
        self.assertEqual('bert-base-uncased', model_args.model)
        self.assertEqual(config, model_args.config)
        self.assertEqual(tokenizer, model_args.tokenizer)
        self.assertIsNotNone(model_args.model_loading_strategy)
        self.assertEqual(ModelLoadingStrategy.DEFAULT, model_args.model_loading_strategy)

    def test_transformer_model_arguments_init_with_model_loading_strategy(self):
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
        self.assertIsNotNone(model_args.model_loading_strategy)
        self.assertEqual(model_loading_strategy, model_args.model_loading_strategy)


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
        self.assertEqual(5, classifier.early_stopping_no_improvement)
        self.assertEqual(-1, classifier.early_stopping_acc)
        self.assertTrue(classifier.model_selection)
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
        early_stopping_no_improvement = 10
        early_stopping_acc = 0.99
        model_selection = True
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
            early_stopping_no_improvement=early_stopping_no_improvement,
            early_stopping_acc=early_stopping_acc,
            model_selection=model_selection,
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
        self.assertEqual(early_stopping_no_improvement, classifier.early_stopping_no_improvement)
        self.assertEqual(early_stopping_acc, classifier.early_stopping_acc)
        self.assertEqual(model_selection, classifier.model_selection)
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

    # TODO: remove this in 2.0.0
    def test_fit_with_early_stopping_fallback_default_kwargs(self):
        train = random_transformer_dataset(8)
        valid = random_transformer_dataset(2)

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args, 2)
        with patch.object(classifier, '_fit_main') as fit_main_mock:
            classifier.fit(train, validation_set=valid)
            fit_main_mock.assert_called()
            self.assertIsNone(classifier.class_weights_)

            call_args = fit_main_mock.call_args[0]
            self.assertTrue(isinstance(call_args[3], EarlyStopping))
            self.assertEqual('val_loss', call_args[3].metric.name)
            self.assertEqual(5, call_args[3].patience)

    # TODO: remove this in 2.0.0
    def test_fit_with_early_stopping_fallback_deprecated_kwargs(self):
        train = random_transformer_dataset(8)
        valid = random_transformer_dataset(2)

        early_stopping_no_improvement = 8
        early_stopping_acc = 0.98

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(
            model_args,
            2,
            early_stopping_no_improvement=early_stopping_no_improvement,
            early_stopping_acc=early_stopping_acc)
        with patch.object(classifier, '_fit_main') as fit_main_mock:
            classifier.fit(train, validation_set=valid)
            fit_main_mock.assert_called()
            self.assertIsNone(classifier.class_weights_)

            call_args = fit_main_mock.call_args[0]
            self.assertTrue(isinstance(call_args[3], EarlyStoppingOrCondition))

            first_handler = call_args[3].early_stopping_handlers[0]
            self.assertEqual('val_loss', first_handler.metric.name)
            self.assertEqual(early_stopping_no_improvement, first_handler.patience)

            second_handler = call_args[3].early_stopping_handlers[1]
            self.assertEqual('train_acc', second_handler.metric.name)
            self.assertEqual(-1, second_handler.patience)

    # TODO: remove this in 2.0.0
    def test_fit_with_early_stopping_and_fall_back_simultaneously(self):
        dataset = random_transformer_dataset(10)

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        early_stopping = EarlyStopping(Metric('val_loss'))

        classifier = TransformerBasedClassification(model_args, 2,
                                                    early_stopping_no_improvement=7)
        with self.assertWarnsRegex(UserWarning, r'Both the fit\(\) argument early_stopping'):
            classifier.fit(dataset, early_stopping=early_stopping)

        classifier = TransformerBasedClassification(model_args, 2,
                                                    early_stopping_acc=0.98)
        with self.assertWarnsRegex(UserWarning, r'Both the fit\(\) argument early_stopping'):
            classifier.fit(dataset, early_stopping=early_stopping)

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

        with patch.object(classifier, '_train', wraps=classifier._train) as train_mock:
            classifier.fit(dataset, optimizer=optimizer, scheduler=scheduler)

            train_mock.assert_called()

            call_args = train_mock.call_args[0]
            self.assertEqual(1, train_mock.call_count)

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
