import unittest
import pytest

import numpy as np

from unittest.mock import patch

from small_text.base import LABEL_UNLABELED
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.utils.logging import VERBOSITY_MORE_VERBOSE

try:
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

    def test_transformer_model_arguments_init_with_paths(self):
        tokenizer = '/path/to/tokenizer/'
        config = '/path/to/config/'
        model_args = TransformerModelArguments('bert-base-uncased', tokenizer=tokenizer, config=config)
        self.assertEqual('bert-base-uncased', model_args.model)
        self.assertEqual(config, model_args.config)
        self.assertEqual(tokenizer, model_args.tokenizer)


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
        self.assertEqual('sample', classifier.no_validation_set_action)
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
        no_validation_set_action = 'sample'

        classifier = TransformerBasedClassification(model_args,
                                                    num_classes,
                                                    num_epochs=num_epochs,
                                                    lr=lr,
                                                    mini_batch_size=mini_batch_size,
                                                    validation_set_size=validation_set_size,
                                                    validations_per_epoch=validations_per_epoch,
                                                    no_validation_set_action=no_validation_set_action)

        self.assertEqual(num_classes, classifier.num_classes)
        self.assertEqual(multi_label, classifier.multi_label)
        self.assertEqual(num_epochs, classifier.num_epochs)
        self.assertEqual(lr, classifier.lr)
        self.assertEqual(mini_batch_size, classifier.mini_batch_size)
        self.assertEqual(validation_set_size, classifier.validation_set_size)
        self.assertEqual(validations_per_epoch, classifier.validations_per_epoch)
        self.assertEqual(no_validation_set_action, classifier.no_validation_set_action)
        # TODO: incomplete

    def test_fit_where_y_train_contains_unlabeled(self):
        train_set = random_transformer_dataset(10)
        train_set.y = np.array([LABEL_UNLABELED] * 10)

        model_args = TransformerModelArguments('bert-base-uncased')
        classifier = TransformerBasedClassification(model_args, 2)
        with self.assertRaisesRegex(ValueError, 'Training set labels must be labeled'):
            classifier.fit(train_set)

    def test_fit_where_y_valid_contains_unlabeled(self):
        train_set = random_transformer_dataset(8)
        validation_set = random_transformer_dataset(2)
        validation_set.y = np.array([LABEL_UNLABELED] * 2)

        model_args = TransformerModelArguments('bert-base-uncased')
        classifier = TransformerBasedClassification(model_args, 2)
        with self.assertRaisesRegex(ValueError, 'Validation set labels must be labeled'):
            classifier.fit(train_set, validation_set=validation_set)

    def test_fit_with_label_information_mismatch(self):
        num_classes_configured = 3
        num_classes_to_be_encountered = 2

        train_set = random_transformer_dataset(8, num_classes=num_classes_to_be_encountered)
        validation_set = random_transformer_dataset(2, num_classes=num_classes_to_be_encountered)

        model_args = TransformerModelArguments('bert-base-uncased')
        classifier = TransformerBasedClassification(model_args, num_classes_configured)

        with self.assertRaisesRegex(ValueError,
                                    'Conflicting information about the number of classes: '
                                    'expected: 3, encountered: 2'):
            classifier.fit(train_set, validation_set=validation_set)

    def test_fit_without_validation_set(self):
        dataset = random_transformer_dataset(10)

        model_args = TransformerModelArguments('bert-base-uncased')
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

        model_args = TransformerModelArguments('bert-base-uncased')
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

        model_args = TransformerModelArguments('bert-base-uncased')
        classifier = TransformerBasedClassification(model_args, 2,
                                                    class_weight='balanced')
        with patch.object(classifier, '_fit_main') as fit_main_mock:
            classifier.fit(dataset)
            fit_main_mock.assert_called()
            self.assertIsNotNone(classifier.class_weights_)

    def test_predict_on_empty_data(self):
        test_set = TransformersDataset([], None)

        model_args = TransformerModelArguments('bert-base-uncased')
        clf = TransformerBasedClassification(model_args, 2)
        # here would be a clf.fit call, which omit due to the runtime costs

        predictions = clf.predict(test_set)
        self.assertEqual(0, predictions.shape[0])
        self.assertTrue(np.issubdtype(predictions.dtype, np.integer))

    def test_predict_proba_on_empty_data(self):
        test_set = TransformersDataset([], None)

        model_args = TransformerModelArguments('bert-base-uncased')
        clf = TransformerBasedClassification(model_args, 2)
        # here would be a clf.fit call, which omit due to the runtime costs

        proba = clf.predict_proba(test_set)
        self.assertEqual(0, proba.shape[0])
        self.assertTrue(np.issubdtype(proba.dtype, np.float))
