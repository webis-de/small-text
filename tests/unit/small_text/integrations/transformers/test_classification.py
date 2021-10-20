import unittest
import pytest
import warnings

from unittest.mock import patch
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from torch.nn.modules import BCEWithLogitsLoss

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
        num_classes = 2
        model_args = TransformerModelArguments('bert-base-uncased')
        classifier = TransformerBasedClassification(model_args, num_classes)
        self.assertEqual(num_classes, classifier.num_classes)

    def test_init_with_non_default_criterion_and_class_weighting(self):
        num_classes = 2
        criterion = BCEWithLogitsLoss()

        with warnings.catch_warnings(record=True) as w:
            model_args = TransformerModelArguments('bert-base-uncased')
            TransformerBasedClassification(model_args, num_classes, criterion=criterion,
                                           class_weight='balanced')

            self.assertEqual(1, len(w))
            self.assertTrue(issubclass(w[0].category, RuntimeWarning))

    @pytest.mark.skip(reason='reevaluate if None is plausible')
    def test_fit_where_y_is_none(self):
        dataset = random_transformer_dataset(10)
        dataset.y = [None] * 10

        model_args = TransformerModelArguments('bert-base-uncased')
        classifier = TransformerBasedClassification(model_args, 2)
        with self.assertRaises(ValueError):
            classifier.fit(dataset)

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

    @pytest.mark.skip(reason='reevaluate if None is plausible')
    def test_fit_with_validation_set_but_missing_labels(self):
        train = random_transformer_dataset(8)
        valid = random_transformer_dataset(2)
        valid.y = [None] * len(valid)

        classifier = TransformerBasedClassification('bert-base-uncased', 2)
        with self.assertRaises(ValueError):
            classifier.fit(train, validation_set=valid)
