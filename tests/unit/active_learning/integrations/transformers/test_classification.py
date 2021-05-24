import unittest
import pytest

from unittest.mock import patch
from active_learning.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from active_learning.integrations.transformers.classifiers.classification import \
        FineTuningArguments, TransformerModelArguments, TransformerBasedClassification
    from active_learning.integrations.transformers.datasets import TransformersDataset
    from tests.utils.datasets import random_transformer_dataset
except PytorchNotFoundError:
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

    @pytest.mark.skip(reason='reevaluate if None is plausible')
    def test_fit_where_y_is_none(self):
        dataset = random_transformer_dataset(10)
        dataset.y = [None] * 10

        classifier = TransformerBasedClassification('bert-base-uncased')
        with self.assertRaises(ValueError):
            classifier.fit(dataset)

    def test_fit_without_validation_set(self):
        dataset = random_transformer_dataset(10)

        classifier = TransformerBasedClassification('bert-base-uncased')
        with patch.object(classifier, '_fit_main') as fit_main_mock:
            classifier.fit(dataset)
            fit_main_mock.assert_called()

            call_args = fit_main_mock.call_args[0]
            self.assertTrue(isinstance(call_args[0], TransformersDataset))
            self.assertTrue(isinstance(call_args[1], TransformersDataset))

            self.assertEqual(len(dataset), len(call_args[0]) + len(call_args[1]))

    def test_fit_with_validation_set(self):
        train = random_transformer_dataset(8)
        valid = random_transformer_dataset(2)

        classifier = TransformerBasedClassification('bert-base-uncased')
        with patch.object(classifier, '_fit_main') as fit_main_mock:
            classifier.fit(train, validation_set=valid)
            fit_main_mock.assert_called()

            call_args = fit_main_mock.call_args[0]
            self.assertTrue(isinstance(call_args[0], TransformersDataset))
            self.assertTrue(isinstance(call_args[1], TransformersDataset))

            self.assertEqual(len(train), len(call_args[0]))
            self.assertEqual(len(valid), len(call_args[1]))

    @pytest.mark.skip(reason='reevaluate if None is plausible')
    def test_fit_with_validation_set_but_missing_labels(self):
        train = random_transformer_dataset(8)
        valid = random_transformer_dataset(2)
        valid.y = [None] * len(valid)

        classifier = TransformerBasedClassification('bert-base-uncased')
        with self.assertRaises(ValueError):
            classifier.fit(train, validation_set=valid)
