import unittest
import pytest

import numpy as np

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from tests.utils.datasets import random_labeling, random_labels


try:
    from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset
except (PytorchNotFoundError, ModuleNotFoundError):
    pass


def get_text_field():
    try:
        from torchtext import data
        text_field = data.Field(lower=True)
    except AttributeError:
        # torchtext >= 0.8.0
        from torchtext.legacy import data
        text_field = data.Field(lower=True)
    return text_field


@pytest.mark.pytorch
class PytorchTextClassificationDatasetSingleLabelTest(unittest.TestCase):

    def test_from_arrays_with_lists(self):
        texts = [f'train data {i}' for i in range(10)]
        labels = np.array([random_labeling(3) for _ in range(10)])

        text_field = get_text_field()

        dataset = PytorchTextClassificationDataset.from_arrays(texts, labels, text_field)
        self.assertEqual(10, len(dataset))

    def test_from_arrays_with_ndarray(self):
        texts = np.array([f'train data {i}' for i in range(10)])
        labels = np.array([random_labeling(3) for _ in range(10)])

        text_field = get_text_field()

        dataset = PytorchTextClassificationDataset.from_arrays(texts, labels, text_field)
        self.assertEqual(10, len(dataset))
        self.assertTrue('train' in dataset.vocab.stoi)
        self.assertTrue('data' in dataset.vocab.stoi)

    def test_from_arrays_with_test_data(self):
        texts_train = np.array([f'train data {i}' for i in range(10)])
        labels_train = np.array([random_labeling(3) for _ in range(10)])

        text_field = get_text_field()

        dataset = PytorchTextClassificationDataset.from_arrays(texts_train,
                                                               labels_train,
                                                               text_field)
        self.assertEqual(10, len(dataset))

        texts_test = np.array([f'test data {i}' for i in range(10)])
        labels_test = np.array([random_labeling(3) for _ in range(10)])

        dataset = PytorchTextClassificationDataset.from_arrays(texts_test,
                                                               labels_test,
                                                               text_field,
                                                               train=False)
        self.assertEqual(10, len(dataset))
        self.assertFalse('test' in dataset.vocab.stoi)
        self.assertTrue('data' in dataset.vocab.stoi)

    def test_from_arrays_with_test_data_and_no_vocab(self):

        texts_test = np.array([f'test data {i}' for i in range(10)])
        labels_test = np.array([random_labeling(3) for _ in range(10)])

        text_field = get_text_field()

        with self.assertRaisesRegex(ValueError, 'Vocab must have been built'):
            PytorchTextClassificationDataset.from_arrays(texts_test,
                                                         labels_test,
                                                         text_field,
                                                         train=False)


@pytest.mark.pytorch
class PytorchTextClassificationDatasetMultiLabelTest(unittest.TestCase):

    def test_from_arrays_with_lists(self):
        texts = [f'train data {i}' for i in range(10)]
        labels = random_labels(10, 3, multi_label=True)

        text_field = get_text_field()

        dataset = PytorchTextClassificationDataset.from_arrays(texts, labels, text_field)
        self.assertEqual(10, len(dataset))
        self.assertTrue(dataset.is_multi_label)

    def test_from_arrays_with_ndarray(self):
        texts = np.array([f'train data {i}' for i in range(10)])
        labels = random_labels(10, 3, multi_label=True)

        text_field = get_text_field()

        dataset = PytorchTextClassificationDataset.from_arrays(texts, labels, text_field)
        self.assertEqual(10, len(dataset))
        self.assertTrue(dataset.is_multi_label)
        self.assertTrue('train' in dataset.vocab.stoi)
        self.assertTrue('data' in dataset.vocab.stoi)

    def test_from_arrays_with_test_data(self):
        texts_train = np.array([f'train data {i}' for i in range(10)])
        labels_train = random_labels(10, 3, multi_label=True)

        text_field = get_text_field()

        dataset = PytorchTextClassificationDataset.from_arrays(texts_train, labels_train,
                                                               text_field)
        self.assertTrue(dataset.is_multi_label)

        texts_test = np.array([f'test data {i}' for i in range(10)])
        labels_test = random_labels(10, 3, multi_label=True)

        dataset = PytorchTextClassificationDataset.from_arrays(texts_test,
                                                               labels_test,
                                                               text_field,
                                                               train=False)
        self.assertEqual(10, len(dataset))
        self.assertTrue(dataset.is_multi_label)
        self.assertFalse('test' in dataset.vocab.stoi)
        self.assertTrue('data' in dataset.vocab.stoi)

    def test_from_arrays_with_test_data_and_no_vocab(self):

        texts_test = np.array([f'test data {i}' for i in range(10)])
        labels_test = random_labels(10, 3, multi_label=True)

        text_field = get_text_field()

        with self.assertRaisesRegex(ValueError, 'Vocab must have been built'):
            PytorchTextClassificationDataset.from_arrays(texts_test,
                                                         labels_test,
                                                         text_field,
                                                         train=False)
