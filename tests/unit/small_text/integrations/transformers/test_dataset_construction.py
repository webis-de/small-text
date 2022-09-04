import unittest
import pytest

import numpy as np

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from tests.utils.datasets import random_labeling, random_labels


try:
    from transformers import AutoTokenizer
    from small_text.integrations.transformers.datasets import TransformersDataset
except (PytorchNotFoundError, ModuleNotFoundError):
    pass


@pytest.mark.pytorch
class TransformersDatasetSingleLabelTest(unittest.TestCase):

    def test_from_arrays_with_lists(self):
        texts = [f'train data {i}' for i in range(10)]
        labels = np.array([random_labeling(3) for _ in range(10)])
        tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-distilroberta-base')

        dataset = TransformersDataset.from_arrays(texts, labels, tokenizer)
        self.assertEqual(10, len(dataset))

    def test_from_arrays_with_ndarray(self):
        texts = np.array([f'train data {i}' for i in range(10)])
        labels = np.array([random_labeling(3) for _ in range(10)])
        tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-distilroberta-base')

        dataset = TransformersDataset.from_arrays(texts, labels, tokenizer)
        self.assertEqual(10, len(dataset))

    def test_from_arrays_with_test_data(self):
        texts_train = np.array([f'train data {i}' for i in range(10)])
        labels_train = np.array([random_labeling(3) for _ in range(10)])
        tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-distilroberta-base')

        dataset = TransformersDataset.from_arrays(texts_train, labels_train, tokenizer)
        self.assertEqual(10, len(dataset))

        texts_test = np.array([f'test data {i}' for i in range(10)])
        labels_test = np.array([random_labeling(3) for _ in range(10)])

        dataset = TransformersDataset.from_arrays(texts_test, labels_test, tokenizer)
        self.assertEqual(10, len(dataset))


@pytest.mark.pytorch
class TransformersDatasetMultiLabelTest(unittest.TestCase):

    def test_from_arrays_with_lists(self):
        texts = [f'train data {i}' for i in range(10)]
        labels = random_labels(10, 3, multi_label=True)
        tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-distilroberta-base')

        dataset = TransformersDataset.from_arrays(texts, labels, tokenizer)
        self.assertEqual(10, len(dataset))

    def test_from_arrays_with_ndarray(self):
        texts = np.array([f'train data {i}' for i in range(10)])
        labels = random_labels(10, 3, multi_label=True)
        tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-distilroberta-base')

        dataset = TransformersDataset.from_arrays(texts, labels, tokenizer)
        self.assertEqual(10, len(dataset))

    def test_from_arrays_with_test_data(self):
        texts_train = np.array([f'train data {i}' for i in range(10)])
        labels_train = random_labels(10, 3, multi_label=True)
        tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-distilroberta-base')

        dataset = TransformersDataset.from_arrays(texts_train, labels_train, tokenizer)
        self.assertEqual(10, len(dataset))

        texts_test = np.array([f'test data {i}' for i in range(10)])
        labels_test = random_labels(10, 3, multi_label=True)

        dataset = TransformersDataset.from_arrays(texts_test, labels_test, tokenizer)
        self.assertEqual(10, len(dataset))
