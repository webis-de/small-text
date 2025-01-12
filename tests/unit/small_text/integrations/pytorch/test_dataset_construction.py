import unittest
import pytest

import numpy as np

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from tests.utils.datasets import random_labeling, random_labels, _train_tokenizer

try:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset
except (PytorchNotFoundError, ModuleNotFoundError):
    pass


@pytest.mark.pytorch
class PytorchTextClassificationDatasetSingleLabelTest(unittest.TestCase):

    def test_from_arrays_with_lists(self):
        texts = [f'train data {i}' for i in range(10)]
        labels = np.array([random_labeling(3) for _ in range(10)])

        tokenizer = _train_tokenizer(texts, vocab_size=100)

        dataset = PytorchTextClassificationDataset.from_arrays(texts, labels, tokenizer)
        self.assertEqual(10, len(dataset))

    def test_from_arrays_with_ndarray(self):
        texts = np.array([f'train data {i}' for i in range(10)])
        labels = np.array([random_labeling(3) for _ in range(10)])

        tokenizer = _train_tokenizer(texts, vocab_size=100)

        dataset = PytorchTextClassificationDataset.from_arrays(texts, labels, tokenizer)
        self.assertEqual(10, len(dataset))


@pytest.mark.pytorch
class PytorchTextClassificationDatasetMultiLabelTest(unittest.TestCase):

    def test_from_arrays_with_lists(self):
        texts = [f'train data {i}' for i in range(10)]
        labels = random_labels(10, 3, multi_label=True)

        tokenizer = _train_tokenizer(texts, vocab_size=100)

        dataset = PytorchTextClassificationDataset.from_arrays(texts, labels, tokenizer)
        self.assertEqual(10, len(dataset))
        self.assertTrue(dataset.is_multi_label)

    def test_from_arrays_with_ndarray(self):
        texts = np.array([f'train data {i}' for i in range(10)])
        labels = random_labels(10, 3, multi_label=True)

        tokenizer = _train_tokenizer(texts, vocab_size=100)

        dataset = PytorchTextClassificationDataset.from_arrays(texts, labels, tokenizer)
        self.assertEqual(10, len(dataset))
        self.assertTrue(dataset.is_multi_label)
