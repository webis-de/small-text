import unittest
import pytest

import numpy as np

from collections import Counter

from numpy.testing import assert_array_equal
from parameterized import parameterized_class

from active_learning.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    from torchtext.vocab import Vocab

    from active_learning.integrations.pytorch.datasets import PytorchTextClassificationDataset
    from active_learning.integrations.pytorch.query_strategies import (
        ExpectedGradientLength, ExpectedGradientLengthMaxWord)
except (PytorchNotFoundError, ModuleNotFoundError):
    pass


@pytest.mark.pytorch
@parameterized_class([{'target_labels': 'explicit'}, {'target_labels': 'inferred'}])
class PytorchTextClassificationDatasetTest(unittest.TestCase):

    NUM_SAMPLES = 100
    NUM_LABELS = 3

    def _random_data(self, num_samples=100, num_dimension=40, num_labels=NUM_LABELS):
        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError('Invalid test parameter value for target_labels:' + self.target_labels)

        vocab = Vocab(Counter({'hello': 3}))
        data = [(torch.randint(10, (num_dimension,)),
                 np.random.randint(0, num_labels))
                for _ in range(num_samples)]

        target_labels = None if self.target_labels == 'inferred' else np.arange(num_labels)
        return PytorchTextClassificationDataset(data, vocab, target_labels=target_labels)

    def test_init(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        self.assertIsNotNone(ds._data)
        self.assertIsNotNone(ds.vocab)

    def test_get_features(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        self.assertEqual(self.NUM_SAMPLES, len(ds.x))

    def test_set_features(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        self.assertEqual(self.NUM_SAMPLES, len(ds.x))
        ds_new = self._random_data(num_samples=self.NUM_SAMPLES)
        ds.x = ds_new.x
        self.assertTrue(ds.x == ds_new.x)

    def test_get_labels(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        self.assertTrue(isinstance(ds.y, np.ndarray))
        self.assertEqual(self.NUM_SAMPLES, ds.y.shape[0])

    def test_set_labels(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        self.assertTrue(isinstance(ds.y, np.ndarray))
        self.assertEqual(self.NUM_SAMPLES, ds.y.shape[0])

    def test_get_target_labels(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        expected_target_labels = np.arange(self.NUM_LABELS)
        assert_array_equal(expected_target_labels, ds.target_labels)

    def test_set_target_labels(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        new_target_labels = np.arange(self.NUM_LABELS+1)
        ds.target_labels = new_target_labels
        assert_array_equal(new_target_labels, ds.target_labels)

    def test_indexing_single_index(self, index = 42):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)

        result = ds[index]
        self.assertEqual(1, len(result))

        self.assertEqual(ds._data[index], result._data[0])

    def test_indexing_list_index(self):
        index = [1, 42, 56, 99]
        ds = self._random_data(num_samples=self.NUM_SAMPLES)

        result = ds[index]
        self.assertEqual(len(index), len(result))

        self.assertEqual([ds._data[i] for i in index], result._data)

    def test_indexing_slicing(self):
        index = np.s_[10:20]
        ds = self._random_data(num_samples=self.NUM_SAMPLES)

        result = ds[index]
        self.assertEqual(10, len(result))

        self.assertEqual([ds._data[i] for i in np.arange(self.NUM_SAMPLES)[index]],
                         result._data)

    def test_datasen_len(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        self.assertEqual(self.NUM_SAMPLES, len(ds))
