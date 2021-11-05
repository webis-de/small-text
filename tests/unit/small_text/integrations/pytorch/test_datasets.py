import unittest
import pytest

import numpy as np

from collections import Counter

from numpy.testing import assert_array_equal
from parameterized import parameterized_class

from small_text.data.exceptions import UnsupportedOperationException
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from tests.utils.testing import assert_list_of_tensors_equal

try:
    import torch
    from torchtext.vocab import Vocab

    from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset, \
        PytorchDatasetView
    from small_text.integrations.pytorch.query_strategies import (
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
        data = [(torch.randint(10, (num_dimension,)), np.random.randint(0, num_labels))
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

        self.assertEqual(len(ds.x), len(ds_new.x))
        tensor_pairs = zip([item for item in ds.x], [item for item in ds_new.x])
        is_equal = [torch.equal(first, second)
                    for first, second in tensor_pairs]
        self.assertTrue(np.alltrue(is_equal))

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

    def test_get_data(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        assert_array_equal(len(ds), len(ds.data))

    def test_indexing_single_index(self, index=42):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)

        result = ds[index]
        self.assertEqual(1, len(result))
        self.assertTrue(isinstance(result, PytorchDatasetView))

        self.assertTrue(torch.equal(ds.x[index], result.x[0]))

    def test_indexing_list_index(self):
        index = [1, 42, 56, 99]
        ds = self._random_data(num_samples=self.NUM_SAMPLES)

        result = ds[index]
        self.assertEqual(len(index), len(result))
        self.assertTrue(isinstance(result, PytorchDatasetView))

        expected = [ds.x[i] for i in index]
        assert_list_of_tensors_equal(self, expected, result.x)

    def test_indexing_slicing(self):
        index = np.s_[10:20]
        ds = self._random_data(num_samples=self.NUM_SAMPLES)

        result = ds[index]
        self.assertEqual(10, len(result))
        self.assertTrue(isinstance(result, PytorchDatasetView))

        expected = [ds.x[i] for i in np.arange(self.NUM_SAMPLES)[index]]
        assert_list_of_tensors_equal(self, expected, result.x)

    def test_indexing_mutability(self):
        selections = [42, [1, 42, 56, 99], np.s_[10:20]]
        for selection in selections:
            ds = self._random_data(num_samples=self.NUM_SAMPLES)

            dataset_view = ds[selection]

            # flip the signs of the view's data (base dataset should be unchanged)
            dataset_view._dataset.x = [-tensor for tensor in dataset_view._dataset.x]

            for i, _ in enumerate(dataset_view.x):
                torch.equal(ds.x[i], dataset_view.x[i])

            # flip the signs of the base dataset (view should reflect changes)
            ds.x = [-item for item in ds.x]

            for i, _ in enumerate(dataset_view.x):
                torch.equal(ds.x[i], dataset_view.x[i])

    def test_iter(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        count = 0
        for _ in ds:
            count += 1
        self.assertEqual(self.NUM_SAMPLES, count)

    def test_datasen_len(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        self.assertEqual(self.NUM_SAMPLES, len(ds))


class _PytorchDatasetViewTest(object):

    NUM_SAMPLES = 20
    NUM_SAMPLES_VIEW = 14

    def test_init_with_slice(self):
        dataset_view = self._dataset()
        view_on_view = PytorchDatasetView(dataset_view, slice(0, 10))
        self.assertEqual(10, len(view_on_view))

    def test_init_with_slice_and_step(self):
        dataset_view = self._dataset()
        view_on_view = PytorchDatasetView(dataset_view, slice(0, 10, 2))
        self.assertEqual(5, len(view_on_view))

    def test_init_with_numpy_slice(self):
        dataset = self._dataset()
        self.assertEqual(self.NUM_SAMPLES_VIEW, len(dataset))
        dataset_view = PytorchDatasetView(dataset, np.s_[0:10])
        self.assertEqual(10, len(dataset_view))

    def test_get_x(self):
        dataset = self._dataset()
        selection = np.random.randint(0, high=len(dataset), size=10)
        dataset_view = PytorchDatasetView(dataset, selection)
        assert_list_of_tensors_equal(self, [dataset.x[i] for i in selection], dataset_view.x)

    def test_set_x(self):
        dataset = self._dataset()
        with self.assertRaises(UnsupportedOperationException):
            dataset.x = self._dataset()

    def test_get_y(self):
        dataset = self._dataset()
        selection = np.random.randint(0, high=len(dataset), size=10)
        dataset_view = PytorchDatasetView(dataset, selection)
        assert_array_equal(dataset.y[selection], dataset_view.y)

    def test_set_y(self, subset_size=10, num_labels=2):
        dataset = self._dataset()
        dataset_view = PytorchDatasetView(dataset, np.s_[0:subset_size])
        with self.assertRaises(UnsupportedOperationException):
            dataset_view.y = np.random.randint(0, high=num_labels, size=subset_size)

    def test_get_target_labels(self, subset_size=10):
        dataset = self._dataset()
        selection = np.random.randint(0, high=len(dataset), size=subset_size)
        dataset_view = PytorchDatasetView(dataset, selection)
        assert_array_equal(dataset.target_labels, dataset_view.target_labels)

    def test_set_target_labels(self):
        dataset = self._dataset()
        with self.assertRaises(UnsupportedOperationException):
            dataset.target_labels = np.array([0])

    def test_get_data(self):
        dataset = self._dataset()
        assert_array_equal(len(dataset), len(dataset.data))

    def test_iter(self):
        ds = self._dataset()
        count = 0
        for _ in ds:
            count += 1
        self.assertEqual(self.NUM_SAMPLES_VIEW, count)

    def test_datasen_len(self):
        ds = self._dataset()
        self.assertEqual(self.NUM_SAMPLES_VIEW, len(ds))


class PytorchDatasetViewTest(unittest.TestCase, _PytorchDatasetViewTest):

    def _dataset(self, num_dimension=20, num_labels=2):
        assert self.NUM_SAMPLES > self.NUM_SAMPLES_VIEW

        vocab = Vocab(Counter({'hello': 3}))
        data = [(torch.randint(10, (num_dimension,)), np.random.randint(0, num_labels))
                for _ in range(self.NUM_SAMPLES)]

        target_labels = np.arange(num_labels)
        ds = PytorchTextClassificationDataset(data, vocab, target_labels=target_labels)
        return PytorchDatasetView(ds, np.s_[:self.NUM_SAMPLES_VIEW])


class NestedPytorchDatasetViewTest(unittest.TestCase, _PytorchDatasetViewTest):

    NUM_SAMPLES_VIEW_OUTER = 17

    def _dataset(self, num_dimension=20, num_labels=2):
        assert self.NUM_SAMPLES > self.NUM_SAMPLES_VIEW_OUTER > self.NUM_SAMPLES_VIEW
        vocab = Vocab(Counter({'hello': 3}))
        data = [(torch.randint(10, (num_dimension,)), np.random.randint(0, num_labels))
                for _ in range(self.NUM_SAMPLES)]

        target_labels = np.arange(num_labels)
        ds = PytorchTextClassificationDataset(data, vocab, target_labels=target_labels)
        ds_view_outer = PytorchDatasetView(ds, np.s_[:self.NUM_SAMPLES_VIEW_OUTER])
        return PytorchDatasetView(ds_view_outer, np.s_[:self.NUM_SAMPLES_VIEW])
