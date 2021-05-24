import unittest

import numpy as np
from parameterized import parameterized_class

from unittest import mock

from numpy.testing import assert_array_equal

from active_learning.data.datasets import SklearnDataSet
from active_learning.data.datasets import split_data
from active_learning.data import balanced_sampling, stratified_sampling

from tests.utils.datasets import random_matrix_data


@parameterized_class([{'matrix_type': 'sparse', 'target_labels': 'explicit'},
                      {'matrix_type': 'sparse', 'target_labels': 'inferred'},
                      {'matrix_type': 'dense', 'target_labels': 'explicit'},
                      {'matrix_type': 'dense', 'target_labels': 'inferred'}])
class SklearnDatasetTest(unittest.TestCase):

    NUM_SAMPLES = 100

    def _dataset(self, num_samples=100, return_data=False):
        x, y = random_matrix_data(self.matrix_type, num_samples=num_samples)
        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError('Invalid test parameter value for target_labels:' + self.target_labels)

        target_labels = None if self.target_labels == 'inferred' else np.unique(y)
        dataset = SklearnDataSet(x, y, target_labels=target_labels)

        if return_data:
            return dataset, x, y
        else:
            return dataset

    # TODO: init

    # TODO: init + target labels

    def test_get_features(self):
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)
        self.assertIsNotNone(ds.y)
        if self.matrix_type == 'dense':
            assert_array_equal(x, ds.x)
        else:
            self.assertTrue((x != ds.x).nnz == 0)

    def test_set_features(self):
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)
        ds_new = self._dataset(num_samples=self.NUM_SAMPLES)
        self.assertIsNotNone(ds.y)
        self.assertIsNotNone(ds_new.y)

        if self.matrix_type == 'dense':
            self.assertFalse((ds.x == ds_new.x).all())
        else:
            self.assertFalse((ds.x != ds_new.x).nnz == 0)

        ds.x = ds_new.x

        if self.matrix_type == 'dense':
            self.assertTrue((ds.x == ds_new.x).all())
        else:
            self.assertTrue((ds.x != ds_new.x).nnz == 0)

    def test_get_labels(self):
        ds, _, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)
        assert_array_equal(y, ds.y)

    def test_set_labels(self):
        ds, _, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)
        ds_new, _, y_new = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)
        self.assertFalse((y == y_new).all())

        ds.y = ds_new.y

        assert_array_equal(y_new, ds.y)

    def test_get_target_labels(self):
        ds = self._dataset(num_samples=self.NUM_SAMPLES)
        expected_target_labels = np.array([0, 1])
        assert_array_equal(expected_target_labels, ds.target_labels)

    def test_set_target_labels(self):
        ds = self._dataset(num_samples=self.NUM_SAMPLES)
        expected_target_labels = np.array([0, 1])
        assert_array_equal(expected_target_labels, ds.target_labels)

        new_target_labels = np.array([2, 3])
        ds.target_labels = new_target_labels
        assert_array_equal(new_target_labels, ds.target_labels)

    def test_indexing_single_index(self):
        index = 42
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[index]
        self.assertEqual(1, len(result))

        if self.matrix_type == 'dense':
            # additional unsqueeze on first dimension
            assert_array_equal(x[np.newaxis, index], result.x)
        else:
            self.assertTrue((x[index] != result.x).nnz == 0)

    def test_indexing_list_index(self):
        index = [1, 42, 56, 99]
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[index]
        self.assertEqual(4, len(result))

        if self.matrix_type == 'dense':
            assert_array_equal(x[index], result.x)
        else:
            self.assertTrue((x[index] != result.x).nnz == 0)

    def test_indexing_slicing(self):
        index = np.s_[10:20]
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[index]
        self.assertEqual(10, len(result))

        if self.matrix_type == 'dense':
            assert_array_equal(x[index], result.x)
        else:
            self.assertTrue((x[index] != result.x).nnz == 0)

    def test_dataset_len(self):
        ds = self._dataset(num_samples=self.NUM_SAMPLES)
        self.assertEqual(self.NUM_SAMPLES, len(ds))


class SplitDataTest(unittest.TestCase):

    @mock.patch('numpy.random.permutation', wraps=np.random.permutation)
    def test_split_data_random(self, permutation_mock):
        train_set = np.random.rand(100, 2)

        subset_train, subset_valid = split_data(train_set)

        self.assertEqual(90, subset_train.shape[0])
        self.assertEqual(10, subset_valid.shape[0])
        permutation_mock.assert_called()

    @mock.patch('numpy.random.permutation', wraps=np.random.permutation)
    def test_split_data_random_return_indices(self, permutation_mock):
        train_set = np.random.rand(100, 2)

        indices_train, indices_valid = split_data(train_set, return_indices=True)

        self.assertEqual(1, len(indices_train.shape))
        self.assertEqual(90, indices_train.shape[0])
        self.assertEqual(10, indices_valid.shape[0])
        permutation_mock.assert_called()

    @mock.patch('active_learning.data.datasets.balanced_sampling',
                wraps=balanced_sampling)
    def test_split_data_balanced(self, balanced_sampling_mock):
        train_set = np.random.rand(100, 2)
        y = np.array([0] * 10 + [1] * 90)

        subset_train, subset_valid = split_data(train_set, y=y, strategy='balanced')

        self.assertEqual(90, subset_train.shape[0])
        self.assertEqual(10, subset_valid.shape[0])
        balanced_sampling_mock.assert_called()

    @mock.patch('active_learning.data.datasets.balanced_sampling',
                wraps=balanced_sampling)
    def test_split_data_balanced_return_indices(self, balanced_sampling_mock):
        train_set = np.random.rand(100, 2)
        y = np.array([0] * 10 + [1] * 90)

        indices_train, indices_valid = split_data(train_set, y=y, strategy='balanced',
                                                  return_indices=True)

        self.assertEqual(1, len(indices_train.shape))
        self.assertEqual(90, indices_train.shape[0])
        self.assertEqual(10, indices_valid.shape[0])
        balanced_sampling_mock.assert_called()

    @mock.patch('active_learning.data.datasets.stratified_sampling',
                wraps=stratified_sampling)
    def test_split_data_balanced(self, stratified_sampling_mock):
        train_set = np.random.rand(100, 2)
        y = np.array([0] * 10 + [1] * 90)

        subset_train, subset_valid = split_data(train_set, y=y, strategy='stratified')

        self.assertEqual(90, subset_train.shape[0])
        self.assertEqual(10, subset_valid.shape[0])
        stratified_sampling_mock.assert_called()

    @mock.patch('active_learning.data.datasets.stratified_sampling',
                wraps=stratified_sampling)
    def test_split_data_stratified_return_indices(self, stratified_sampling_mock):
        train_set = np.random.rand(100, 2)
        y = np.array([0] * 10 + [1] * 90)

        indices_train, indices_valid = split_data(train_set, y=y, strategy='stratified',
                                                  return_indices=True)

        self.assertEqual(1, len(indices_train.shape))
        self.assertEqual(90, indices_train.shape[0])
        self.assertEqual(10, indices_valid.shape[0])
        stratified_sampling_mock.assert_called()

    def test_data_invalid_strategy(self):
        with self.assertRaises(ValueError):
            train_set = np.random.rand(100, 2)
            split_data(train_set, strategy='does_not_exist')
