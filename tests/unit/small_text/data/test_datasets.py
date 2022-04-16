import unittest

import numpy as np
from parameterized import parameterized_class

from unittest import mock

from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from small_text.base import LABEL_UNLABELED

from small_text.data.datasets import is_multi_label
from small_text.data.datasets import SklearnDataset, DatasetView, SklearnDatasetView
from small_text.data.datasets import split_data
from small_text.data.exceptions import UnsupportedOperationException
from small_text.data import balanced_sampling, stratified_sampling

from tests.utils.datasets import random_matrix_data, random_sklearn_dataset
from tests.utils.testing import assert_array_not_equal
from tests.utils.testing import assert_labels_equal


class IsMultiLabelTest(unittest.TestCase):

    def test_is_multi_label_csr(self):
        _, y = random_matrix_data('dense', 'sparse', num_samples=10)
        self.assertTrue(is_multi_label(y))

    def test_is_multi_label_ndarray(self):
        _, y = random_matrix_data('dense', 'dense', num_samples=10)
        self.assertFalse(is_multi_label(y))


@parameterized_class([{'matrix_type': 'sparse', 'labels_type': 'dense', 'target_labels': 'explicit'},
                      {'matrix_type': 'sparse', 'labels_type': 'sparse', 'target_labels': 'explicit'},
                      {'matrix_type': 'sparse', 'labels_type': 'dense', 'target_labels': 'inferred'},
                      {'matrix_type': 'sparse', 'labels_type': 'sparse', 'target_labels': 'inferred'},
                      {'matrix_type': 'dense', 'labels_type': 'dense', 'target_labels': 'explicit'},
                      {'matrix_type': 'dense', 'labels_type': 'sparse', 'target_labels': 'explicit'},
                      {'matrix_type': 'dense', 'labels_type': 'dense', 'target_labels': 'inferred'},
                      {'matrix_type': 'dense', 'labels_type': 'sparse', 'target_labels': 'inferred'}])
class SklearnDatasetTest(unittest.TestCase):

    NUM_SAMPLES = 100

    def _dataset(self, num_samples=100, return_data=False, num_labels=2):
        x, y = random_matrix_data(self.matrix_type, self.labels_type, num_samples=num_samples,
                                  num_labels=num_labels)
        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError('Invalid test parameter value for target_labels:' + self.target_labels)

        if self.labels_type == 'sparse':
            target_labels = None if self.target_labels == 'inferred' else np.unique(y.indices)
        else:
            target_labels = None if self.target_labels == 'inferred' else np.unique(y)
        dataset = SklearnDataset(x, y, target_labels=target_labels)

        if return_data:
            return dataset, x, y
        else:
            return dataset

    def test_init(self):
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        if self.labels_type == 'sparse':
            self.assertTrue(ds.is_multi_label)
        else:
            self.assertFalse(ds.is_multi_label)

    def test_init_when_some_samples_are_unlabeled(self):

        x, y = random_matrix_data(self.matrix_type, self.labels_type, num_samples=self.NUM_SAMPLES)

        if self.labels_type == 'sparse':
            y_data_new = y.data
            y_data_new[0:10] = [LABEL_UNLABELED] * 10
            y = csr_matrix((y_data_new, y.indices, y.indptr))
        else:
            y[0:10] = [LABEL_UNLABELED] * 10

        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError('Invalid test parameter value for target_labels:' + self.target_labels)

        target_labels = np.array([0, 1]) if self.target_labels == 'inferred' else np.unique(y[10:])

        # passes when no exeption is raised here
        SklearnDataset(x, y, target_labels=target_labels)

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
        assert_labels_equal(y, ds.y)

    def test_set_labels(self, num_labels=2):
        ds, _, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True,
                                 num_labels=num_labels)
        ds_new, _, y_new = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True,
                                         num_labels=num_labels+1)
        if self.labels_type == 'sparse':
            self.assertFalse(np.all(y == y_new.data))
        else:
            self.assertFalse((y == y_new).all())

        ds.y = ds_new.y
        assert_labels_equal(y_new, ds.y)
        assert_array_equal(np.arange(num_labels+1), ds.target_labels)

    def test_is_multi_label(self):
        ds = self._dataset(num_samples=self.NUM_SAMPLES)
        if self.labels_type == 'sparse':
            self.assertTrue(ds.is_multi_label)
        else:
            self.assertFalse(ds.is_multi_label)

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

    def test_clone(self):
        ds = self._dataset(num_samples=self.NUM_SAMPLES)
        ds_cloned = ds.clone()

        if self.matrix_type == 'dense':
            self.assertTrue((ds.x == ds_cloned.x).all())
        else:
            self.assertTrue((ds.x != ds_cloned.x).nnz == 0)
        if self.labels_type == 'sparse':
            assert_array_equal(ds.y.indices, ds_cloned.y.indices)
        else:
            assert_array_equal(ds.y, ds_cloned.y)
        assert_array_equal(ds.target_labels, ds_cloned.target_labels)

        # mutability test
        if self.matrix_type == 'dense':
            ds_cloned.x += 1
            self.assertFalse((ds.x == ds_cloned.x).all())
        else:
            ds_cloned.x.data += 1
            self.assertFalse((ds.x != ds_cloned.x).nnz == 0)
        if self.labels_type == 'sparse':
            ds_cloned.y.data += 1
            assert_array_not_equal(ds.y.data, ds_cloned.y.data)
        else:
            ds_cloned.y += 1
            assert_array_not_equal(ds.y, ds_cloned.y)
        ds_cloned.target_labels = np.arange(10)
        assert_array_not_equal(ds.target_labels, ds_cloned.target_labels)

    def test_indexing_single_index(self):
        index = 42
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[index]
        self.assertEqual(1, len(result))
        self.assertTrue(isinstance(result, DatasetView))

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
        self.assertTrue(isinstance(result, DatasetView))

        if self.matrix_type == 'dense':
            assert_array_equal(x[index], result.x)
        else:
            self.assertTrue((x[index] != result.x).nnz == 0)

    def test_indexing_slicing(self):
        index = np.s_[10:20]
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[index]
        self.assertEqual(10, len(result))
        self.assertTrue(isinstance(result, DatasetView))

        if self.matrix_type == 'dense':
            assert_array_equal(x[index], result.x)
        else:
            self.assertTrue((x[index] != result.x).nnz == 0)

    def test_indexing_mutability(self):
        selections = [42, [1, 42, 56, 99], np.s_[10:20]]
        for selection in selections:
            ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)
            assert x.sum() > 0

            dataset_view = ds[selection]

            # flip the signs of the view's data (base dataset should be unchanged)
            dataset_view._dataset.x[selection] = -dataset_view.x

            if self.matrix_type == 'dense':
                # squeeze is only necessary for testing the single index case
                assert_array_equal(x[selection], dataset_view.x.squeeze())
            else:
                self.assertTrue((x[selection] != dataset_view.x).nnz == 0)

            # flip the signs of the base dataset (view should reflect changes)
            ds.x = -ds.x

            if self.matrix_type == 'dense':
                assert_array_not_equal(x[selection], dataset_view.x)
            else:
                self.assertTrue((x[selection] != dataset_view.x).nnz > 0)

    def test_dataset_len(self):
        ds = self._dataset(num_samples=self.NUM_SAMPLES)
        self.assertEqual(self.NUM_SAMPLES, len(ds))


class _DatasetViewTest(object):

    def test_init_with_slice(self):
        dataset = self._dataset()
        self.assertEqual(100, len(dataset))
        dataset_view = SklearnDatasetView(dataset, slice(0, 10))
        self.assertEqual(10, len(dataset_view))

    def test_init_with_slice_and_step(self):
        dataset = self._dataset()
        self.assertEqual(100, len(dataset))
        dataset_view = SklearnDatasetView(dataset, slice(0, 10, 2))
        self.assertEqual(5, len(dataset_view))

    def test_init_with_numpy_slice(self):
        dataset = self._dataset()
        self.assertEqual(100, len(dataset))
        dataset_view = SklearnDatasetView(dataset, np.s_[0:10])
        self.assertEqual(10, len(dataset_view))

    def test_get_dataset(self):
        dataset = self._dataset()
        self.assertEqual(100, len(dataset))
        dataset_view = SklearnDatasetView(dataset, np.s_[0:10])
        self.assertEqual(dataset, dataset_view.dataset)

    def test_get_x(self, subset_size=10):
        dataset = self._dataset()
        selection = np.random.randint(0, high=len(dataset), size=subset_size)
        dataset_view = SklearnDatasetView(dataset, selection)
        if self.matrix_type == 'dense':
            assert_array_equal(dataset.x[selection], dataset_view.x)
        else:
            self.assertTrue((dataset.x[selection] != dataset_view.x).nnz == 0)

    def test_set_x(self, subset_size=10):
        dataset = self._dataset()
        dataset_view = SklearnDatasetView(dataset, np.s_[0:subset_size])
        with self.assertRaises(UnsupportedOperationException):
            dataset_view.x = self._dataset(num_samples=subset_size)

    def test_get_y(self, subset_size=10):
        dataset = self._dataset()
        selection = np.random.randint(0, high=len(dataset), size=subset_size)
        dataset_view = SklearnDatasetView(dataset, selection)
        assert_labels_equal(dataset.y[selection], dataset_view.y)

    def test_set_y(self, subset_size=10, num_labels=2):
        dataset = self._dataset()
        dataset_view = SklearnDatasetView(dataset, np.s_[0:subset_size])
        with self.assertRaises(UnsupportedOperationException):
            dataset_view.y = np.random.randint(0, high=num_labels, size=subset_size)

    def test_is_multi_label(self):
        ds = self._dataset()
        if self.labels_type == 'sparse':
            self.assertTrue(ds.is_multi_label)
        else:
            self.assertFalse(ds.is_multi_label)

    def test_get_target_labels(self, subset_size=10):
        dataset = self._dataset()
        selection = np.random.randint(0, high=len(dataset), size=subset_size)
        dataset_view = SklearnDatasetView(dataset, selection)
        assert_array_equal(dataset.target_labels, dataset_view.target_labels)

    def test_set_target_labels(self, subset_size=10):
        dataset = self._dataset()
        dataset_view = SklearnDatasetView(dataset, np.s_[0:subset_size])
        with self.assertRaises(UnsupportedOperationException):
            dataset_view.target_labels = np.array([0])

    def test_clone_single_index(self):
        ds = self._dataset(num_samples=self.NUM_SAMPLES)

        if self.labels_type == 'sparse':
            # get first row which has at least one label
            indptr_deltas = np.array([ds.y.indptr[i+1] - ds.y.indptr[i]
                                     for i in range(ds.y.indptr.shape[0] - 1)])
            index = np.where(indptr_deltas > 0)[0][0]
        else:
            index = 42
        ds_view = SklearnDatasetView(ds, index)

        self._clone_test(ds_view)

    def test_clone_list_index(self):
        ds = self._dataset(num_samples=self.NUM_SAMPLES)

        if self.labels_type == 'sparse':
            # get first 5 rows which have at least one label
            indptr_deltas = np.array([ds.y.indptr[i+1] - ds.y.indptr[i]
                                     for i in range(ds.y.indptr.shape[0] - 1)])
            index = np.where(indptr_deltas > 0)[0][:5]
        else:
            index = [1, 42, 56, 99]
        ds_view = SklearnDatasetView(ds, index)

        self._clone_test(ds_view)

    def test_clone_slicing(self):
        ds = self._dataset(num_samples=self.NUM_SAMPLES)

        if self.labels_type == 'sparse':
            indptr_deltas = np.array([ds.y.indptr[i+1] - ds.y.indptr[i]
                                     for i in range(ds.y.indptr.shape[0] - 1)])
            index = np.s_[np.where(indptr_deltas > 0)[0][3]:]
        else:
            index = np.s_[10:20]
        ds_view = SklearnDatasetView(ds, index)

        self._clone_test(ds_view)

    def _clone_test(self, ds_view):
        ds_cloned = ds_view.clone()
        self.assertTrue(isinstance(ds_cloned, SklearnDataset))

        if self.matrix_type == 'dense':
            self.assertTrue((ds_view.x == ds_cloned.x).all())
        else:
            self.assertTrue((ds_view.x != ds_cloned.x).nnz == 0)
        if self.labels_type == 'sparse':
            assert_array_equal(ds_view.y.indices, ds_cloned.y.indices)
        else:
            assert_array_equal(ds_view.y, ds_cloned.y)
        assert_array_equal(ds_view.target_labels, ds_cloned.target_labels)

        # mutability test
        if self.matrix_type == 'dense':
            ds_cloned.x += 1
            self.assertFalse((ds_view.x == ds_cloned.x).all())
        else:
            ds_cloned.x.data += 1
            self.assertFalse((ds_view.x != ds_cloned.x).nnz == 0)
        if self.labels_type == 'sparse':
            ds_cloned.y.data += 1
            assert_array_not_equal(ds_view.y.data, ds_cloned.y.data)
        else:
            ds_cloned.y += 1
            assert_array_not_equal(ds_view.y, ds_cloned.y)
        ds_cloned.target_labels = np.arange(10)
        assert_array_not_equal(ds_view.target_labels, ds_cloned.target_labels)

    def test_indexing_single_index(self):
        index = 42
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[index]
        self.assertEqual(1, len(result))
        self.assertTrue(isinstance(result, DatasetView))

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
        self.assertTrue(isinstance(result, DatasetView))

        if self.matrix_type == 'dense':
            assert_array_equal(x[index], result.x)
        else:
            self.assertTrue((x[index] != result.x).nnz == 0)

    def test_indexing_slicing(self):
        index = np.s_[10:20]
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[index]
        self.assertEqual(10, len(result))
        self.assertTrue(isinstance(result, DatasetView))

        if self.matrix_type == 'dense':
            assert_array_equal(x[index], result.x)
        else:
            self.assertTrue((x[index] != result.x).nnz == 0)


@parameterized_class([{'matrix_type': 'sparse', 'labels_type': 'dense'},
                      {'matrix_type': 'sparse', 'labels_type': 'sparse'},
                      {'matrix_type': 'dense', 'labels_type': 'dense'},
                      {'matrix_type': 'dense', 'labels_type': 'sparse'}])
class SklearnDatasetViewTest(unittest.TestCase, _DatasetViewTest):

    NUM_SAMPLES = 100

    # https://github.com/wolever/parameterized/issues/119
    @classmethod
    def setUpClass(cls):
        if cls == SklearnDatasetViewTest:
            raise unittest.SkipTest('parameterized_class bug')
        super().setUpClass()

    def _dataset(self, num_samples=100, return_data=False):
        x, y = random_matrix_data(self.matrix_type, self.labels_type, num_samples=num_samples)
        if isinstance(y, csr_matrix):
            target_labels = np.unique(y.indices)
        else:
            target_labels = np.unique(y)

        dataset = SklearnDataset(x, y, target_labels=target_labels)

        if return_data:
            return dataset, x, y
        else:
            return dataset


@parameterized_class([{'matrix_type': 'sparse', 'labels_type': 'dense'},
                      {'matrix_type': 'sparse', 'labels_type': 'sparse'},
                      {'matrix_type': 'dense', 'labels_type': 'dense'},
                      {'matrix_type': 'dense', 'labels_type': 'sparse'}])
class NestedDatasetViewTest(unittest.TestCase, _DatasetViewTest):

    NUM_SAMPLES = 100

    # https://github.com/wolever/parameterized/issues/119
    @classmethod
    def setUpClass(cls):
        if cls == NestedDatasetViewTest:
            raise unittest.SkipTest('parameterized_class bug')
        super().setUpClass()

    def _dataset(self, num_samples=100, return_data=False):
        x, y = random_matrix_data(self.matrix_type, self.labels_type, num_samples=num_samples)
        if isinstance(y, csr_matrix):
            target_labels = np.unique(y.indices)
        else:
            target_labels = np.unique(y)

        dataset_view = SklearnDatasetView(
            SklearnDataset(x, y, target_labels=target_labels),
            np.s_[:]
        )

        if return_data:
            return dataset_view, x, y
        else:
            return dataset_view


class SplitDataTest(unittest.TestCase):

    def test_split_data_invalid_strategy(self):
        with self.assertRaises(ValueError):
            train_set = random_sklearn_dataset(100)
            split_data(train_set, strategy='does_not_exist')

    def test_split_data_invalid_validation_set_sizes(self):
        expected_regex = 'Invalid value encountered for "validation_set_size".'
        with self.assertRaisesRegex(ValueError, expected_regex):
            train_set = random_sklearn_dataset(100)
            split_data(train_set, validation_set_size=0.0)

        with self.assertRaisesRegex(ValueError, expected_regex):
            train_set = random_sklearn_dataset(100)
            split_data(train_set, validation_set_size=1.0)

    @mock.patch('numpy.random.permutation', wraps=np.random.permutation)
    def test_split_data_random(self, permutation_mock):
        train_set = random_sklearn_dataset(100)

        subset_train, subset_valid = split_data(train_set)

        self.assertEqual(90, len(subset_train))
        self.assertEqual(10, len(subset_valid))
        permutation_mock.assert_called()

    @mock.patch('numpy.random.permutation', wraps=np.random.permutation)
    def test_split_data_random_return_indices(self, permutation_mock):
        train_set = random_sklearn_dataset(100)

        indices_train, indices_valid = split_data(train_set, return_indices=True)

        self.assertEqual(1, len(indices_train.shape))
        self.assertEqual(90, indices_train.shape[0])
        self.assertEqual(10, indices_valid.shape[0])
        permutation_mock.assert_called()

    @mock.patch('small_text.data.datasets.balanced_sampling',
                wraps=balanced_sampling)
    def test_split_data_balanced(self, balanced_sampling_mock):
        train_set = random_sklearn_dataset(100)
        y = np.array([0] * 10 + [1] * 90)

        subset_train, subset_valid = split_data(train_set, y=y, strategy='balanced')

        self.assertEqual(90, len(subset_train))
        self.assertEqual(10, len(subset_valid))
        balanced_sampling_mock.assert_called()

    @mock.patch('small_text.data.datasets.balanced_sampling',
                wraps=balanced_sampling)
    def test_split_data_balanced_return_indices(self, balanced_sampling_mock):
        train_set = random_sklearn_dataset(100)
        y = np.array([0] * 10 + [1] * 90)

        indices_train, indices_valid = split_data(train_set, y=y, strategy='balanced',
                                                  return_indices=True)

        self.assertEqual(1, len(indices_train.shape))
        self.assertEqual(90, indices_train.shape[0])
        self.assertEqual(10, indices_valid.shape[0])
        balanced_sampling_mock.assert_called()

    @mock.patch('small_text.data.datasets.stratified_sampling',
                wraps=stratified_sampling)
    def test_split_data_stratified(self, stratified_sampling_mock):
        train_set = random_sklearn_dataset(100)
        y = np.array([0] * 10 + [1] * 90)

        subset_train, subset_valid = split_data(train_set, y=y, strategy='stratified')

        self.assertEqual(90, len(subset_train))
        self.assertEqual(10, len(subset_valid))
        stratified_sampling_mock.assert_called()

    @mock.patch('small_text.data.datasets.stratified_sampling',
                wraps=stratified_sampling)
    def test_split_data_stratified_return_indices(self, stratified_sampling_mock):
        train_set = random_sklearn_dataset(100)
        y = np.array([0] * 10 + [1] * 90)

        indices_train, indices_valid = split_data(train_set, y=y, strategy='stratified',
                                                  return_indices=True)

        self.assertEqual(1, len(indices_train.shape))
        self.assertEqual(90, indices_train.shape[0])
        self.assertEqual(10, indices_valid.shape[0])
        stratified_sampling_mock.assert_called()
