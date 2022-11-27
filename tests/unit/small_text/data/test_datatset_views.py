import unittest
import numpy as np

from copy import copy
from unittest import mock

from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from small_text.data.datasets import (
    DatasetView,
    SklearnDataset,
    SklearnDatasetView,
    TextDataset,
    TextDatasetView
)
from small_text.data.datasets import split_data
from small_text.data.exceptions import UnsupportedOperationException
from small_text.data import balanced_sampling, stratified_sampling

from tests.utils.datasets import (
    random_labels,
    random_matrix_data,
    random_sklearn_dataset,
    random_text_data
)
from tests.utils.testing import (
    assert_array_not_equal,
    assert_labels_equal
)


class _DatasetViewTest(object):

    def test_init_with_slice(self):
        dataset = self._dataset()
        self.assertEqual(100, len(dataset))
        dataset_view = self.view_class(dataset, slice(0, 10))
        self.assertEqual(10, len(dataset_view))

    def test_init_with_slice_and_step(self):
        dataset = self._dataset()
        self.assertEqual(100, len(dataset))
        dataset_view = self.view_class(dataset, slice(0, 10, 2))
        self.assertEqual(5, len(dataset_view))

    def test_init_with_numpy_slice(self):
        dataset = self._dataset()
        self.assertEqual(100, len(dataset))
        dataset_view = self.view_class(dataset, np.s_[0:10])
        self.assertEqual(10, len(dataset_view))

    def test_get_dataset(self):
        dataset = self._dataset()
        self.assertEqual(100, len(dataset))
        dataset_view = self.view_class(dataset, np.s_[0:10])
        self.assertEqual(dataset, dataset_view.dataset)

    def test_get_x(self, subset_size=10):
        dataset = self._dataset()
        selection = np.random.randint(0, high=len(dataset), size=subset_size)
        dataset_view = self.view_class(dataset, selection)
        if self.matrix_type == 'dense':
            assert_array_equal(dataset.x[selection], dataset_view.x)
        else:
            self.assertTrue((dataset.x[selection] != dataset_view.x).nnz == 0)

    def test_set_x(self, subset_size=10):
        dataset = self._dataset()
        dataset_view = self.view_class(dataset, np.s_[0:subset_size])
        with self.assertRaises(UnsupportedOperationException):
            dataset_view.x = self._dataset(num_samples=subset_size)

    def test_get_y(self, subset_size=10):
        dataset = self._dataset()
        selection = np.random.randint(0, high=len(dataset), size=subset_size)
        dataset_view = self.view_class(dataset, selection)
        assert_labels_equal(dataset.y[selection], dataset_view.y)

    def test_set_y(self, subset_size=10, num_labels=2):
        dataset = self._dataset()
        dataset_view = self.view_class(dataset, np.s_[0:subset_size])
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
        dataset_view = self.view_class(dataset, selection)
        assert_array_equal(dataset.target_labels, dataset_view.target_labels)

    def test_set_target_labels(self, subset_size=10):
        dataset = self._dataset()
        dataset_view = self.view_class(dataset, np.s_[0:subset_size])
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
        ds_view = self.view_class(ds, index)

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
        ds_view = self.view_class(ds, index)

        self._clone_test(ds_view)

    def test_clone_slicing(self):
        ds = self._dataset(num_samples=self.NUM_SAMPLES)

        if self.labels_type == 'sparse':
            indptr_deltas = np.array([ds.y.indptr[i+1] - ds.y.indptr[i]
                                     for i in range(ds.y.indptr.shape[0] - 1)])
            index = np.s_[np.where(indptr_deltas > 0)[0][3]:]
        else:
            index = np.s_[10:20]
        ds_view = self.view_class(ds, index)

        self._clone_test(ds_view)

    def test_clone_target_labels_inferred(self):
        ds = self._dataset(num_samples=self.NUM_SAMPLES)

        if self.labels_type == 'sparse':
            # get first 5 rows which have at least one label
            indptr_deltas = np.array([ds.y.indptr[i+1] - ds.y.indptr[i]
                                     for i in range(ds.y.indptr.shape[0] - 1)])
            index = np.where(indptr_deltas > 0)[0][:5]
        else:
            index = [1, 42, 56, 99]
        ds_view = self.view_class(ds, index)

        self._clone_test(ds_view)

    def _clone_test(self, ds_view):
        ds_cloned = ds_view.clone()
        self.assertTrue(isinstance(ds_cloned, SklearnDataset))

        if self.matrix_type == 'dense':
            self.assertTrue((ds_view.x == ds_cloned.x).all())
        else:
            self.assertTrue((ds_view.x != ds_cloned.x).nnz == 0)
        if self.labels_type == 'sparse':
            if self.target_labels == 'explicit':
                assert_array_equal(ds_view.y.indices, ds_cloned.y.indices)
        else:
            assert_array_equal(ds_view.y, ds_cloned.y)
        if self.target_labels == 'explicit':
            assert_array_equal(ds_view.target_labels, ds_cloned.target_labels)

        # test propagation of target_labels setting
        if self.target_labels == 'explicit':
            self.assertFalse(ds_view.dataset.track_target_labels)
            self.assertFalse(ds_cloned.track_target_labels)
            assert_array_equal(ds_view.target_labels, ds_cloned.target_labels)
        else:
            self.assertTrue(ds_view.dataset.track_target_labels)
            self.assertTrue(ds_cloned.track_target_labels)

        # mutability test
        if self.matrix_type == 'dense':
            ds_cloned.x = (ds_cloned.x + 1) % 2
            self.assertFalse((ds_view.x == ds_cloned.x).all())
        else:
            ds_cloned.x.indices = (ds_cloned.x.indices + 1) % 2
            self.assertFalse((ds_view.x != ds_cloned.x).nnz == 0)
        if self.labels_type == 'sparse':
            ds_cloned.y.indices = (ds_cloned.y.indices + 1) % 2
            assert_array_not_equal(ds_view.y.indices, ds_cloned.y.indices)
        else:
            ds_cloned.y = (ds_cloned.y + 1) % 2
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
        selection = [1, 42, 56, 99]
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[selection]
        self.assertEqual(4, len(result))
        self.assertTrue(isinstance(result, DatasetView))

        if self.matrix_type == 'dense':
            assert_array_equal(x[selection], result.x)
        else:
            self.assertTrue((x[selection] != result.x).nnz == 0)

    def test_indexing_slicing(self):
        selection = np.s_[10:20]
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[selection]
        self.assertEqual(10, len(result))
        self.assertTrue(isinstance(result, DatasetView))

        if self.matrix_type == 'dense':
            assert_array_equal(x[selection], result.x)
        else:
            self.assertTrue((x[selection] != result.x).nnz == 0)


class _SklearnDatasetViewTest(_DatasetViewTest):

    NUM_SAMPLES = 100

    def _dataset(self, num_samples=NUM_SAMPLES, return_data=False):
        """Creates a dataset (and not a view) which is used by the tests for creating views.
        """
        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError(f'Invalid value: self.target_labels={self.target_labels}')

        x, y = random_matrix_data(self.matrix_type, self.labels_type, num_samples=num_samples)
        if self.target_labels == 'explicit':
            if isinstance(y, csr_matrix):
                target_labels = np.unique(y.indices)
            else:
                target_labels = np.unique(y)
        else:
            target_labels = None

        dataset = SklearnDataset(x, y, target_labels=target_labels)

        if return_data:
            return dataset, x, y
        else:
            return dataset


class SklearnDatasetViewSparseDenseExplicitTest(unittest.TestCase, _SklearnDatasetViewTest):

    def setUp(self):
        self.view_class = SklearnDatasetView
        self.matrix_type = 'sparse'
        self.labels_type = 'dense'
        self.target_labels = 'explicit'


class SklearnDatasetViewSparseSparseExplicitTest(unittest.TestCase, _SklearnDatasetViewTest):

    def setUp(self):
        self.view_class = SklearnDatasetView
        self.matrix_type = 'sparse'
        self.labels_type = 'sparse'
        self.target_labels = 'explicit'


class SklearnDatasetViewSparseDenseInferredTest(unittest.TestCase, _SklearnDatasetViewTest):

    def setUp(self):
        self.view_class = SklearnDatasetView
        self.matrix_type = 'sparse'
        self.labels_type = 'dense'
        self.target_labels = 'inferred'


class SklearnDatasetViewSparseSparseInferredTest(unittest.TestCase, _SklearnDatasetViewTest):

    def setUp(self):
        self.view_class = SklearnDatasetView
        self.matrix_type = 'sparse'
        self.labels_type = 'sparse'
        self.target_labels = 'inferred'


class SklearnDatasetViewDenseDenseExplicitTest(unittest.TestCase, _SklearnDatasetViewTest):

    def setUp(self):
        self.view_class = SklearnDatasetView
        self.matrix_type = 'dense'
        self.labels_type = 'dense'
        self.target_labels = 'explicit'


class SklearnDatasetViewDenseSparseExplicitTest(unittest.TestCase, _SklearnDatasetViewTest):

    def setUp(self):
        self.view_class = SklearnDatasetView
        self.matrix_type = 'dense'
        self.labels_type = 'sparse'
        self.target_labels = 'explicit'


class SklearnDatasetViewDenseDenseInferredTest(unittest.TestCase, _SklearnDatasetViewTest):

    def setUp(self):
        self.view_class = SklearnDatasetView
        self.matrix_type = 'dense'
        self.labels_type = 'dense'
        self.target_labels = 'inferred'


class SklearnDatasetViewDenseSparseInferredTest(unittest.TestCase, _SklearnDatasetViewTest):

    def setUp(self):
        self.view_class = SklearnDatasetView
        self.matrix_type = 'dense'
        self.labels_type = 'sparse'
        self.target_labels = 'inferred'


class _TextDatasetViewTest(_DatasetViewTest):
    """Creates a dataset (and not a view) which is used by the tests for creating views.
    """

    NUM_SAMPLES = 100

    def _dataset(self, num_samples=NUM_SAMPLES, return_data=False):
        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError(f'Invalid value: self.target_labels={self.target_labels}')

        x = random_text_data(num_samples)
        y = random_labels(num_samples, 3, multi_label=self.labels_type == 'sparse')

        if self.target_labels == 'explicit':
            if isinstance(y, csr_matrix):
                target_labels = np.unique(y.indices)
            else:
                target_labels = np.unique(y)
        else:
            target_labels = None

        dataset = TextDataset(x, y, target_labels=target_labels)

        if return_data:
            return dataset, x, y
        else:
            return dataset

    def _clone_test(self, ds_view):
        ds_cloned = ds_view.clone()
        self.assertTrue(isinstance(ds_cloned, TextDataset))

        assert_array_equal(ds_view.x, ds_cloned.x)
        if self.labels_type == 'sparse':
            if self.target_labels == 'explicit':
                assert_array_equal(ds_view.y.indices, ds_cloned.y.indices)
        else:
            assert_array_equal(ds_view.y, ds_cloned.y)
        if self.target_labels == 'explicit':
            assert_array_equal(ds_view.target_labels, ds_cloned.target_labels)

        # test propagation of target_labels setting
        if self.target_labels == 'explicit':
            self.assertFalse(ds_view.dataset.track_target_labels)
            self.assertFalse(ds_cloned.track_target_labels)
            assert_array_equal(ds_view.target_labels, ds_cloned.target_labels)
        else:
            self.assertTrue(ds_view.dataset.track_target_labels)
            self.assertTrue(ds_cloned.track_target_labels)

        x_changed = copy(ds_cloned.x)
        x_changed[0] = 'this sentence has been changed'
        ds_cloned.x = x_changed
        assert_array_not_equal(ds_view.x, ds_cloned.x)

        if self.labels_type == 'sparse':
            ds_cloned.y.indices = (ds_cloned.y.indices + 1) % 2
            assert_array_not_equal(ds_view.y.indices, ds_cloned.y.indices)
        else:
            ds_cloned.y = (ds_cloned.y + 1) % 2
            assert_array_not_equal(ds_view.y, ds_cloned.y)
        ds_cloned.target_labels = np.arange(10)
        assert_array_not_equal(ds_view.target_labels, ds_cloned.target_labels)

    def test_get_x(self, subset_size=10):
        dataset = self._dataset()
        selection = np.random.randint(0, high=len(dataset), size=subset_size)
        dataset_view = self.view_class(dataset, selection)

        assert_array_equal([dataset.x[i] for i in selection], dataset_view.x)

    def test_indexing_single_index(self):
        index = 42
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[index]
        self.assertEqual(1, len(result))
        self.assertTrue(isinstance(result, DatasetView))

        assert_array_equal(x[index], result.x)

    def test_indexing_list_index(self):
        selection = [1, 42, 56, 99]
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[selection]
        self.assertEqual(4, len(result))
        self.assertTrue(isinstance(result, DatasetView))

        assert_array_equal([x[i] for i in selection], result.x)

    def test_indexing_slicing(self):
        selection = np.s_[10:20]
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[selection]
        self.assertEqual(10, len(result))
        self.assertTrue(isinstance(result, DatasetView))

        assert_array_equal([x[i] for i in np.arange(len(ds))[selection]], result.x)


class TextDatasetViewDenseExplicitTest(unittest.TestCase, _TextDatasetViewTest):

    def setUp(self):
        self.view_class = TextDatasetView
        self.labels_type = 'dense'
        self.target_labels = 'explicit'


class TextDatasetViewSparseExplicitTest(unittest.TestCase, _TextDatasetViewTest):

    def setUp(self):
        self.view_class = TextDatasetView
        self.labels_type = 'sparse'
        self.target_labels = 'explicit'


class TextDatasetViewDenseInferredTest(unittest.TestCase, _TextDatasetViewTest):

    def setUp(self):
        self.view_class = TextDatasetView
        self.labels_type = 'dense'
        self.target_labels = 'inferred'


class TextDatasetViewSparseInferredTest(unittest.TestCase, _TextDatasetViewTest):

    def setUp(self):
        self.view_class = TextDatasetView
        self.labels_type = 'sparse'
        self.target_labels = 'inferred'


class _NestedSklearnDatasetViewTest(_DatasetViewTest):

    NUM_SAMPLES = 100

    def setUp(self):
        self.view_class = SklearnDatasetView

    def _dataset(self, num_samples=100, return_data=False):
        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError(f'Invalid value: self.target_labels={self.target_labels}')

        x, y = random_matrix_data(self.matrix_type, self.labels_type, num_samples=num_samples)
        if self.target_labels == 'explicit':
            if isinstance(y, csr_matrix):
                target_labels = np.unique(y.indices)
            else:
                target_labels = np.unique(y)
        else:
            target_labels = None

        dataset_view = SklearnDatasetView(
            SklearnDataset(x, y, target_labels=target_labels),
            np.s_[:]
        )

        if return_data:
            return dataset_view, x, y
        else:
            return dataset_view

    def _clone_test(self, ds_view):
        ds_cloned = ds_view.clone()
        self.assertTrue(isinstance(ds_cloned, SklearnDataset))

        if self.matrix_type == 'dense':
            self.assertTrue((ds_view.x == ds_cloned.x).all())
        else:
            self.assertTrue((ds_view.x != ds_cloned.x).nnz == 0)
        if self.labels_type == 'sparse':
            if self.target_labels == 'explicit':
                assert_array_equal(ds_view.y.indices, ds_cloned.y.indices)
        else:
            assert_array_equal(ds_view.y, ds_cloned.y)
        if self.target_labels == 'explicit':
            assert_array_equal(ds_view.target_labels, ds_cloned.target_labels)

        # test propagation of target_labels setting
        if self.target_labels == 'explicit':
            self.assertFalse(ds_cloned.track_target_labels)
            assert_array_equal(ds_view.target_labels, ds_cloned.target_labels)
        else:
            self.assertTrue(ds_cloned.track_target_labels)

        # mutability test
        if self.matrix_type == 'dense':
            ds_cloned.x = (ds_cloned.x + 1) % 2
            self.assertFalse((ds_view.x == ds_cloned.x).all())
        else:
            ds_cloned.x.indices = (ds_cloned.x.indices + 1) % 2
            self.assertFalse((ds_view.x != ds_cloned.x).nnz == 0)
        if self.labels_type == 'sparse':
            ds_cloned.y.indices = (ds_cloned.y.indices + 1) % 2
            assert_array_not_equal(ds_view.y.indices, ds_cloned.y.indices)
        else:
            ds_cloned.y = (ds_cloned.y + 1) % 2
            assert_array_not_equal(ds_view.y, ds_cloned.y)
        ds_cloned.target_labels = np.arange(10)
        assert_array_not_equal(ds_view.target_labels, ds_cloned.target_labels)


class NestedSklearnDatasetViewSparseDenseExplicitTest(unittest.TestCase,
                                                      _NestedSklearnDatasetViewTest):

    def setUp(self):
        self.view_class = SklearnDatasetView
        self.matrix_type = 'sparse'
        self.labels_type = 'dense'
        self.target_labels = 'explicit'


class NestedSklearnDatasetViewSparseSparseExplicitTest(unittest.TestCase,
                                                       _NestedSklearnDatasetViewTest):

    def setUp(self):
        self.view_class = SklearnDatasetView
        self.matrix_type = 'sparse'
        self.labels_type = 'sparse'
        self.target_labels = 'explicit'


class NestedSklearnDatasetViewSparseDenseInferredTest(unittest.TestCase,
                                                      _NestedSklearnDatasetViewTest):

    def setUp(self):
        self.view_class = SklearnDatasetView
        self.matrix_type = 'sparse'
        self.labels_type = 'dense'
        self.target_labels = 'inferred'


class NestedSklearnDatasetViewSparseSparseInferredTest(unittest.TestCase,
                                                       _NestedSklearnDatasetViewTest):

    def setUp(self):
        self.view_class = SklearnDatasetView
        self.matrix_type = 'sparse'
        self.labels_type = 'sparse'
        self.target_labels = 'inferred'


class NestedSklearnDatasetViewDenseDenseExplicitTest(unittest.TestCase,
                                                     _NestedSklearnDatasetViewTest):

    def setUp(self):
        self.view_class = SklearnDatasetView
        self.matrix_type = 'dense'
        self.labels_type = 'dense'
        self.target_labels = 'explicit'


class NestedSklearnDatasetViewDenseSparseExplicitTest(unittest.TestCase,
                                                      _NestedSklearnDatasetViewTest):

    def setUp(self):
        self.view_class = SklearnDatasetView
        self.matrix_type = 'dense'
        self.labels_type = 'sparse'
        self.target_labels = 'explicit'


class NestedSklearnDatasetViewDenseDenseInferredTest(unittest.TestCase,
                                                     _NestedSklearnDatasetViewTest):

    def setUp(self):
        self.view_class = SklearnDatasetView
        self.matrix_type = 'dense'
        self.labels_type = 'dense'
        self.target_labels = 'inferred'


class NestedSklearnDatasetViewDenseSparseInferredTest(unittest.TestCase,
                                                      _NestedSklearnDatasetViewTest):

    def setUp(self):
        self.view_class = SklearnDatasetView
        self.matrix_type = 'dense'
        self.labels_type = 'sparse'
        self.target_labels = 'inferred'


class _NestedTextDatasetViewTest(_DatasetViewTest):

    NUM_SAMPLES = 100

    def setUp(self):
        self.view_class = TextDatasetView

    def _dataset(self, num_samples=100, return_data=False):
        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError(f'Invalid value: self.target_labels={self.target_labels}')

        x = random_text_data(num_samples)
        y = random_labels(num_samples, 3, multi_label=self.labels_type == 'sparse')

        if self.target_labels == 'explicit':
            if isinstance(y, csr_matrix):
                target_labels = np.unique(y.indices)
            else:
                target_labels = np.unique(y)
        else:
            target_labels = None

        dataset_view = TextDatasetView(
            TextDataset(x, y, target_labels=target_labels),
            np.s_[:]
        )

        if return_data:
            return dataset_view, x, y
        else:
            return dataset_view

    def _clone_test(self, ds_view):
        ds_cloned = ds_view.clone()
        self.assertTrue(isinstance(ds_cloned, TextDataset))

        assert_array_equal(ds_view.x, ds_cloned.x)
        if self.labels_type == 'sparse':
            if self.target_labels == 'explicit':
                assert_array_equal(ds_view.y.indices, ds_cloned.y.indices)
        else:
            assert_array_equal(ds_view.y, ds_cloned.y)
        if self.target_labels == 'explicit':
            assert_array_equal(ds_view.target_labels, ds_cloned.target_labels)

        # test propagation of target_labels setting
        if self.target_labels == 'explicit':
            self.assertFalse(ds_cloned.track_target_labels)
            assert_array_equal(ds_view.target_labels, ds_cloned.target_labels)
        else:
            self.assertTrue(ds_cloned.track_target_labels)

        # mutability test
        x_changed = copy(ds_cloned.x)
        x_changed[0] = 'this sentence has been changed'
        ds_cloned.x = x_changed
        assert_array_not_equal(ds_view.x, ds_cloned.x)

        if self.labels_type == 'sparse':
            ds_cloned.y.indices = (ds_cloned.y.indices + 1) % 2
            assert_array_not_equal(ds_view.y.indices, ds_cloned.y.indices)
        else:
            ds_cloned.y = (ds_cloned.y + 1) % 2
            assert_array_not_equal(ds_view.y, ds_cloned.y)
        ds_cloned.target_labels = np.arange(10)
        assert_array_not_equal(ds_view.target_labels, ds_cloned.target_labels)

    def test_get_x(self, subset_size=10):
        dataset = self._dataset()
        selection = np.random.randint(0, high=len(dataset), size=subset_size)
        dataset_view = self.view_class(dataset, selection)
        assert_array_equal([dataset.x[i] for i in selection], dataset_view.x)

    def test_indexing_single_index(self):
        index = 42
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[index]
        self.assertEqual(1, len(result))
        self.assertTrue(isinstance(result, DatasetView))

        assert_array_equal([x[index]], result.x)

    def test_indexing_list_index(self):
        selection = [1, 42, 56, 99]
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[selection]
        self.assertEqual(4, len(result))
        self.assertTrue(isinstance(result, DatasetView))

        assert_array_equal([x[i] for i in selection], result.x)

    def test_indexing_slicing(self):
        selection = np.s_[10:20]
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[selection]
        self.assertEqual(10, len(result))
        self.assertTrue(isinstance(result, DatasetView))

        assert_array_equal([x[i] for i in np.arange(len(ds))[selection]], result.x)


class NestedTextDatasetViewDenseExplicitTest(unittest.TestCase,
                                             _NestedTextDatasetViewTest):

    def setUp(self):
        self.view_class = TextDatasetView
        self.labels_type = 'dense'
        self.target_labels = 'explicit'


class NestedTextDatasetViewSparseExplicitTest(unittest.TestCase,
                                              _NestedTextDatasetViewTest):

    def setUp(self):
        self.view_class = TextDatasetView
        self.labels_type = 'sparse'
        self.target_labels = 'explicit'


class NestedTextDatasetViewDenseInferredTest(unittest.TestCase,
                                             _NestedTextDatasetViewTest):

    def setUp(self):
        self.view_class = TextDatasetView
        self.labels_type = 'dense'
        self.target_labels = 'inferred'


class NestedTextDatasetViewSparseInferredTest(unittest.TestCase,
                                              _NestedTextDatasetViewTest):

    def setUp(self):
        self.view_class = TextDatasetView
        self.labels_type = 'sparse'
        self.target_labels = 'inferred'


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
