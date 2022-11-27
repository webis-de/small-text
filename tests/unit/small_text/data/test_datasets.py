import unittest

import numpy as np

from copy import copy
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from small_text.base import LABEL_UNLABELED

from small_text.data.datasets import is_multi_label
from small_text.data.datasets import (
    SklearnDataset,
    DatasetView,
    TextDataset
)
from small_text.utils.labels import csr_to_list, list_to_csr

from tests.utils.datasets import (
    random_labels,
    random_matrix_data,
    random_text_data
)
from tests.utils.testing import (
    assert_array_not_equal,
    assert_labels_equal,
    assert_csr_matrix_not_equal
)


class IsMultiLabelTest(unittest.TestCase):

    def test_is_multi_label_csr(self):
        _, y = random_matrix_data('dense', 'sparse', num_samples=10)
        self.assertTrue(is_multi_label(y))

    def test_is_multi_label_ndarray(self):
        _, y = random_matrix_data('dense', 'dense', num_samples=10)
        self.assertFalse(is_multi_label(y))


class _DatasetTest(object):

    NUM_LABELS = 2
    NUM_SAMPLES = 100

    def test_init(self):
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        if self.labels_type == 'sparse':
            self.assertTrue(ds.is_multi_label)
        else:
            self.assertFalse(ds.is_multi_label)

    def test_init_with_target_label_warning(self):
        if self.target_labels == 'inferred':
            with self.assertWarnsRegex(UserWarning, 'Passing target_labels=None is discouraged'):
                self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

    def test_init_with_dimension_mismatch(self):
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        if self.labels_type == 'sparse':
            mask = np.ones(self.NUM_SAMPLES, dtype=bool)
            mask[0] = False
            y = y[mask]
        else:
            y = np.delete(y, 0)

        with self.assertRaisesRegex(ValueError, 'Feature and label dimensions do not match'):
            SklearnDataset(x, y, target_labels=ds.target_labels)

    # TODO: copy to kimcnn / transformers
    def test_init_when_some_samples_are_unlabeled(self):
        x, y = random_matrix_data(self.matrix_type, self.labels_type, num_samples=self.NUM_SAMPLES)

        if self.labels_type == 'sparse':
            y_list = csr_to_list(y)
            y_list[1] = []
            y_list[2] = []
            y = list_to_csr(y_list, y.shape)
        else:
            y[0:10] = [LABEL_UNLABELED] * 10

        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError('Invalid test parameter value for target_labels:' + self.target_labels)

        target_labels = None if self.target_labels == 'inferred' else np.arange(5)

        # passes when no exeption is raised here
        SklearnDataset(x, y, target_labels=target_labels)

    # TODO: copy to kimcnn / transformers
    def test_init_when_all_samples_are_unlabeled(self):
        x, y = random_matrix_data(self.matrix_type, self.labels_type, num_samples=self.NUM_SAMPLES)

        if self.labels_type == 'sparse':
            y = csr_matrix(y.shape, dtype=y.dtype)
        else:
            y[:] = LABEL_UNLABELED

        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError('Invalid test parameter value for target_labels:' + self.target_labels)

        target_labels = None if self.target_labels == 'inferred' else np.arange(5)

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

    def test_set_features_with_dimension_mismatch(self):
        ds, x, _ = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        if self.matrix_type == 'sparse':
            mask = np.ones(self.NUM_SAMPLES, dtype=bool)
            mask[0] = False
            x = x[mask]
        else:
            x = np.delete(x, 0)

        with self.assertRaisesRegex(ValueError, 'Feature and label dimensions do not match'):
            ds.x = x

    def test_get_labels(self):
        ds, _, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)
        assert_labels_equal(y, ds.y)

    def test_set_labels(self, num_labels=2):
        ds, _, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True,
                                 num_labels=num_labels)
        if self.labels_type == 'sparse':
            y_new = ds.y.copy()
            y_new.indices = (y_new.indices + 1) % self.NUM_LABELS
            assert_csr_matrix_not_equal(ds.y, y_new)
        else:
            y_new = (ds.y + 1) % self.NUM_LABELS
            assert_array_not_equal(ds.y, y_new)

        ds.y = y_new
        assert_labels_equal(y_new, ds.y)

    def test_set_labels_with_dimension_mismatch(self):
        ds, _, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        if self.labels_type == 'sparse':
            mask = np.ones(self.NUM_SAMPLES, dtype=bool)
            mask[0] = False
            y = y[mask]
        else:
            y = np.delete(y, 0)

        with self.assertRaisesRegex(ValueError, 'Feature and label dimensions do not match'):
            ds.y = y

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

        new_target_labels = np.array([0])
        with self.assertRaisesRegex(ValueError, 'Cannot remove existing labels'):
            ds.target_labels = new_target_labels

        new_target_labels = np.array([0])

        if self.labels_type == 'sparse':
            y_new = ds.y.copy()
            y_new = y_new[:, 0]
            ds.y = y_new
        else:
            ds.y = np.array([0] * len(ds))
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
        if self.target_labels == 'explicit':
            assert_array_equal(ds.target_labels, ds_cloned.target_labels)

        # test propagation of target_labels setting
        if self.target_labels == 'explicit':
            self.assertFalse(ds.track_target_labels)
            self.assertFalse(ds_cloned.track_target_labels)
        else:
            self.assertTrue(ds.track_target_labels)
            self.assertTrue(ds_cloned.track_target_labels)
        assert_array_equal(ds.target_labels, ds_cloned.target_labels)

        # mutability test
        if self.matrix_type == 'dense':
            ds_cloned.x += 1
            self.assertFalse((ds.x == ds_cloned.x).all())
        else:
            ds_cloned.x.data += 1
            self.assertFalse((ds.x != ds_cloned.x).nnz == 0)
        if self.labels_type == 'sparse':
            ds_cloned.y.indices = (ds_cloned.y.indices + 1) % self.NUM_LABELS
            assert_array_not_equal(ds.y.indices, ds_cloned.y.indices)
        else:
            ds_cloned.y = (ds_cloned.y + 1) % self.NUM_LABELS
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

    def test_dataset_len(self):
        ds = self._dataset(num_samples=self.NUM_SAMPLES)
        self.assertEqual(self.NUM_SAMPLES, len(ds))


class _SklearnDatasetTest(_DatasetTest):

    def _dataset(self, num_samples=100, return_data=False, num_labels=_DatasetTest.NUM_LABELS):
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


class SklearnDatasetSparseDenseExplicitTest(unittest.TestCase, _SklearnDatasetTest):

    def setUp(self):
        self.matrix_type = 'sparse'
        self.labels_type = 'dense'
        self.target_labels = 'explicit'


class SklearnDatasetSparseSparseExplicitTest(unittest.TestCase, _SklearnDatasetTest):

    def setUp(self):
        self.matrix_type = 'sparse'
        self.labels_type = 'sparse'
        self.target_labels = 'explicit'


class SklearnDatasetSparseDenseInferredTest(unittest.TestCase, _SklearnDatasetTest):

    def setUp(self):
        self.matrix_type = 'sparse'
        self.labels_type = 'dense'
        self.target_labels = 'inferred'


class SklearnDatasetSparseSparseInferredTest(unittest.TestCase, _SklearnDatasetTest):

    def setUp(self):
        self.matrix_type = 'sparse'
        self.labels_type = 'sparse'
        self.target_labels = 'inferred'


class SklearnDatasetDenseDenseExplicitTest(unittest.TestCase, _SklearnDatasetTest):

    def setUp(self):
        self.matrix_type = 'dense'
        self.labels_type = 'dense'
        self.target_labels = 'explicit'


class SklearnDatasetDenseSparseExplicitTest(unittest.TestCase, _SklearnDatasetTest):

    def setUp(self):
        self.matrix_type = 'dense'
        self.labels_type = 'sparse'
        self.target_labels = 'explicit'


class SklearnDatasetDenseDenseInferredTest(unittest.TestCase, _SklearnDatasetTest):

    def setUp(self):
        self.matrix_type = 'dense'
        self.labels_type = 'dense'
        self.target_labels = 'inferred'


class SklearnDatasetDenseSparseInferredTest(unittest.TestCase, _SklearnDatasetTest):

    def setUp(self):
        self.matrix_type = 'dense'
        self.labels_type = 'sparse'
        self.target_labels = 'inferred'


class _TextDatasetTest(_DatasetTest):

    def _dataset(self, num_samples=100, return_data=False, num_labels=_DatasetTest.NUM_LABELS):
        x = random_text_data(num_samples)
        y = random_labels(num_samples, num_labels, multi_label=self.labels_type == 'sparse')

        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError('Invalid test parameter value for target_labels:' + self.target_labels)

        if self.labels_type == 'sparse':
            target_labels = None if self.target_labels == 'inferred' else np.unique(y.indices)
        else:
            target_labels = None if self.target_labels == 'inferred' else np.unique(y)
        dataset = TextDataset(x, y, target_labels=target_labels)

        if return_data:
            return dataset, x, y
        else:
            return dataset

    def test_init_with_numpy_object(self):
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        x = np.array(x, dtype=object)
        ds = TextDataset(x, y, target_labels=ds.target_labels)

        if self.labels_type == 'sparse':
            self.assertTrue(ds.is_multi_label)
        else:
            self.assertFalse(ds.is_multi_label)

    def test_init_with_dimension_mismatch(self):
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        if self.labels_type == 'sparse':
            mask = np.ones(self.NUM_SAMPLES, dtype=bool)
            mask[0] = False
            y = y[mask]
        else:
            y = np.delete(y, 0)

        with self.assertRaisesRegex(ValueError, 'Feature and label dimensions do not match'):
            TextDataset(x, y, target_labels=ds.target_labels)

    def test_init_when_some_samples_are_unlabeled(self):
        x = random_text_data(self.NUM_SAMPLES)
        y = random_labels(self.NUM_SAMPLES, self.NUM_LABELS,
                          multi_label=self.labels_type == 'sparse')

        if self.labels_type == 'sparse':
            y_list = csr_to_list(y)
            y_list[1] = []
            y_list[2] = []
            y = list_to_csr(y_list, y.shape)
        else:
            y[0:10] = [LABEL_UNLABELED] * 10

        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError('Invalid test parameter value for target_labels:' + self.target_labels)

        target_labels = None if self.target_labels == 'inferred' else np.arange(5)

        # passes when no exeption is raised here
        TextDataset(x, y, target_labels=target_labels)

    def test_init_when_all_samples_are_unlabeled(self):
        x = random_text_data(self.NUM_SAMPLES)
        y = random_labels(self.NUM_SAMPLES,
                          self.NUM_LABELS,
                          multi_label=self.labels_type == 'sparse')

        if self.labels_type == 'sparse':
            y = csr_matrix(y.shape, dtype=y.dtype)
        else:
            y[:] = LABEL_UNLABELED

        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError('Invalid test parameter value for target_labels:' + self.target_labels)

        target_labels = None if self.target_labels == 'inferred' else np.arange(5)

        # passes when no exeption is raised here
        TextDataset(x, y, target_labels=target_labels)

    def test_get_features(self):
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)
        self.assertIsNotNone(ds.y)
        assert_array_equal(x, ds.x)

    def test_set_features(self):
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)
        ds_new = self._dataset(num_samples=self.NUM_SAMPLES)
        self.assertIsNotNone(ds.y)
        self.assertIsNotNone(ds_new.y)

        assert_array_not_equal(ds.x, ds_new.x)
        ds.x = ds_new.x

        assert_array_equal(ds.x, ds_new.x)

    def test_set_features_with_dimension_mismatch(self):
        ds, x, _ = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)
        x = x[1:]

        with self.assertRaisesRegex(ValueError, 'Feature and label dimensions do not match'):
            ds.x = x

    def test_clone(self):
        ds = self._dataset(num_samples=self.NUM_SAMPLES)
        ds_cloned = ds.clone()

        assert_array_equal(ds.x, ds_cloned.x)
        if self.labels_type == 'sparse':
            assert_array_equal(ds.y.indices, ds_cloned.y.indices)
        else:
            assert_array_equal(ds.y, ds_cloned.y)
        if self.target_labels == 'explicit':
            assert_array_equal(ds.target_labels, ds_cloned.target_labels)

        # test propagation of target_labels setting
        if self.target_labels == 'explicit':
            self.assertFalse(ds.track_target_labels)
            self.assertFalse(ds_cloned.track_target_labels)
        else:
            self.assertTrue(ds.track_target_labels)
            self.assertTrue(ds_cloned.track_target_labels)
        assert_array_equal(ds.target_labels, ds_cloned.target_labels)

        # mutability test
        x_changed = copy(ds_cloned.x)
        x_changed[0] = 'this sentence has been changed'
        ds_cloned.x = x_changed
        assert_array_not_equal(ds.x, ds_cloned.x)

        if self.labels_type == 'sparse':
            ds_cloned.y.indices = (ds_cloned.y.indices + 1) % self.NUM_LABELS
            assert_array_not_equal(ds.y.indices, ds_cloned.y.indices)
        else:
            ds_cloned.y = (ds_cloned.y + 1) % self.NUM_LABELS
            assert_array_not_equal(ds.y, ds_cloned.y)
        ds_cloned.target_labels = np.arange(10)
        assert_array_not_equal(ds.target_labels, ds_cloned.target_labels)

    def test_indexing_single_index(self):
        index = 42
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[index]
        self.assertEqual(1, len(result))
        self.assertTrue(isinstance(result, DatasetView))

        assert_array_equal([x[index]], result.x)

    def test_indexing_list_index(self):
        index = [1, 42, 56, 99]
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[index]
        self.assertEqual(4, len(result))
        self.assertTrue(isinstance(result, DatasetView))

        assert_array_equal([x[i] for i in index], result.x)

    def test_indexing_slicing(self):
        index = np.s_[10:20]
        ds, x, y = self._dataset(num_samples=self.NUM_SAMPLES, return_data=True)

        result = ds[index]
        self.assertEqual(10, len(result))
        self.assertTrue(isinstance(result, DatasetView))

        assert_array_equal([x[i] for i in np.arange(len(ds))[index]], result.x)


class TextDatasetDenseExplicitTest(unittest.TestCase, _TextDatasetTest):

    def setUp(self):
        self.labels_type = 'dense'
        self.target_labels = 'explicit'


class TextDatasetSparseExplicitTest(unittest.TestCase, _TextDatasetTest):

    def setUp(self):
        self.labels_type = 'sparse'
        self.target_labels = 'explicit'


class TextDatasetDenseInferredTest(unittest.TestCase, _TextDatasetTest):

    def setUp(self):
        self.labels_type = 'dense'
        self.target_labels = 'inferred'


class TextDatasetSparseInferredTest(unittest.TestCase, _TextDatasetTest):

    def setUp(self):
        self.labels_type = 'sparse'
        self.target_labels = 'inferred'
