import unittest
import pytest

import numpy as np

from scipy.sparse import csr_matrix

from numpy.testing import assert_array_equal
from parameterized import parameterized_class

from small_text.base import LABEL_UNLABELED
from small_text.integrations.pytorch.datasets import PytorchDatasetView
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from tests.utils.misc import increase_dense_labels_safe
from tests.utils.testing import (
    assert_array_not_equal,
    assert_csr_matrix_equal,
    assert_csr_matrix_not_equal,
    assert_list_of_tensors_equal,
    assert_list_of_tensors_not_equal
)

try:
    import torch
    from small_text.integrations.transformers.datasets import (
        TransformersDataset,
        TransformersDatasetView
    )
    from tests.unit.small_text.integrations.pytorch.test_datasets import (
        _PytorchDatasetViewTest
    )
    from tests.utils.datasets import random_transformer_dataset
except (ModuleNotFoundError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
@parameterized_class([{'target_labels': 'explicit', 'multi_label': True},
                      {'target_labels': 'explicit', 'multi_label': False},
                      {'target_labels': 'inferred', 'multi_label': True},
                      {'target_labels': 'inferred', 'multi_label': False}])
class TransformersDatasetTest(unittest.TestCase):
    NUM_SAMPLES = 100
    NUM_LABELS = 3

    def _random_data(self, num_samples=NUM_SAMPLES, num_classes=NUM_LABELS):
        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError('Invalid test parameter value for target_labels:' + self.target_labels)

        return random_transformer_dataset(num_samples=num_samples,
                                          multi_label=self.multi_label,
                                          num_classes=num_classes,
                                          target_labels=self.target_labels)

    def test_init(self):
        ds = self._random_data()
        self.assertIsNotNone(ds._data)

    def test_init_with_target_label_warning(self):
        if self.target_labels == 'inferred':
            with self.assertWarnsRegex(UserWarning, 'Passing target_labels=None is discouraged'):
                self._random_data()

    def test_init_when_some_samples_are_unlabeled(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        if self.multi_label:
            data_list = list(ds.data[2])
            data_list[TransformersDataset.INDEX_LABEL] = []
            ds.data[2] = tuple(data_list)
        else:
            data_list = list(ds.data[2])
            data_list[TransformersDataset.INDEX_LABEL] = LABEL_UNLABELED
            ds.data[2] = tuple(data_list)

        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError('Invalid test parameter value for target_labels:' + self.target_labels)

        target_labels = None if self.target_labels == 'inferred' else ds.target_labels

        # passes when no exeption is raised here
        TransformersDataset(ds.data,
                            multi_label=self.multi_label,
                            target_labels=target_labels)

    def test_init_when_all_samples_are_unlabeled(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        if self.multi_label:
            for i in range(len(ds)):
                data_list = list(ds.data[i])
                data_list[TransformersDataset.INDEX_LABEL] = []
                ds.data[i] = tuple(data_list)
        else:
            for i in range(len(ds)):
                data_list = list(ds.data[i])
                data_list[TransformersDataset.INDEX_LABEL] = LABEL_UNLABELED
                ds.data[i] = tuple(data_list)

        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError('Invalid test parameter value for target_labels:' + self.target_labels)

        target_labels = None if self.target_labels == 'inferred' else ds.target_labels

        # passes when no exeption is raised here
        TransformersDataset(ds.data,
                            multi_label=self.multi_label,
                            target_labels=target_labels)

    def test_get_features(self):
        ds = self._random_data()
        self.assertEqual(self.NUM_SAMPLES, len(ds.x))

    def test_set_features(self):
        ds = self._random_data()
        self.assertEqual(self.NUM_SAMPLES, len(ds.x))
        ds_new = self._random_data()
        ds.x = ds_new.x
        assert_list_of_tensors_equal(self, ds.x, ds_new.x)

    def test_get_labels(self):
        ds = self._random_data()
        if self.multi_label:
            self.assertTrue(isinstance(ds.y, csr_matrix))
            y_expected = np.zeros((self.NUM_SAMPLES, self.NUM_LABELS))
            for i, d in enumerate(ds.data):
                labels = d[ds.INDEX_LABEL]
                if labels is not None and labels.shape[0] > 0:
                    y_expected[i, labels] = 1
            y_expected = csr_matrix(y_expected)
            assert_csr_matrix_equal(y_expected, ds.y)
        else:
            self.assertTrue(isinstance(ds.y, np.ndarray))
            self.assertEqual(self.NUM_SAMPLES, ds.y.shape[0])

    def test_set_labels(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES, num_classes=self.NUM_LABELS)
        ds_new = self._random_data(num_samples=self.NUM_SAMPLES, num_classes=self.NUM_LABELS+1)

        if self.multi_label:
            assert_csr_matrix_not_equal(ds.y, ds_new.y)
        else:
            assert_array_not_equal(ds.y, ds_new.y)

        if self.target_labels == 'explicit':
            with self.assertRaisesRegex(ValueError, 'Error while assigning new labels'):
                ds.y = ds_new.y
        else:
            ds.y = ds_new.y

            if self.multi_label:
                assert_csr_matrix_equal(ds.y, ds_new.y)
            else:
                assert_array_equal(ds.y, ds_new.y)
                self.assertTrue(isinstance(ds.y, np.ndarray))
                self.assertEqual(self.NUM_SAMPLES, ds.y.shape[0])

            if self.target_labels == 'explicit':
                assert_array_equal(np.arange(self.NUM_LABELS), ds.target_labels)
            else:
                assert_array_equal(np.arange(self.NUM_LABELS+1), ds.target_labels)

    def test_set_labels_with_mismatching_data_length(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        ds_new = self._random_data(num_samples=self.NUM_SAMPLES+1)

        with self.assertRaisesRegex(ValueError, 'Size mismatch: '):
            ds.y = ds_new.y

    def test_is_multi_label(self):
        dataset = self._random_data(num_samples=self.NUM_SAMPLES)
        if self.multi_label:
            self.assertTrue(dataset.is_multi_label)
        else:
            self.assertFalse(dataset.is_multi_label)

    def test_get_target_labels(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        expected_target_labels = np.arange(self.NUM_LABELS)
        assert_array_equal(expected_target_labels, ds.target_labels)

    def test_set_target_labels(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        new_target_labels = np.arange(self.NUM_LABELS + 1)
        ds.target_labels = new_target_labels
        assert_array_equal(new_target_labels, ds.target_labels)

    def test_set_target_labels_where_existing_labels_are_outside_of_given_set(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        new_target_labels = np.array([0])
        with self.assertRaisesRegex(ValueError, 'Cannot remove existing labels'):
            ds.target_labels = new_target_labels

    def test_get_data(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        assert_array_equal(len(ds), len(ds.data))

    def test_clone(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        ds_cloned = ds.clone()

        assert_list_of_tensors_equal(self, ds.x, ds_cloned.x)
        if self.multi_label:
            assert_csr_matrix_equal(ds.y, ds_cloned.y)
        else:
            assert_array_equal(ds.y, ds_cloned.y)

        assert_array_equal(ds.target_labels, ds_cloned.target_labels)

        # test propagation of target_labels setting
        if self.target_labels == 'explicit':
            self.assertFalse(ds.track_target_labels)
            self.assertFalse(ds_cloned.track_target_labels)
            assert_array_equal(ds.target_labels, ds_cloned.target_labels)
        else:
            self.assertTrue(ds.track_target_labels)
            self.assertTrue(ds_cloned.track_target_labels)

        # mutability test
        ds_cloned.x[0][0] += 1
        assert_list_of_tensors_not_equal(self, ds.x, ds_cloned.x)

        if self.multi_label:
            y_tmp = ds_cloned.y.todense()
            y_tmp[2:10] = 0
            ds_cloned.y = csr_matrix(y_tmp)
            assert_array_not_equal(ds.y.indices, ds_cloned.y.indices)
        else:
            ds_cloned = increase_dense_labels_safe(ds_cloned)
            assert_array_not_equal(ds.y, ds_cloned.y)

        ds_cloned.target_labels = np.arange(10)
        assert_array_not_equal(ds.target_labels, ds_cloned.target_labels)

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
        self.assertEqual(4, len(result))
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


class _TransformersDatasetViewTest(_PytorchDatasetViewTest):

    @property
    def dataset_class(self):
        return TransformersDataset

    @property
    def dataset_view_class(self):
        return TransformersDatasetView

    @property
    def default_result_size(self):
        return self.NUM_SAMPLES

    def _clone_test(self, ds_view):
        ds_cloned = ds_view.clone()
        self.assertTrue(isinstance(ds_cloned, self.dataset_class))

        self.assertTrue(np.all([
            torch.equal(t, t_cloned)
            for t, t_cloned in zip(ds_view.x, ds_cloned.x)
        ]))
        if self.multi_label:
            if self.target_labels == 'explicit':
                assert_csr_matrix_equal(ds_view.y, ds_cloned.y)
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
        ds_cloned.x[0][0] += 1
        assert_list_of_tensors_not_equal(self, ds_view.x, ds_cloned.x)

        if self.multi_label:
            y_tmp = ds_cloned.y.todense()
            y_tmp = (y_tmp + 1) % 2
            ds_cloned.y = csr_matrix(y_tmp, shape=ds_view.y.shape)
            try:
                assert_csr_matrix_not_equal(ds_view.y, ds_cloned.y)
            except (AssertionError, ValueError):
                print()
                assert_csr_matrix_not_equal(ds_view.y, ds_cloned.y)
        else:
            ds_cloned = increase_dense_labels_safe(ds_cloned)
            assert_array_not_equal(ds_view.y, ds_cloned.y)

        ds_cloned.target_labels = np.arange(10)
        assert_array_not_equal(ds_view.target_labels, ds_cloned.target_labels)


class TransformersDatasetViewSingleLabelExplicitTest(unittest.TestCase, _TransformersDatasetViewTest):

    def setUp(self):
        self.target_labels = 'explicit'

    def _dataset(self, num_labels=3):
        assert self.NUM_SAMPLES > self.NUM_SAMPLES_VIEW
        self.multi_label = False

        dataset = random_transformer_dataset(num_samples=self.NUM_SAMPLES,
                                             multi_label=self.multi_label,
                                             num_classes=num_labels,
                                             target_labels=self.target_labels)

        return dataset


class TransformersDatasetViewSingleLabelInferredTest(unittest.TestCase,
                                                     _TransformersDatasetViewTest):

    def setUp(self):
        self.target_labels = 'inferred'

    def _dataset(self, num_labels=3):
        assert self.NUM_SAMPLES > self.NUM_SAMPLES_VIEW
        self.multi_label = False

        dataset = random_transformer_dataset(num_samples=self.NUM_SAMPLES,
                                             multi_label=self.multi_label,
                                             num_classes=num_labels,
                                             target_labels=self.target_labels)

        return dataset


class TransformersDatasetViewMultiLabelExplicitTest(unittest.TestCase, _TransformersDatasetViewTest):

    def setUp(self):
        self.target_labels = 'explicit'

    def _dataset(self, num_labels=3):
        assert self.NUM_SAMPLES > self.NUM_SAMPLES_VIEW
        self.multi_label = True

        dataset = random_transformer_dataset(num_samples=self.NUM_SAMPLES,
                                             multi_label=self.multi_label,
                                             num_classes=num_labels,
                                             target_labels=self.target_labels)

        return dataset


class TransformersDatasetViewMultiLabelInferredTest(unittest.TestCase, _TransformersDatasetViewTest):

    def setUp(self):
        self.target_labels = 'inferred'

    def _dataset(self, num_labels=3):
        assert self.NUM_SAMPLES > self.NUM_SAMPLES_VIEW
        self.multi_label = True

        dataset = random_transformer_dataset(num_samples=self.NUM_SAMPLES,
                                             multi_label=self.multi_label,
                                             num_classes=num_labels,
                                             target_labels=self.target_labels)

        return dataset
