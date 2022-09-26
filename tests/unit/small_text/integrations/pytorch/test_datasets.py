import unittest
import pytest

import numpy as np

from abc import abstractmethod
from scipy.sparse import csr_matrix

from numpy.testing import assert_array_equal
from parameterized import parameterized_class

from small_text.base import LABEL_UNLABELED
from small_text.data.exceptions import UnsupportedOperationException
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from tests.utils.datasets import random_text_classification_dataset
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

    from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset, \
        PytorchTextClassificationDatasetView
except (PytorchNotFoundError, ModuleNotFoundError):
    pass


@pytest.mark.pytorch
@parameterized_class([{'target_labels': 'explicit', 'multi_label': True},
                      {'target_labels': 'explicit', 'multi_label': False},
                      {'target_labels': 'inferred', 'multi_label': True},
                      {'target_labels': 'inferred', 'multi_label': False}])
class PytorchTextClassificationDatasetTest(unittest.TestCase):

    NUM_SAMPLES = 100
    NUM_LABELS = 3

    def _random_data(self, num_samples=100, num_labels=NUM_LABELS):
        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError('Invalid test parameter value for target_labels:' + self.target_labels)

        return random_text_classification_dataset(num_samples=num_samples,
                                                  multi_label=self.multi_label,
                                                  num_classes=num_labels,
                                                  target_labels=self.target_labels)

    def test_init(self):
        ds = random_text_classification_dataset(num_samples=self.NUM_SAMPLES)
        self.assertIsNotNone(ds._data)
        self.assertIsNotNone(ds.vocab)
        assert_array_equal(np.array([0, 1]), ds.target_labels)

    def test_init_with_target_labels(self):
        ds_rnd = random_text_classification_dataset(num_samples=self.NUM_SAMPLES)
        ds = PytorchTextClassificationDataset(ds_rnd.data,
                                              ds_rnd.vocab,
                                              target_labels=ds_rnd.target_labels)
        assert_array_equal(np.array([0, 1]), ds.target_labels)

    def test_init_with_target_label_warning(self):
        if self.target_labels == 'inferred':
            with self.assertWarnsRegex(UserWarning, 'Passing target_labels=None is discouraged'):
                self._random_data()

    def test_init_when_some_samples_are_unlabeled(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        if self.multi_label:
            data_list = list(ds.data[2])
            data_list[PytorchTextClassificationDataset.INDEX_LABEL] = []
            ds.data[2] = tuple(data_list)
        else:
            data_list = list(ds.data[2])
            data_list[PytorchTextClassificationDataset.INDEX_LABEL] = LABEL_UNLABELED
            ds.data[2] = tuple(data_list)

        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError('Invalid test parameter value for target_labels:' + self.target_labels)

        target_labels = None if self.target_labels == 'inferred' else ds.target_labels

        # passes when no exeption is raised here
        PytorchTextClassificationDataset(ds.data,
                                         ds.vocab,
                                         multi_label=self.multi_label,
                                         target_labels=target_labels)

    def test_init_when_all_samples_are_unlabeled(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        if self.multi_label:
            for i in range(len(ds)):
                data_list = list(ds.data[i])
                data_list[PytorchTextClassificationDataset.INDEX_LABEL] = []
                ds.data[i] = tuple(data_list)
        else:
            for i in range(len(ds)):
                data_list = list(ds.data[i])
                data_list[PytorchTextClassificationDataset.INDEX_LABEL] = LABEL_UNLABELED
                ds.data[i] = tuple(data_list)

        if self.target_labels not in ['explicit', 'inferred']:
            raise ValueError('Invalid test parameter value for target_labels:' + self.target_labels)

        target_labels = None if self.target_labels == 'inferred' else ds.target_labels

        # passes when no exeption is raised here
        PytorchTextClassificationDataset(ds.data,
                                         ds.vocab,
                                         multi_label=self.multi_label,
                                         target_labels=target_labels)

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
        if self.multi_label:
            self.assertTrue(isinstance(ds.y, csr_matrix))
            y_expected = np.zeros((self.NUM_SAMPLES, self.NUM_LABELS))
            for i, d in enumerate(ds.data):
                labels = d[ds.INDEX_LABEL]
                if labels is not None:
                    y_expected[i, labels] = 1
            y_expected = csr_matrix(y_expected)
            assert_csr_matrix_equal(y_expected, ds.y)
        else:
            self.assertTrue(isinstance(ds.y, np.ndarray))
            self.assertEqual(self.NUM_SAMPLES, ds.y.shape[0])

    def test_set_labels(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES, num_labels=self.NUM_LABELS)
        ds_new = self._random_data(num_samples=self.NUM_SAMPLES, num_labels=self.NUM_LABELS+1)

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
            assert_array_equal(np.arange(self.NUM_LABELS+1), ds.target_labels)

    def test_set_labels_with_mismatching_data_length(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        ds_new = self._random_data(num_samples=self.NUM_SAMPLES+1)

        with self.assertRaisesRegex(ValueError, 'Size mismatch: '):
            ds.y = ds_new.y

    def test_is_multi_label(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        if self.multi_label:
            self.assertTrue(ds.is_multi_label)
        else:
            self.assertFalse(ds.is_multi_label)

    def test_get_target_labels(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        expected_target_labels = np.arange(self.NUM_LABELS)
        assert_array_equal(expected_target_labels, ds.target_labels)

    def test_set_target_labels(self):
        ds = self._random_data(num_samples=self.NUM_SAMPLES)
        new_target_labels = np.arange(self.NUM_LABELS+1)
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

        self.assertTrue(np.all([
            torch.equal(t, t_cloned)
            for t, t_cloned in zip(ds.x, ds_cloned.x)
        ]))
        if self.multi_label:
            assert_csr_matrix_equal(ds.y, ds_cloned.y)
        else:
            assert_array_equal(ds.y, ds_cloned.y)
        if self.target_labels == 'explicit':
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
        self.assertTrue(isinstance(result, PytorchTextClassificationDatasetView))

        self.assertTrue(torch.equal(ds.x[index], result.x[0]))

    def test_indexing_list_index(self):
        index = [1, 42, 56, 99]
        ds = self._random_data(num_samples=self.NUM_SAMPLES)

        result = ds[index]
        self.assertEqual(len(index), len(result))
        self.assertTrue(isinstance(result, PytorchTextClassificationDatasetView))

        expected = [ds.x[i] for i in index]
        assert_list_of_tensors_equal(self, expected, result.x)

    def test_indexing_slicing(self):
        index = np.s_[10:20]
        ds = self._random_data(num_samples=self.NUM_SAMPLES)

        result = ds[index]
        self.assertEqual(10, len(result))
        self.assertTrue(isinstance(result, PytorchTextClassificationDatasetView))

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
    """Size of the dataset."""

    NUM_SAMPLES_VIEW = 14
    """Size of the default view."""

    @property
    @abstractmethod
    def dataset_view_class(self):
        pass

    @property
    @abstractmethod
    def dataset_class(self):
        pass

    @property
    @abstractmethod
    def default_result_size(self):
        pass

    def test_init_with_slice(self):
        dataset_view = self._dataset()
        view_on_view = self.dataset_view_class(dataset_view, slice(0, 10))
        self.assertEqual(10, len(view_on_view))

    def test_init_with_slice_and_step(self):
        dataset_view = self._dataset()
        view_on_view = self.dataset_view_class(dataset_view, slice(0, 10, 2))
        self.assertEqual(5, len(view_on_view))

    def test_init_with_numpy_slice(self):
        dataset = self._dataset()
        dataset_view = self.dataset_view_class(dataset, np.s_[0:10])
        self.assertEqual(10, len(dataset_view))

    def test_get_dataset(self):
        dataset = self._dataset()
        dataset_view = self.dataset_view_class(dataset, np.s_[0:10])
        self.assertEqual(dataset, dataset_view.dataset)

    def test_get_x(self):
        dataset = self._dataset()
        selection = np.random.randint(0, high=len(dataset), size=10)
        dataset_view = self.dataset_view_class(dataset, selection)
        assert_list_of_tensors_equal(self, [dataset.x[i] for i in selection], dataset_view.x)

    def test_set_x(self):
        dataset = self._dataset()
        dataset_view = self.dataset_view_class(dataset, np.s_[0:10])
        with self.assertRaises(UnsupportedOperationException):
            dataset_view.x = self._dataset()

    def test_get_y(self):
        dataset = self._dataset()
        selection = np.random.randint(0, high=len(dataset), size=10)
        dataset_view = self.dataset_view_class(dataset, selection)
        if self.multi_label:
            assert_csr_matrix_equal(dataset.y[selection], dataset_view.y)
        else:
            assert_array_equal(dataset.y[selection], dataset_view.y)

    def test_set_y(self, subset_size=10, num_labels=2):
        dataset = self._dataset()
        dataset_view = PytorchTextClassificationDatasetView(dataset, np.s_[0:subset_size])
        with self.assertRaises(UnsupportedOperationException):
            dataset_view.y = np.random.randint(0, high=num_labels, size=subset_size)

    def test_is_multi_label(self):
        dataset = self._dataset()
        if self.multi_label:
            self.assertTrue(dataset.is_multi_label)
        else:
            self.assertFalse(dataset.is_multi_label)

    def test_get_target_labels(self, subset_size=10):
        dataset = self._dataset()
        selection = np.random.randint(0, high=len(dataset), size=subset_size)
        dataset_view = self.dataset_view_class(dataset, selection)
        assert_array_equal(dataset.target_labels, dataset_view.target_labels)

    def test_set_target_labels(self):
        dataset = self._dataset()
        dataset_view = self.dataset_view_class(dataset, np.s_[0:10])
        with self.assertRaises(UnsupportedOperationException):
            dataset_view.target_labels = np.array([0])

    def test_clone_single_index(self):
        ds = self._dataset()

        if self.multi_label:
            # get first row which has at least one label
            indptr_deltas = np.array([ds.y.indptr[i + 1] - ds.y.indptr[i]
                                      for i in range(ds.y.indptr.shape[0] - 1)])
            index = np.where(indptr_deltas > 0)[0][0]
            index = int(index)
        else:
            index = 3

        ds_view = self.dataset_view_class(ds, index)

        self._clone_test(ds_view)

    def test_clone_list_index(self):
        ds = self._dataset()

        index = [1, 3, 4, 9, 10, 12]
        ds_view = self.dataset_view_class(ds, index)

        self._clone_test(ds_view)

    def test_clone_slicing(self):
        ds = self._dataset()

        index = np.s_[2:5]
        ds_view = self.dataset_view_class(ds, index)

        self._clone_test(ds_view)

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
                assert_csr_matrix_not_equal(ds_view.y, ds_cloned.y)
        else:
            ds_cloned = increase_dense_labels_safe(ds_cloned)
            assert_array_not_equal(ds_view.y, ds_cloned.y)

        ds_cloned.target_labels = np.arange(10)
        assert_array_not_equal(ds_view.target_labels, ds_cloned.target_labels)

    def test_indexing_single_index(self):
        index = 12
        ds = self._dataset()

        result = ds[index]
        self.assertEqual(1, len(result))
        self.assertTrue(isinstance(result, self.dataset_view_class))

        expected_x = [ds.x[index]]
        self.assertTrue(np.all(expected_x == result.x))

    def test_indexing_list_index(self):
        index = [1, 2, 7, 13]
        ds = self._dataset()

        result = ds[index]
        self.assertEqual(4, len(result))
        self.assertTrue(isinstance(result, self.dataset_view_class))

        expected_x = [ds.x[i] for i in index]
        self.assertTrue(np.all(expected_x == result.x))

    def test_indexing_slicing(self):
        index = np.s_[4:11]
        ds = self._dataset()

        result = ds[index]
        self.assertEqual(7, len(result))
        self.assertTrue(isinstance(result, self.dataset_view_class))

        expected_x = [ds.x[i] for i in np.arange(len(ds))[index]]
        self.assertTrue(np.all(expected_x == result.x))

    def test_get_data(self):
        dataset = self._dataset()
        assert_array_equal(len(dataset), len(dataset.data))

    def test_iter(self):
        ds = self._dataset()
        count = 0
        for _ in ds:
            count += 1
        self.assertEqual(self.default_result_size, count)

    def test_datasen_len(self):
        ds = self._dataset()
        self.assertEqual(self.default_result_size, len(ds))


class _PytorchTextClassificationDatasetViewTest(_PytorchDatasetViewTest):

    @property
    def dataset_class(self):
        return PytorchTextClassificationDataset

    @property
    def dataset_view_class(self):
        return PytorchTextClassificationDatasetView

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


class PytorchTextClassificationDatasetViewSingleLabelExplicitTest(
    unittest.TestCase,
    _PytorchTextClassificationDatasetViewTest
):
    def setUp(self):
        self.target_labels = 'explicit'

    def _dataset(self, num_labels=3):
        assert self.NUM_SAMPLES > self.NUM_SAMPLES_VIEW
        self.multi_label = False

        dataset = random_text_classification_dataset(num_samples=self.NUM_SAMPLES,
                                                     multi_label=self.multi_label,
                                                     num_classes=num_labels,
                                                     target_labels=self.target_labels)
        return dataset


class PytorchTextClassificationDatasetViewSingleLabelInferredTest(
    unittest.TestCase,
    _PytorchTextClassificationDatasetViewTest
):
    def setUp(self):
        self.target_labels = 'inferred'

    def _dataset(self, num_labels=3):
        assert self.NUM_SAMPLES > self.NUM_SAMPLES_VIEW
        self.multi_label = False

        dataset = random_text_classification_dataset(num_samples=self.NUM_SAMPLES,
                                                     multi_label=self.multi_label,
                                                     num_classes=num_labels,
                                                     target_labels=self.target_labels)
        return dataset


class PytorchTextClassificationDatasetViewMultiLabelExplicitTest(
    unittest.TestCase,
    _PytorchTextClassificationDatasetViewTest
):

    def setUp(self):
        self.target_labels = 'explicit'

    def _dataset(self, num_labels=3):
        assert self.NUM_SAMPLES > self.NUM_SAMPLES_VIEW
        self.multi_label = True

        dataset = random_text_classification_dataset(num_samples=self.NUM_SAMPLES,
                                                     multi_label=self.multi_label,
                                                     num_classes=num_labels,
                                                     target_labels=self.target_labels)
        return dataset


class PytorchTextClassificationDatasetViewMultiLabelInferredTest(
    unittest.TestCase,
    _PytorchTextClassificationDatasetViewTest
):

    def setUp(self):
        self.target_labels = 'inferred'

    def _dataset(self, num_labels=3):
        assert self.NUM_SAMPLES > self.NUM_SAMPLES_VIEW
        self.multi_label = True

        dataset = random_text_classification_dataset(num_samples=self.NUM_SAMPLES,
                                                     multi_label=self.multi_label,
                                                     num_classes=num_labels,
                                                     target_labels=self.target_labels)
        return dataset


class _NestedPytorchTextClassificationDatasetViewTest(_PytorchDatasetViewTest):

    @property
    def dataset_class(self):
        return PytorchTextClassificationDataset

    @property
    def dataset_view_class(self):
        return PytorchTextClassificationDatasetView

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
            # TODO: we would not need a third nesting level here (caused by indexing)
            self.assertFalse(ds_view.dataset.dataset.dataset.track_target_labels)
            self.assertFalse(ds_cloned.track_target_labels)
            assert_array_equal(ds_view.target_labels, ds_cloned.target_labels)
        else:
            # TODO: we would not need a third nesting level here (caused by indexing)
            self.assertTrue(ds_view.dataset.dataset.dataset.track_target_labels)
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


class NestedPytorchTextClassificationDatasetViewSingleLabelExplicitTest(
    unittest.TestCase,
    _NestedPytorchTextClassificationDatasetViewTest
):

    NUM_SAMPLES_VIEW_OUTER = 17

    def setUp(self):
        self.target_labels = 'explicit'

    def _dataset(self, num_labels=2):
        assert self.NUM_SAMPLES > self.NUM_SAMPLES_VIEW_OUTER > self.NUM_SAMPLES_VIEW
        self.multi_label = False

        ds = random_text_classification_dataset(num_samples=self.NUM_SAMPLES,
                                                multi_label=self.multi_label,
                                                num_classes=num_labels,
                                                target_labels=self.target_labels)
        ds_view_inner = PytorchTextClassificationDatasetView(ds, np.s_[:self.NUM_SAMPLES_VIEW_OUTER])
        return PytorchTextClassificationDatasetView(ds_view_inner, np.s_[:self.NUM_SAMPLES_VIEW])

    @property
    def default_result_size(self):
        return self.NUM_SAMPLES_VIEW


class NestedPytorchTextClassificationDatasetViewMultiLabelInferredTest(
    unittest.TestCase,
    _NestedPytorchTextClassificationDatasetViewTest
):
    NUM_SAMPLES_VIEW_INNER = 17
    """Size of the inner view."""

    def setUp(self):
        self.target_labels = 'inferred'

    def _dataset(self, num_labels=2):
        assert self.NUM_SAMPLES > self.NUM_SAMPLES_VIEW_INNER > self.NUM_SAMPLES_VIEW
        self.multi_label = True

        ds = random_text_classification_dataset(num_samples=self.NUM_SAMPLES,
                                                multi_label=self.multi_label,
                                                num_classes=num_labels,
                                                target_labels=self.target_labels)
        ds_view_inner = PytorchTextClassificationDatasetView(ds, np.s_[:self.NUM_SAMPLES_VIEW_INNER])
        return PytorchTextClassificationDatasetView(ds_view_inner, np.s_[:self.NUM_SAMPLES_VIEW])

    @property
    def default_result_size(self):
        return self.NUM_SAMPLES_VIEW
