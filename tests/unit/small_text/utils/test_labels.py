import unittest
import numpy as np

from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from small_text.base import LABEL_IGNORED, LABEL_UNLABELED
from small_text.utils.labels import (
    concatenate,
    csr_to_list,
    get_flattened_unique_labels,
    get_ignored_labels_mask,
    get_num_labels,
    list_to_csr,
    remove_by_index
)

from tests.utils.testing import assert_csr_matrix_equal
from tests.utils.datasets import random_sklearn_dataset


class LabelUtilsTest(unittest.TestCase):

    def test_get_num_labels_dense(self):
        self.assertEqual(4, get_num_labels(np.array([3, 2, 1, 0])))
        self.assertEqual(4, get_num_labels(np.array([3])))
        with self.assertRaisesRegex(ValueError, 'Invalid labeling'):
            self.assertEqual(0, get_num_labels(np.array([])))

    def test_get_num_labels_sparse(self):
        mat = csr_matrix(np.array([
            [1, 1],
            [0, 1],
            [1, 0],
            [0, 0]
        ]))
        self.assertEqual(2, get_num_labels(mat))
        mat = csr_matrix(np.array([
            [1, 1]
        ]))
        self.assertEqual(2, get_num_labels(mat))
        mat = csr_matrix((0, 0), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, 'Invalid labeling'):
            self.assertEqual(0, get_num_labels(mat))

    def test_concatenate_dense(self):
        x = np.array([1, 2, 3])
        y = np.array([3, 2, 1])

        result = concatenate(x, y)
        expected = np.array([1, 2, 3, 3, 2, 1])

        assert_array_equal(expected, result)

    def test_concatenate_sparse(self):
        x = csr_matrix(np.array([[0, 1], [1, 0], [1, 1]]))
        y = csr_matrix(np.array([[1, 1], [1, 0], [0, 1]]))

        result = concatenate(x, y)
        expected = csr_matrix(
            np.array([
                [0, 1], [1, 0], [1, 1], [1, 1], [1, 0], [0, 1]
            ])
        )

        assert_csr_matrix_equal(expected, result)

    def test_get_ignored_labels_mask_dense(self):

        y = np.array([1, LABEL_IGNORED, 3, 2])
        mask = get_ignored_labels_mask(y, LABEL_IGNORED)

        assert_array_equal(np.array([False, True, False, False]), mask)

    def test_get_ignored_labels_mask_sparse(self):

        y = csr_matrix(np.array([[1, 1], [LABEL_IGNORED, 0], [LABEL_IGNORED, LABEL_IGNORED], [1, 0]]))
        mask = get_ignored_labels_mask(y, LABEL_IGNORED)

        assert_array_equal(np.array([False, True, True, False]), mask)

    def test_remove_by_index_dense(self):
        y = np.array([3, 2, 1, 2, 1])
        y_new = remove_by_index(y, 3)
        expected = np.array([3, 2, 1, 1])

        assert_array_equal(expected, y_new)

    def test_remove_by_index_list_dense(self):
        y = np.array([3, 2, 1, 2, 1])
        y_new = remove_by_index(y, [3, 4])
        expected = np.array([3, 2, 1])

        assert_array_equal(expected, y_new)

    def test_remove_by_index_sparse(self):
        y = csr_matrix(np.array([[1, 1], [1, 0], [0, 1], [1, 1]]))

        y_new = remove_by_index(y, 2)
        expected = csr_matrix(
            np.array([
                [1, 1], [1, 0], [1, 1]
            ])
        )

        assert_csr_matrix_equal(expected, y_new)

    def test_remove_by_index_list_sparse(self):
        y = csr_matrix(np.array([[1, 1], [1, 0], [0, 1], [1, 1]]))

        y_new = remove_by_index(y, [2, 3])
        expected = csr_matrix(
            np.array([
                [1, 1], [1, 0]
            ])
        )
        assert_csr_matrix_equal(expected, y_new)

    def test_csr_to_list(self):
        mat = csr_matrix(np.array([
            [1, 1],
            [0, 1],
            [1, 0],
            [0, 0]
        ]))
        label_list = csr_to_list(mat)
        self.assertEqual([[0, 1], [1], [0], []], label_list)

    def test_list_to_csr(self):
        label_list = [[], [0, 1], [1, 2, 3], [1], [], [0]]
        result = list_to_csr(label_list, (6, 4))

        self.assertTrue(isinstance(result, csr_matrix))
        self.assertEqual(np.int64, result.dtype)
        self.assertEqual(np.int64, result.data.dtype)
        self.assertEqual(np.int32, result.indices.dtype)
        self.assertEqual(np.int32, result.indices.dtype)

    def test_list_to_csr_all_empty(self):
        label_list = [[], [], [], [], [], []]
        result = list_to_csr(label_list, (6, 4), dtype=np.float64)

        self.assertTrue(isinstance(result, csr_matrix))
        self.assertEqual(np.float64, result.dtype)
        self.assertEqual(np.float64, result.data.dtype)
        self.assertEqual(np.int32, result.indices.dtype)
        self.assertEqual(np.int32, result.indices.dtype)

    def test_list_to_csr_float(self):
        label_list = [[], [0, 1], [1, 2, 3], [1], [], [0]]
        result = list_to_csr(label_list, (6, 4), dtype=np.float64)

        self.assertTrue(isinstance(result, csr_matrix))
        self.assertEqual(np.float64, result.dtype)
        self.assertEqual(np.float64, result.data.dtype)
        self.assertEqual(np.int32, result.indices.dtype)
        self.assertEqual(np.int32, result.indices.dtype)

    def test_get_flattened_unique_labels(self):
        dataset = random_sklearn_dataset(10)
        labels = get_flattened_unique_labels(dataset)
        assert_array_equal(np.array([0, 1]), labels)

    def test_get_flattened_unique_labels_no_labels(self):
        dataset = random_sklearn_dataset(10)
        dataset.y = np.array([LABEL_UNLABELED] * len(dataset))
        labels = get_flattened_unique_labels(dataset)
        self.assertEqual((0,), labels.shape)

    def test_get_flattened_unique_labels_multi_label(self):
        num_classes = 3
        dataset = random_sklearn_dataset(10, multi_label=True, num_classes=num_classes)
        labels = get_flattened_unique_labels(dataset)
        assert_array_equal(np.array([0, 1, 2]), labels)

    def test_get_flattened_unique_labels_multi_label_no_labels(self):
        num_classes = 3
        dataset = random_sklearn_dataset(10, multi_label=True, num_classes=num_classes)
        dataset.y = csr_matrix((10, num_classes))
        labels = get_flattened_unique_labels(dataset)
        self.assertEqual((0,), labels.shape)
