import unittest
import numpy as np

from scipy.sparse import csr_matrix

from small_text.base import LABEL_UNLABELED
from small_text.utils.data import check_training_data, list_length

from tests.utils.datasets import random_sklearn_dataset


class DataUtilsTest(unittest.TestCase):

    def test_list_length(self):
        self.assertEqual(10, list_length(list(range(10))))
        self.assertEqual(10, list_length(np.random.rand(10, 2)))

    def test_check_training_data(self):
        ds_train = random_sklearn_dataset(8)
        ds_valid = random_sklearn_dataset(2)
        # should not raise an error
        check_training_data(ds_train, ds_valid)

        ds_train.y = np.array([LABEL_UNLABELED, 0, 1, 0, 1, 0, 1, 0])
        with self.assertRaisesRegex(ValueError, 'Training set labels must be labeled'):
            check_training_data(ds_train, ds_valid)

        ds_train = random_sklearn_dataset(8)
        ds_valid.y = np.array([LABEL_UNLABELED, 0])

        with self.assertRaisesRegex(ValueError, 'Validation set labels must be labeled'):
            check_training_data(ds_train, ds_valid)

    def test_check_training_data_no_validation_set(self):
        ds_train = random_sklearn_dataset(8)
        check_training_data(ds_train, None)

        ds_train.y = np.array([LABEL_UNLABELED, 0, 1, 0, 1, 0, 1, 0])
        with self.assertRaisesRegex(ValueError, 'Training set labels must be labeled'):
            check_training_data(ds_train, None)

    def test_check_training_data_sample_weights(self):
        ds_train = random_sklearn_dataset(8)
        weights = np.random.randn(8)
        weights = weights - weights.min() + 1e-8
        check_training_data(ds_train, None, weights=weights)

    def test_check_training_data_sample_weights_invalid_size(self):
        ds_train = random_sklearn_dataset(8)

        weights = np.random.randn(7)
        weights = weights - weights.min() + 1e-8

        with self.assertRaisesRegex(ValueError,
                                    'Training data and weights must have the same size'):
            check_training_data(ds_train, None, weights=weights)

    def test_check_training_data_sample_weights_lesser_or_equal_zero(self):
        ds_train = random_sklearn_dataset(8)

        weights = np.random.randn(8)
        weights[0] = -1

        with self.assertRaisesRegex(ValueError,
                                    'Weights must be greater zero'):
            check_training_data(ds_train, None, weights=weights)

    def test_check_training_data_multi_label(self):
        ds_train = random_sklearn_dataset(8, multi_label=True)
        ds_train.y = csr_matrix(np.array([[0, 0], [1, 0], [1, 0], [1, 0],
                                          [0, 1], [1, 1], [1, 1], [0, 1]]))

        ds_valid = random_sklearn_dataset(2, multi_label=True)
        # should not raise an error
        check_training_data(ds_train, ds_valid)

    def test_check_training_data_multi_label_sample_weights(self):
        ds_train = random_sklearn_dataset(8, multi_label=True)
        ds_train.y = csr_matrix(np.array([[0, 0], [1, 0], [1, 0], [1, 0],
                                          [0, 1], [1, 1], [1, 1], [0, 1]]))
        weights = np.random.randn(8)
        weights = weights - weights.min() + 1e-8

        ds_valid = random_sklearn_dataset(2, multi_label=True)
        # should not raise an error
        check_training_data(ds_train, ds_valid, weights=weights)

    def test_check_training_data_multi_label_sample_weights_invalid_size(self):
        ds_train = random_sklearn_dataset(8, multi_label=True)

        weights = np.random.randn(7)
        weights = weights - weights.min() + 1e-8

        with self.assertRaisesRegex(ValueError,
                                    'Training data and weights must have the same size'):
            check_training_data(ds_train, None, weights=weights)

    def test_check_training_data_multi_label_sample_weights_lesser_or_equal_zero(self):
        ds_train = random_sklearn_dataset(8, multi_label=True)

        weights = np.random.randn(8)
        weights[0] = -1

        with self.assertRaisesRegex(ValueError,
                                    'Weights must be greater zero'):
            check_training_data(ds_train, None, weights=weights)
