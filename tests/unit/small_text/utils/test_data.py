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

    def test_check_training_data_multi_label(self):
        ds_train = random_sklearn_dataset(8, multi_label=True)
        ds_valid = random_sklearn_dataset(2, multi_label=True)
        # should not raise an error
        check_training_data(ds_train, ds_valid)

        ds_train.y = csr_matrix(np.array([[LABEL_UNLABELED, 0], [1, 0], [0, 0], [1, 0],
                                          [0, 0], [0, 1], [1, 0], [0, 0]]))
        with self.assertRaisesRegex(ValueError, 'Training set labels must be labeled'):
            check_training_data(ds_train, ds_valid)

        ds_train = random_sklearn_dataset(8, multi_label=True)
        ds_valid.y = csr_matrix(np.array([[LABEL_UNLABELED, 0], [1, 0]]))

        with self.assertRaisesRegex(ValueError, 'Validation set labels must be labeled'):
            check_training_data(ds_train, ds_valid)
