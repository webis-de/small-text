import unittest
import numpy as np

from unittest import mock

from small_text.data.sampling import balanced_sampling, stratified_sampling
from small_text.data.splits import split_data


from tests.utils.datasets import random_sklearn_dataset


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

    @mock.patch('small_text.data.splits.balanced_sampling',
                wraps=balanced_sampling)
    def test_split_data_balanced(self, balanced_sampling_mock):
        train_set = random_sklearn_dataset(100)
        y = np.array([0] * 10 + [1] * 90)

        subset_train, subset_valid = split_data(train_set, y=y, strategy='balanced')

        self.assertEqual(90, len(subset_train))
        self.assertEqual(10, len(subset_valid))
        balanced_sampling_mock.assert_called()

    @mock.patch('small_text.data.splits.balanced_sampling',
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

    @mock.patch('small_text.data.splits.stratified_sampling',
                wraps=stratified_sampling)
    def test_split_data_stratified(self, stratified_sampling_mock):
        train_set = random_sklearn_dataset(100)
        y = np.array([0] * 10 + [1] * 90)

        subset_train, subset_valid = split_data(train_set, y=y, strategy='stratified')

        self.assertEqual(90, len(subset_train))
        self.assertEqual(10, len(subset_valid))
        stratified_sampling_mock.assert_called()

    @mock.patch('small_text.data.splits.stratified_sampling',
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
