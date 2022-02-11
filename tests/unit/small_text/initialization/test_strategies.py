import unittest
import numpy as np

from unittest.mock import patch
from scipy.sparse import csr_matrix

from small_text.data import balanced_sampling, stratified_sampling
from small_text.initialization import random_initialization, \
    random_initialization_stratified, random_initialization_balanced

from tests.utils.datasets import random_sklearn_dataset


class RandomInitializationTest(unittest.TestCase):

    def test_random_initialization(self):

        x = random_sklearn_dataset(100, vocab_size=2)
        indices = random_initialization(x)

        self.assertEqual(10, len(indices))
        self.assertEqual(10, len(np.unique(indices)))
        self.assertTrue(all([i >= 0 and i < len(x) for i in indices]))

    def test_random_initialization_num_samples_too_large(self):

        dataset = random_sklearn_dataset(100, vocab_size=2)
        with self.assertRaises(ValueError):
            random_initialization(dataset, n_samples=101)


class RandomInitializationStratifiedTest(unittest.TestCase):

    @patch('small_text.initialization.strategies.stratified_sampling',
           wraps=stratified_sampling)
    def test_random_initialization_stratified(self, stratified_sampling_mock):
        n_samples = 10
        y = np.array([0]*10 + [1]*10 + [2]*10 + [3]*70)
        indices = random_initialization_stratified(y, n_samples=n_samples)
        stratified_labels = [y[i] for i in indices]

        stratified_sampling_mock.assert_called_with(y, n_samples=n_samples)
        self.assertEqual(10, len(indices))
        self.assertEqual(10, len(np.unique(indices)))
        self.assertEqual(4, len(np.unique(stratified_labels)))

    def test_random_initialization_stratified_num_samples_too_large(self):
        y = np.array([0]*25 + [1]*25 + [2]*25 + [3]*25)

        with self.assertRaises(ValueError):
            random_initialization_stratified(y, n_samples=101)

    def test_random_initialization_stratified_multilabel(self):
        y = np.array([[0, 0, 0, 0]]*10
                     + [[0, 0, 0, 1]]*10
                     + [[0, 0, 1, 0]]*10
                     + [[0, 1, 0, 0]]*10
                     + [[0, 1, 0, 1]]*10
                     + [[1, 0, 0, 0]]*50)
        y = csr_matrix(y)

        indices = random_initialization_stratified(y)
        self.assertEqual(10, indices.shape[0])
        self.assertEqual(10, len(np.unique(indices)))

    def test_random_initialization_stratified_multilabel_illegal_strategy(self):
        y = np.array([[0, 0, 0, 0]]*10
                     + [[0, 0, 0, 1]]*10
                     + [[0, 0, 1, 0]]*10
                     + [[0, 1, 0, 0]]*10
                     + [[0, 1, 0, 1]]*10
                     + [[1, 0, 0, 0]]*50)
        y = csr_matrix(y)

        with self.assertRaises(ValueError):
            random_initialization_stratified(y, multilabel_strategy='does-not-exist')


class RandomInitializationBalancedTest(unittest.TestCase):

    @patch('small_text.initialization.strategies.balanced_sampling',
           wraps=balanced_sampling)
    def test_random_initialization_balanced(self, balanced_sampling_mock):
        n_samples = 10
        y = [0]*10 + [1]*10 + [2]*10 + [3]*70
        indices = random_initialization_balanced(y, n_samples=n_samples)
        stratified_labels = [y[i] for i in indices]

        balanced_sampling_mock.assert_called_with(y, n_samples=n_samples)
        self.assertEqual(10, len(indices))
        self.assertEqual(10, len(np.unique(indices)))
        self.assertEqual(4, len(np.unique(stratified_labels)))

    def test_random_initialization_balanced_multilabel(self):
        y = np.array([[0, 0, 0, 0]]*10
                     + [[0, 0, 0, 1]]*10
                     + [[0, 0, 1, 0]]*10
                     + [[0, 1, 0, 0]]*10
                     + [[0, 1, 0, 1]]*10
                     + [[1, 0, 0, 0]]*50)
        y = csr_matrix(y)

        with self.assertRaises(NotImplementedError):
            random_initialization_balanced(y)

    def test_random_initialization_balanced_num_samples_too_large(self):
        y = [0]*25 + [1]*25 + [2]*25 + [3]*25

        with self.assertRaises(ValueError):
            random_initialization_balanced(y, n_samples=101)
