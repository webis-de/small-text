import unittest
import numpy as np

from unittest.mock import patch

from active_learning.data import balanced_sampling, stratified_sampling
from active_learning.initialization import random_initialization, \
    random_initialization_stratified, random_initialization_balanced


class RandomInitializationTest(unittest.TestCase):

    def test_random_initialization(self):

        x = np.random.rand(100,2)
        indices = random_initialization(x)

        self.assertEqual(10, len(indices))
        self.assertEqual(10, len(np.unique(indices)))
        self.assertTrue(all([i >= 0 and i < len(x) for i in indices]))

    def test_random_initialization_num_samples_too_large(self):

        dataset = np.random.rand(100,2)
        with self.assertRaises(ValueError):
            random_initialization(dataset, n_samples=101)


class RandomInitializationStratifiedTest(unittest.TestCase):

    @patch('active_learning.initialization.strategies.stratified_sampling',
           wraps=stratified_sampling)
    def test_random_initialization_stratified(self, stratified_sampling_mock):
        n_samples = 10
        y = [0]*10 + [1]*10 + [2]*10 + [3]*70
        indices = random_initialization_stratified(y, n_samples=n_samples)
        stratified_labels = [y[i] for i in indices]

        stratified_sampling_mock.assert_called_with(y, n_samples=n_samples)
        self.assertEqual(10, len(indices))
        self.assertEqual(10, len(np.unique(indices)))
        self.assertEqual(4, len(np.unique(stratified_labels)))

    def test_random_initialization_stratified_num_samples_too_large(self):
        y = [0]*25 + [1]*25 + [2]*25 + [3]*25

        with self.assertRaises(ValueError):
            random_initialization_stratified(y, n_samples=101)


class RandomInitializationBalancedTest(unittest.TestCase):

    @patch('active_learning.initialization.strategies.balanced_sampling',
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

    def test_random_initialization_balanced_num_samples_too_large(self):
        y = [0]*25 + [1]*25 + [2]*25 + [3]*25

        with self.assertRaises(ValueError):
            random_initialization_balanced(y, n_samples=101)
