import unittest
import numpy as np

from small_text.data import balanced_sampling, stratified_sampling
from small_text.data.sampling import _get_class_histogram


class StratifiedSamplingTest(unittest.TestCase):

    def test_stratified_sampling(self):
        y = np.array([0]*10 + [1]*10 + [2]*10 + [3]*70)
        indices = stratified_sampling(y, n_samples=10)
        stratified_labels = [y[i] for i in indices]

        counts = _get_class_histogram(stratified_labels, 4)
        self.assertEqual(10, len(indices))
        self.assertEqual(10, len(np.unique(indices)))
        self.assertEqual(4, len(np.unique(stratified_labels)))

        self.assertEqual(1, counts[0])
        self.assertEqual(1, counts[1])
        self.assertEqual(1, counts[2])
        self.assertEqual(7, counts[3])

    def test_stratified_sampling_with_rare_class(self):
        y = np.array([0]*1 + [1]*10 + [2]*10 + [3]*79)
        indices = stratified_sampling(y, n_samples=10)
        stratified_labels = [y[i] for i in indices]

        counts = _get_class_histogram(stratified_labels, 4)
        self.assertEqual(10, len(indices))
        self.assertEqual(10, len(np.unique(indices)))

        self.assertTrue(counts[0] >= 0 and counts[0] <= 1)
        self.assertTrue(counts[1] >= 1 and counts[1] <= 2)
        self.assertTrue(counts[2] >= 1 and counts[2] <= 2)
        self.assertTrue(counts[3] >= 7 and counts[3] <= 8)

    def test_stratified_sampling_n_samples(self):
        y = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)

        indices = stratified_sampling(y, n_samples=30)
        counts = np.bincount(np.array([y[i] for i in indices]))
        self.assertEqual(30, len(indices))
        self.assertEqual(4, counts.shape[0])
        self.assertEqual(4, counts[counts >= 7].shape[0])

    def test_stratified_sampling_with_gaps(self):
        y = np.array([0] * 25 + [2] * 25 + [3] * 50)

        indices = stratified_sampling(y, n_samples=30)
        counts = np.bincount(np.array([y[i] for i in indices]))
        self.assertEqual(30, len(indices))
        self.assertEqual(4, len(counts))
        self.assertEqual(3, len(counts[counts >= 7]))

    def test_stratified_sampling_num_samples_too_large(self):
        y = np.array([0]*25 + [1]*25 + [2]*25 + [3]*25)

        with self.assertRaises(ValueError):
            stratified_sampling(y, n_samples=101)


class BalancedSamplingTest(unittest.TestCase):

    def test_balanced_sampling(self):
        y = [0]*10 + [1]*10 + [2]*10 + [3]*70
        indices = balanced_sampling(y, n_samples=10)
        stratified_labels = [y[i] for i in indices]

        counts = _get_class_histogram(stratified_labels, 4)
        self.assertEqual(10, len(indices))
        self.assertEqual(10, len(np.unique(indices)))
        self.assertEqual(4, len(np.unique(stratified_labels)))

        self.assertTrue(np.all(counts >= 2))

    def test_balanced_sampling_with_rare_class(self):
        y = [0]*1 + [1]*10 + [2]*10 + [3]*79
        indices = balanced_sampling(y, n_samples=10)
        stratified_labels = [y[i] for i in indices]

        counts = _get_class_histogram(stratified_labels, 4)
        self.assertEqual(10, len(indices))
        self.assertEqual(10, len(np.unique(indices)))

        self.assertTrue(counts[0] >= 1)
        self.assertTrue(counts[1] >= 2)
        self.assertTrue(counts[2] >= 2)
        self.assertTrue(counts[3] >= 2)

    def test_balanced_sampling_n_samples(self):
        y = [0] * 25 + [1] * 25 + [2] * 25 + [3] * 25

        indices = balanced_sampling(y, n_samples=30)
        counts = np.bincount(np.array([y[i] for i in indices]))
        self.assertEqual(30, len(indices))
        self.assertEqual(4, len(counts))
        self.assertEqual(4, len(counts[counts >= 7]))

    def test_balanced_sampling_with_gaps(self):
        y = [0] * 25 + [2] * 25 + [3] * 50

        indices = balanced_sampling(y, n_samples=30)
        counts = np.bincount(np.array([y[i] for i in indices]))
        self.assertEqual(30, len(indices))
        self.assertEqual(4, len(counts))
        self.assertEqual(3, len(counts[counts >= 7]))

    def test_balanced_sampling_num_samples_too_large(self):
        y = [0]*25 + [1]*25 + [2]*25 + [3]*25

        with self.assertRaises(ValueError):
            balanced_sampling(y, n_samples=101)
