import unittest
import numpy as np

from small_text.classifiers import ConfidenceEnhancedLinearSVC
from small_text.query_strategies.bayesian import BALD, _bald

from tests.utils.datasets import random_sklearn_dataset
from tests.utils.testing import assert_array_equal


class BALDHelperTest(unittest.TestCase):

    def test_bald_with_only_zeros(self):
        p = np.zeros((10, 5, 3))
        result = _bald(p)
        assert_array_equal(np.zeros(10,), result)

    def test_bald_with_only_zeros_no_epsilon(self):
        p = np.zeros((10, 5, 3))
        # Don't set eps=0. This yields nan values in the result.
        result = _bald(p, eps=0)
        self.assertTrue(np.all(np.isnan(result)))


class BALDTest(unittest.TestCase):

    def _get_clf(self):
        return ConfidenceEnhancedLinearSVC()

    def _get_query_strategy(self):
        return BALD()

    def test_simple_query(self, num_classes=3, num_samples=20):
        clf = self._get_clf()
        dataset = random_sklearn_dataset(num_samples, vocab_size=10,
                                         num_classes=num_classes)

        strategy = self._get_query_strategy()
        indices_labeled = np.random.choice(np.arange(num_samples), size=10, replace=False)
        indices_unlabeled = np.array([i for i in range(len(dataset))
                                      if i not in set(indices_labeled)])

        with self.assertRaisesRegex(TypeError, r'predict_proba\(\) got an unexpected keyword arg'):
            strategy.query(clf,
                           dataset,
                           indices_unlabeled,
                           indices_labeled,
                           self._get_query_strategy(),
                           n=5)

    def test_str(self):
        strategy = self._get_query_strategy()
        expected_str = 'BALD(dropout_samples=10)'
        self.assertEqual(expected_str, str(strategy))
