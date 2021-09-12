import unittest
import numpy as np

from parameterized import parameterized_class

from small_text.classifiers import ConfidenceEnhancedLinearSVC
from small_text.query_strategies import (EmptyPoolException,
                                         lightweight_coreset,
                                         LightweightCoreset)

from tests.unit.small_text.query_strategies.test_strategies import (DEFAULT_QUERY_SIZE,
                                                                    SamplingStrategiesTests,
                                                                    query_random_data)


class LightweightCoresetBaseTest(unittest.TestCase):

    def test_lightweight_coreset(self, num_samples=20, num_features=100, n=10):
        x = np.random.rand(num_samples, num_features)
        indices = lightweight_coreset(n, x, np.mean(x, axis=0))
        self.assertEqual(n, indices.shape[0])
        self.assertEqual(n, np.unique(indices).shape[0])

    def test_lightweight_coreset_query_remaining(self, num_samples=20, num_features=100):
        x = np.random.rand(num_samples, num_features)
        lightweight_coreset(num_samples, x, np.mean(x, axis=0))

    def test_lightweight_coreset_query_with_overlarge_n(self, num_samples=20, num_features=100):
        x = np.random.rand(num_samples, num_features)
        with self.assertRaises(ValueError):
            lightweight_coreset(num_samples+1, x, np.mean(x, axis=0))


@parameterized_class([{'normalize': True}])
class LightweightCoresetTest(unittest.TestCase,SamplingStrategiesTests):

    # https://github.com/wolever/parameterized/issues/119
    @classmethod
    def setUpClass(cls):
        if cls == LightweightCoresetTest:
            raise unittest.SkipTest('parameterized_class bug')
        super().setUpClass()

    def _get_clf(self):
        return ConfidenceEnhancedLinearSVC()

    def _get_query_strategy(self):
        return LightweightCoreset(self.normalize)

    # overrides test from SamplingStrategiesTests (to use embeddings)
    def test_simple_query(self, embedding_dim=100):
        embeddings = np.random.rand(SamplingStrategiesTests.DEFAULT_NUM_SAMPLES, embedding_dim)
        indices = self._query(self._get_query_strategy(),
                              num_samples=self.DEFAULT_NUM_SAMPLES,
                              n=5,
                              embeddings=embeddings)
        self.assertEqual(5, len(indices))

    # overrides test from SamplingStrategiesTests (to use embeddings)
    def test_default_query(self, embedding_dim=100):
        embeddings = np.random.rand(SamplingStrategiesTests.DEFAULT_NUM_SAMPLES, embedding_dim)
        indices = self._query(self._get_query_strategy(), num_samples=100, embeddings=embeddings)
        self.assertEqual(DEFAULT_QUERY_SIZE, len(indices))

    # overrides test from SamplingStrategiesTests (to use embeddings)
    def test_query_takes_remaining_pool(self, embedding_dim=100):
        embeddings = np.random.rand(SamplingStrategiesTests.DEFAULT_NUM_SAMPLES, embedding_dim)
        indices = self._query(self._get_query_strategy(),
                              num_samples=self.DEFAULT_NUM_SAMPLES,
                              n=10,
                              embeddings=embeddings)
        self.assertEqual(DEFAULT_QUERY_SIZE, len(indices))

    def test_lightweight_coreset_str(self):
        strategy = self._get_query_strategy()
        expected_str = f'LightweightCoreset(normalize={str(self.normalize)})'
        self.assertEqual(expected_str, str(strategy))

    def test_lightweight_coreset_query_default(self):
        indices = query_random_data(self._get_query_strategy(), use_embeddings=True)
        self.assertEqual(10, len(indices))

    def test_lightweight_coreset_empty_pool(self, num_samples=20, n=10):
        strategy = self._get_query_strategy()

        x = np.random.rand(num_samples, 10)

        x_indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        x_indices_unlabeled = []
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with self.assertRaises(EmptyPoolException):
            strategy.query(None, x, x_indices_unlabeled, x_indices_labeled, y, n=n)
