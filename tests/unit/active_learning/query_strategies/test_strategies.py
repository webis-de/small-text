import unittest

import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal
from unittest.mock import Mock

from active_learning.classifiers import ConfidenceEnhancedLinearSVC
from active_learning.query_strategies import EmptyPoolException, PoolExhaustedException
from active_learning.query_strategies import (RandomSampling,
                                              SubsamplingQueryStrategy,
                                              BreakingTies,
                                              LeastConfidence,
                                              PredictionEntropy,
                                              lightweight_coreset,
                                              LightweightCoreset)


DEFAULT_QUERY_SIZE = 10


def query_random_data(strategy, num_samples=100, n=10, use_embeddings=False, embedding_dim=100):

    x = np.random.rand(num_samples, 10)
    kwargs = dict()

    if use_embeddings:
        kwargs['embeddings'] = np.random.rand(SamplingStrategiesTests.DEFAULT_NUM_SAMPLES,
                                              embedding_dim)

    x_indices_labeled = np.random.choice(np.arange(num_samples), size=10, replace=False)
    x_indices_unlabeled = [i for i in range(x.shape[0]) if i not in set(x_indices_labeled)]
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    return strategy.query(None,
                          x,
                          x_indices_unlabeled,
                          x_indices_labeled,
                          y,
                          n=n,
                          **kwargs)


class SamplingStrategiesTests(object):

    DEFAULT_NUM_SAMPLES = 100

    def _get_clf(self):
        return ConfidenceEnhancedLinearSVC()

    def _get_query_strategy(self):
        raise NotImplementedError()

    def test_simple_query(self):
        """Tests a query of n=5."""
        indices = self._query(self._get_query_strategy(),
                              num_samples=self.DEFAULT_NUM_SAMPLES,
                              n=5)
        self.assertEqual(5, len(indices))

    def test_default_query(self):
        """Tests the query with default args."""
        indices = self._query(self._get_query_strategy(),
                              num_samples=self.DEFAULT_NUM_SAMPLES)
        self.assertEqual(DEFAULT_QUERY_SIZE, len(indices))

    def test_query_takes_remaining_pool(self):
        indices = self._query(self._get_query_strategy(),
                              num_samples=self.DEFAULT_NUM_SAMPLES,
                              n=10)
        self.assertEqual(DEFAULT_QUERY_SIZE, len(indices))

    def test_query_exhausts_pool(self):
        """Tests for PoolExhaustedException."""
        with self.assertRaises(PoolExhaustedException):
            self._query(self._get_query_strategy(), n=11)

    def _query(self, strategy, num_samples=20, n=10, **kwargs):
        x = np.random.rand(num_samples, 10)

        x_indices_labeled = np.random.choice(np.arange(num_samples), size=10, replace=False)
        x_indices_unlabeled = [i for i in range(x.shape[0]) if i not in set(x_indices_labeled)]

        clf_mock = self._get_clf()
        if clf_mock is not None:
            proba = np.random.random_sample((num_samples,2))
            clf_mock.predict_proba = Mock(return_value=proba)

        # TODO: must be of size `num_samples`
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        return strategy.query(clf_mock, x, x_indices_unlabeled, x_indices_labeled, y, n=n, **kwargs)


class RandomSamplingTest(unittest.TestCase,SamplingStrategiesTests):

    def _get_clf(self):
        return None

    def _get_query_strategy(self):
        return RandomSampling()

    def test_random_sampling_str(self):
        strategy = RandomSampling()
        self.assertEqual('RandomSampling()', str(strategy))

    def test_random_sampling_query_default(self):
        indices = query_random_data(self._get_query_strategy())
        self.assertEqual(10, len(indices))

    def test_random_sampling_empty_pool(self, num_samples=20, n=10):
        strategy = RandomSampling()

        x = np.random.rand(num_samples, 10)

        x_indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        x_indices_unlabeled = []
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with self.assertRaises(EmptyPoolException):
            strategy.query(None, x, x_indices_unlabeled, x_indices_labeled, y, n=n)


class BreakingTiesTest(unittest.TestCase,SamplingStrategiesTests):

    def _get_clf(self):
        return ConfidenceEnhancedLinearSVC()

    def _get_query_strategy(self):
        return BreakingTies()

    def test_breaking_ties_str(self):
        strategy = self._get_query_strategy()
        self.assertEqual('BreakingTies()', str(strategy))

    def test_breaking_ties_binary(self):
        proba = np.array([
            [0.1, 0.9],
            [0.45, 0.55],
            [0.5, 0.5],
            [0.7, 0.3]
        ])
        clf_mock = self._get_clf()
        clf_mock.predict_proba = Mock(return_value=proba)

        x = np.random.rand(proba.shape[0], 10)
        strategy = self._get_query_strategy()
        indicies = strategy.query(clf_mock, x, np.arange(0, 4), np.array([]), np.array([]), n=2)

        expected = np.array([2, 1])
        assert_array_equal(expected, indicies)

        self.assertIsNotNone(strategy.scores_)
        assert_array_almost_equal(np.array([0.8, 0.1, 0, 0.4]), strategy.scores_)

    def test_breaking_ties_multiclass(self):
        proba = np.array([
            [0.1, 0.75, 0.15],
            [0.45, 0.5, 0.05],
            [0.1, 0.8, 0.1],
            [0.7, 0.15, 0.15]
        ])
        clf_mock = self._get_clf()
        clf_mock.predict_proba = Mock(return_value=proba)

        x = np.random.rand(proba.shape[0], 10)
        strategy = self._get_query_strategy()
        indicies = strategy.query(clf_mock, x, np.arange(0, 4), np.array([]), np.array([]), n=2)

        expected = np.array([1, 3])
        assert_array_equal(expected, indicies)

        self.assertIsNotNone(strategy.scores_)
        assert_array_almost_equal(np.array([0.6, 0.05, 0.7, 0.55]), strategy.scores_)


class LeastConfidenceTest(unittest.TestCase,SamplingStrategiesTests):

    def _get_clf(self):
        return ConfidenceEnhancedLinearSVC()

    def _get_query_strategy(self):
        return LeastConfidence()

    def test_least_confidence_str(self):
        strategy = self._get_query_strategy()
        self.assertEqual('LeastConfidence()', str(strategy))

    def test_least_confidence_binary(self):
        proba = np.array([
            [0.1, 0.9],
            [0.45, 0.55],
            [0.2, 0.8],
            [0.7, 0.3]
        ])
        clf_mock = self._get_clf()
        clf_mock.predict_proba = Mock(return_value=proba)

        x = np.random.rand(proba.shape[0], 10)
        strategy = self._get_query_strategy()
        indicies = strategy.query(clf_mock, x, np.arange(0, 4), np.array([]), np.array([]), n=2)

        expected = np.array([1, 3])
        assert_array_equal(expected, indicies)

        self.assertIsNotNone(strategy.scores_)
        assert_array_almost_equal(np.array([0.9, 0.55, 0.8, 0.7]), strategy.scores_)

    def test_least_confidence_multiclass(self):
        proba = np.array([
            [0.1, 0.75, 0.15],
            [0.45, 0.5, 0.05],
            [0.1, 0.8, 0.1],
            [0.7, 0.15, 0.15]
        ])
        clf_mock = self._get_clf()
        clf_mock.predict_proba = Mock(return_value=proba)

        x = np.random.rand(proba.shape[0], 10)
        strategy = self._get_query_strategy()
        indicies = strategy.query(clf_mock, x, np.arange(0, 4), np.array([]), np.array([]), n=2)

        expected = np.array([1, 3])
        assert_array_equal(expected, indicies)

        self.assertIsNotNone(strategy.scores_)
        assert_array_almost_equal(np.array([0.75, 0.5, 0.8, 0.7]), strategy.scores_)


class PredictionEntropyTest(unittest.TestCase,SamplingStrategiesTests):

    def _get_clf(self):
        return ConfidenceEnhancedLinearSVC()

    def _get_query_strategy(self):
        return PredictionEntropy()

    def test_prediction_entropy_str(self):
        strategy = self._get_query_strategy()
        self.assertEqual('PredictionEntropy()', str(strategy))

    def test_prediction_entropy_binary(self):
        proba = np.array([
            [0.1, 0.9],
            [0.45, 0.55],
            [0.5, 0.5],
            [0.7, 0.3]
        ])
        clf_mock = self._get_clf()
        clf_mock.predict_proba = Mock(return_value=proba)

        x = np.random.rand(proba.shape[0], 10)
        strategy = self._get_query_strategy()
        indicies = strategy.query(clf_mock, x, np.arange(0, 4), np.array([]), np.array([]), n=2)

        expected = np.array([2, 1])
        assert_array_equal(expected, indicies)

        self.assertIsNotNone(strategy.scores_)
        assert_array_almost_equal(np.array([0.325083, 0.688139, 0.693147, 0.610864]),
                                  strategy.scores_)

    def test_prediction_entropy_multiclass(self):
        proba = np.array([
            [0.1, 0.8, 0.1],
            [0.45, 0.25, 0.3],
            [0.33, 0.33, 0.34],
            [0.7, 0.3, 0]
        ])
        clf_mock = self._get_clf()
        clf_mock.predict_proba = Mock(return_value=proba)

        x = np.random.rand(proba.shape[0], 10)
        strategy = self._get_query_strategy()
        indicies = strategy.query(clf_mock, x, np.arange(0, 4), np.array([]), np.array([]), n=2)

        expected = np.array([2, 1])
        assert_array_equal(expected, indicies)

        assert_array_almost_equal(np.array([0.639032, 1.067094, 1.098513, 0.610864]),
                                  strategy.scores_)


class SubSamplingTest(unittest.TestCase,SamplingStrategiesTests):

    def _get_clf(self):
        return ConfidenceEnhancedLinearSVC()

    def _get_query_strategy(self):
        return SubsamplingQueryStrategy(RandomSampling(), 20)

    def test_subsampling_str(self):
        strategy = SubsamplingQueryStrategy(RandomSampling(), subsample_size=20)
        expected_str = 'SubsamplingQueryStrategy(base_query_strategy=RandomSampling(), ' \
                       'subsample_size=20)'
        self.assertEqual(expected_str, str(strategy))

    def test_subsampling_query_default(self):
        indices = query_random_data(self._get_query_strategy())
        self.assertEqual(10, len(indices))

    def test_subsampling_empty_pool(self, num_samples=20, n=10):
        strategy = self._get_query_strategy()

        x = np.random.rand(num_samples, 10)

        x_indices_labeled = np.random.choice(np.arange(100), size=10, replace=False)
        x_indices_unlabeled = []
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with self.assertRaises(EmptyPoolException):
            strategy.query(None, x, x_indices_unlabeled, x_indices_labeled, y, n=n)

    def test_scores_property(self):
        num_samples = 20
        scores = np.random.rand(num_samples, 1)

        strategy = self._get_query_strategy()

        strategy.base_query_strategy.scores_ = scores
        assert_array_equal(scores, strategy.scores_)

        strategy = self._get_query_strategy()
        self.assertIsNone(strategy.scores_)


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


class LightweightCoresetTest(unittest.TestCase,SamplingStrategiesTests):

    def _get_clf(self):
        return ConfidenceEnhancedLinearSVC()

    def _get_query_strategy(self):
        return LightweightCoreset()

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
        expected_str = 'LightweightCoreset()'
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
