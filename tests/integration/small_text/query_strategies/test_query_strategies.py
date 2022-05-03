import unittest
import pytest

import numpy as np

from unittest.mock import Mock, patch

from numpy.testing import assert_array_equal
from sklearn.preprocessing import normalize

from small_text.classifiers import ConfidenceEnhancedLinearSVC
from small_text.query_strategies import LeastConfidence, SEALS

from tests.unit.small_text.query_strategies.test_strategies import (
    SamplingStrategiesTests,
    SklearnClassifierWithRandomEmbeddings
)

from tests.utils.datasets import random_sklearn_dataset


@pytest.mark.optional
class SEALSTest(unittest.TestCase, SamplingStrategiesTests):

    def _get_clf(self):
        return SklearnClassifierWithRandomEmbeddings(ConfidenceEnhancedLinearSVC(), 2)

    def _get_query_strategy(self):
        return SEALS(LeastConfidence(), k=5)

    @patch('small_text.query_strategies.strategies.normalize', wraps=normalize)
    def test_normalize(self, normalize_spy, num_samples=100, n=10):
        dataset = random_sklearn_dataset(num_samples, vocab_size=10)
        strategy = SEALS(LeastConfidence(), k=5)

        indices_labeled = np.random.choice(np.arange(num_samples), size=10, replace=False)
        indices_unlabeled = np.array([i for i in range(len(dataset))
                                      if i not in set(indices_labeled)])

        fake_embeddings = np.random.rand(num_samples, 5)

        clf_mock = self._get_clf()
        if clf_mock is not None:
            proba = np.random.random_sample((num_samples, 2))
            clf_mock.predict_proba = Mock(return_value=proba)
            clf_mock.embed = Mock(return_value=fake_embeddings)

        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        strategy.query(clf_mock, dataset, indices_unlabeled, indices_labeled, y, n=n)

        normalize_spy.assert_called()
        assert_array_equal(fake_embeddings, normalize_spy.call_args[0][0])

    @patch('small_text.query_strategies.strategies.normalize', wraps=normalize)
    def test_normalize_false(self, normalize_spy, num_samples=100, n=10):
        dataset = random_sklearn_dataset(num_samples, vocab_size=10)
        strategy = SEALS(LeastConfidence(), k=5, normalize=False)

        indices_labeled = np.random.choice(np.arange(num_samples), size=10, replace=False)
        indices_unlabeled = np.array([i for i in range(len(dataset))
                                      if i not in set(indices_labeled)])

        clf_mock = self._get_clf()
        if clf_mock is not None:
            proba = np.random.random_sample((num_samples, 2))
            clf_mock.predict_proba = Mock(return_value=proba)

        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        strategy.query(clf_mock, dataset, indices_unlabeled, indices_labeled, y, n=n)

        normalize_spy.assert_not_called()

    def test_query(self, num_samples=100, n=10):
        dataset = random_sklearn_dataset(num_samples, vocab_size=10)
        base_query_strategy = LeastConfidence()
        strategy = SEALS(base_query_strategy, k=5)

        with patch.object(strategy,
                          'get_subset_indices',
                          wraps=strategy.get_subset_indices) as get_subset_indices_spy, \
            patch.object(base_query_strategy,
                         'query',
                         wraps=base_query_strategy.query) as base_strategy_query_spy:

            indices_labeled = np.random.choice(np.arange(num_samples), size=10, replace=False)
            indices_unlabeled = np.array([i for i in range(len(dataset))
                                          if i not in set(indices_labeled)])

            clf_mock = self._get_clf()
            if clf_mock is not None:
                proba = np.random.random_sample((num_samples, 2))
                clf_mock.predict_proba = Mock(return_value=proba)

            y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

            # no error should be thrown
            strategy.query(clf_mock, dataset, indices_unlabeled, indices_labeled, y, n=n)

        get_subset_indices_spy.assert_called_once()
        self.assertEqual(clf_mock, get_subset_indices_spy.call_args[0][0])
        self.assertEqual(dataset, get_subset_indices_spy.call_args[0][1])
        assert_array_equal(indices_unlabeled, get_subset_indices_spy.call_args[0][2])
        assert_array_equal(indices_labeled, get_subset_indices_spy.call_args[0][3])

        base_strategy_query_spy.assert_called_once()
        self.assertTrue(len(base_strategy_query_spy.call_args[0][2]) <= 50)

    def test_query_when_unlabeled_pool_is_smaller_than_k(self, num_samples=100, n=10):
        dataset = random_sklearn_dataset(num_samples, vocab_size=10)
        base_query_strategy = LeastConfidence()
        strategy = SEALS(base_query_strategy, k=100)

        with patch.object(strategy,
                          'get_subset_indices',
                          wraps=strategy.get_subset_indices) as get_subset_indices_spy, \
            patch.object(base_query_strategy,
                         'query',
                         wraps=base_query_strategy.query) as base_strategy_query_spy:

            indices_labeled = np.random.choice(np.arange(num_samples), size=10, replace=False)
            indices_unlabeled = np.array([i for i in range(len(dataset))
                                          if i not in set(indices_labeled)])

            clf_mock = self._get_clf()
            if clf_mock is not None:
                proba = np.random.random_sample((num_samples, 2))
                clf_mock.predict_proba = Mock(return_value=proba)

            y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

            # no error should be thrown
            strategy.query(clf_mock, dataset, indices_unlabeled, indices_labeled, y, n=n)

        get_subset_indices_spy.assert_not_called()
        base_strategy_query_spy.assert_called_once_with(clf_mock, dataset, indices_unlabeled, indices_labeled, y, n=n)

    def test_initialize_index_default_hnsw_kwargs(self, num_samples=100):
        embeddings = np.random.rand(num_samples, 5)
        indices_unlabeled = np.random.choice(np.arange(num_samples), 10, replace=False)
        hnsw_kwargs = dict()

        index = SEALS.initialize_index(embeddings, indices_unlabeled, hnsw_kwargs)

        self.assertIsNotNone(index)
        self.assertEqual('l2', index.space)
        self.assertEqual(200, index.ef_construction)
        self.assertEqual(64, index.M)
        self.assertEqual(200, index.ef)

    def test_initialize_index_custom_hnsw_kwargs(self, num_samples=100):
        embeddings = np.random.rand(num_samples, 5)
        indices_unlabeled = np.random.choice(np.arange(num_samples), 10, replace=False)
        hnsw_kwargs = dict({
            'space': 'cosine',
            'ef_construction': 100,
            'M': 32,
            'ef': 80,
        })

        index = SEALS.initialize_index(embeddings, indices_unlabeled, hnsw_kwargs)

        self.assertIsNotNone(index)
        self.assertEqual('cosine', index.space)
        self.assertEqual(100, index.ef_construction)
        self.assertEqual(32, index.M)
        self.assertEqual(80, index.ef)

    def test_str(self):
        query_strategy = self._get_query_strategy()
        self.assertEqual(
            'SEALS(base_query_strategy=LeastConfidence(), k=5, embed_kwargs={}, normalize=True)',
            str(query_strategy))

    def test_str_non_default(self):
        query_strategy = SEALS(LeastConfidence(), k=40, embed_kwargs={'a': 'b'}, normalize=False)
        self.assertEqual(
            'SEALS(base_query_strategy=LeastConfidence(), k=40, '
            'embed_kwargs={\'a\': \'b\'}, normalize=False)',
            str(query_strategy))
