import unittest
import pytest

import numpy as np

from unittest.mock import Mock, patch

from numpy.testing import assert_array_equal
from sklearn.base import clone
from sklearn.preprocessing import normalize

from small_text.query_strategies.exceptions import EmptyPoolException
from small_text.classifiers import ConfidenceEnhancedLinearSVC, SklearnClassifierFactory
from small_text.query_strategies import (
    BreakingTies,
    CategoryVectorInconsistencyAndRanking,
    ContrastiveActiveLearning,
    DiscriminativeActiveLearning,
    EmbeddingKMeans,
    GreedyCoreset,
    LeastConfidence,
    LightweightCoreset,
    SEALS,
    SubsamplingQueryStrategy
)
from tests.unit.small_text.query_strategies.test_strategies import (
    SamplingStrategiesTests,
    SklearnClassifierWithRandomEmbeddingsAndProba
)

from tests.utils.datasets import random_sklearn_dataset, random_matrix_data
from tests.utils.object_factory import get_initialized_active_learner


class SklearnClassifierWithRandomEmbeddingsAndProbaFactory(SklearnClassifierFactory):

    def new(self):
        return SklearnClassifierWithRandomEmbeddingsAndProba(clone(self.base_estimator),
                                                             self.num_classes,
                                                             **self.kwargs)


class QueryStrategiesExhaustiveIntegrationTest(object):

    def _get_dataset(self, num_classes, multi_label=False):
        return random_sklearn_dataset(60, multi_label=multi_label, num_classes=num_classes)

    def _get_factory(self, num_classes, multi_label=False):
        return SklearnClassifierWithRandomEmbeddingsAndProbaFactory(
            ConfidenceEnhancedLinearSVC(),
            num_classes,
            kwargs={'multi_label': multi_label}
        )

    def _simple_exhaustive_active_learning_test(self, query_strategy, query_size=10,
                                                num_classes=6, num_initial=30, multi_label=False):
        dataset = self._get_dataset(num_classes, multi_label=multi_label)
        clf_factory = self._get_factory(num_classes, multi_label=multi_label)
        active_learner = get_initialized_active_learner(clf_factory, query_strategy, dataset,
                                                        initial_indices=num_initial,
                                                        num_classes=num_classes,
                                                        multi_label=multi_label)

        for _ in range(3):
            active_learner.query()
            if multi_label:
                _, y_new = random_matrix_data('dense', 'sparse', num_samples=10,
                                              num_labels=num_classes)
            else:
                _, y_new = random_matrix_data('dense', 'dense', num_samples=10,
                                              num_labels=num_classes)
            active_learner.update(y_new)

        with self.assertRaises(EmptyPoolException):
            active_learner.query()

        self.assertEqual(query_size * 3 + num_initial, active_learner.indices_labeled.shape[0])


class QueryStrategiesTest(QueryStrategiesExhaustiveIntegrationTest, unittest.TestCase):

    def test_breaking_ties(self):
        query_strategy = BreakingTies()
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=3)

    def test_category_vector_inconsistency_and_ranking(self):
        query_strategy = CategoryVectorInconsistencyAndRanking()
        self._simple_exhaustive_active_learning_test(query_strategy, multi_label=True,
                                                     num_classes=3)

    def test_contrastive_active_learning(self):
        query_strategy = ContrastiveActiveLearning()
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=3)

    @pytest.mark.skip(reason='not yet supported')
    def test_discriminative_active_learning(self):
        classifier_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC(), num_classes=3)
        query_strategy = DiscriminativeActiveLearning(classifier_factory, 2)
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=3)

    def test_embedding_kmeans(self):
        query_strategy = EmbeddingKMeans()
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=3)

    def test_greedy_coreset(self):
        query_strategy = GreedyCoreset()
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=3)

    def test_lightweight_coreset(self):
        query_strategy = LightweightCoreset()
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=3)

    @pytest.mark.optional
    def test_seals(self):
        query_strategy = SEALS(LeastConfidence(), k=5)
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=3)

    def test_subsampling_query_strategy(self):
        query_strategy = SubsamplingQueryStrategy(LeastConfidence(), subsample_size=10)
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=3)


@pytest.mark.optional
class SEALSTest(unittest.TestCase, SamplingStrategiesTests):

    def _get_clf(self):
        return SklearnClassifierWithRandomEmbeddingsAndProba(ConfidenceEnhancedLinearSVC(), 2)

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
