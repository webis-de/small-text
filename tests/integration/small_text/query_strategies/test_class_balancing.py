import unittest
import numpy as np

from unittest.mock import Mock

from small_text.classifiers import ConfidenceEnhancedLinearSVC
from small_text.query_strategies import (
    ClassBalancer,
    GreedyCoreset
)

from tests.unit.small_text.query_strategies.test_strategies import (
    SamplingStrategiesTests,
    SklearnClassifierWithRandomEmbeddings
)

from tests.utils.datasets import random_sklearn_dataset


class ClassBalancerWithGreedyCoresetTest(unittest.TestCase, SamplingStrategiesTests):

    def _get_clf(self):
        return SklearnClassifierWithRandomEmbeddings(ConfidenceEnhancedLinearSVC(), 2, multi_label=False)

    def _get_query_strategy(self):
        return ClassBalancer(GreedyCoreset(), subsample_size=256)

    def _is_multi_label(self):
        return False

    def test_prediction_with_only_one_label(self, num_samples=100, subsample_size=32):
        query_strategy_mock = Mock(GreedyCoreset())
        query_strategy_mock.query.side_effect = [np.arange(5), np.arange(5)+5]

        strategy = ClassBalancer(query_strategy_mock, subsample_size=subsample_size)

        dataset = random_sklearn_dataset(num_samples, multi_label=False, num_classes=3)
        indices_labeled = np.random.choice(np.arange(num_samples), size=10, replace=False)
        indices_unlabeled = np.array([i for i in range(len(dataset))
                                      if i not in set(indices_labeled)])

        clf_mock = self._get_clf()
        if clf_mock is not None:
            clf_mock.predict = Mock(return_value=np.array([0] * subsample_size))

        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        # no error should be thrown
        strategy.query(clf_mock, dataset, indices_unlabeled, indices_labeled, y)
