import unittest
import numpy as np

from unittest import mock

from small_text import (
    ConfidenceEnhancedLinearSVC,
    RandomSampling,
    SklearnClassifier,
    SklearnDataset
)
from small_text.data.sampling import _get_class_histogram
from small_text.query_strategies.class_balancing import ClassBalancer, _get_rebalancing_distribution

from tests.utils.testing_numpy import AnyNumpyArrayOfShape


class RebalanceDistributionTest(unittest.TestCase):

    def test_get_rebalanced_target_distribution(self):

        num_classes = 5
        num_samples = 10

        # histogram of y: [1, 1, 1, 1, 1]
        y = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20)

        current_distribution = _get_class_histogram(y, num_classes)
        balancing_distribution = _get_rebalancing_distribution(num_samples, num_classes, y, y_pred)

        final_distribution = current_distribution + balancing_distribution

        self.assertTrue(np.all(final_distribution == 3))

    def test_get_rebalanced_target_distribution_for_imbalanced_distribution(self):

        num_classes = 5
        num_samples = 10

        # histogram of y: [1, 3, 0, 3, 0]
        y = np.array([0, 1, 1, 1, 3, 3, 3, 3])
        y_pred = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20)

        current_distribution = _get_class_histogram(y, num_classes)
        balancing_distribution = _get_rebalancing_distribution(num_samples, num_classes, y, y_pred)

        final_distribution = current_distribution + balancing_distribution

        self.assertTrue(np.all(final_distribution > 2))
        self.assertTrue(np.all(final_distribution <= 4))

    def test_get_rebalanced_target_distribution_with_ignored_classes(self):

        num_classes = 5
        num_samples = 10

        # histogram of y: [1, 3, 0, 3, 0]
        y = np.array([0, 1, 1, 1, 3, 3, 3, 3])
        y_pred = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20)

        current_distribution = _get_class_histogram(y, num_classes)
        balancing_distribution = _get_rebalancing_distribution(num_samples, num_classes, y, y_pred,
                                                               ignored_classes=[0, 4])

        final_distribution = current_distribution + balancing_distribution

        self.assertTrue(final_distribution[0] == 1)
        self.assertTrue(final_distribution[4] == 0)
        self.assertTrue(np.all(final_distribution <= 6))

    def test_get_rebalanced_target_distribution_fallback_when_not_enough_active_classes_are_sampled(self):

        num_classes = 5
        num_samples = 10

        y = np.array([], dtype=int)
        y_pred = np.array([0] * 5 + [1] * 3 + [2] * 3 + [3] * 2 + [4] * 20)

        current_distribution = _get_class_histogram(y, num_classes)
        balancing_distribution = _get_rebalancing_distribution(num_samples, num_classes, y, y_pred,
                                                               ignored_classes=[0, 4])

        final_distribution = current_distribution + balancing_distribution
        self.assertTrue((final_distribution[0] > 1) or (final_distribution[4] > 0))


class ClassBalancerSubsamplingTest(unittest.TestCase):

    def test_subsample_when_not_enough_samples(self, n=10, num_samples=100, num_classes=2):

        clf = SklearnClassifier(ConfidenceEnhancedLinearSVC(), num_classes)

        dataset = SklearnDataset(np.random.rand(num_samples, 10),
                                 np.random.randint(0, high=2, size=num_samples))

        indices_labeled = np.random.choice(np.arange(num_samples), size=10, replace=False)
        indices_unlabeled = np.array([i for i in np.arange(num_samples) if i not in set(indices_labeled)])
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        clf.fit(dataset[indices_labeled])

        query_strategy = ClassBalancer(RandomSampling(), subsample_size=4096)

        with mock.patch.object(query_strategy,
                               '_query_class_balanced',
                               wraps=query_strategy._query_class_balanced) as query_class_balanced:
            query_strategy.query(clf, dataset, indices_unlabeled, indices_labeled, y, n=n)

        query_class_balanced.assert_called_once_with(clf, dataset, indices_unlabeled, indices_labeled, y, n)

    def test_subsample_when_subsampling_is_none(self, n=10, num_samples=100, num_classes=2):

        clf = SklearnClassifier(ConfidenceEnhancedLinearSVC(), num_classes)

        dataset = SklearnDataset(np.random.rand(num_samples, 10),
                                 np.random.randint(0, high=2, size=num_samples))

        indices_labeled = np.random.choice(np.arange(num_samples), size=10, replace=False)
        indices_unlabeled = np.array([i for i in np.arange(num_samples) if i not in set(indices_labeled)])
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        clf.fit(dataset[indices_labeled])

        query_strategy = ClassBalancer(RandomSampling(), subsample_size=None)

        with mock.patch.object(query_strategy,
                               '_query_class_balanced',
                               wraps=query_strategy._query_class_balanced) as query_class_balanced:
            query_strategy.query(clf, dataset, indices_unlabeled, indices_labeled, y, n=n)

        query_class_balanced.assert_called_once_with(clf, dataset, indices_unlabeled, indices_labeled, y, n)

    def test_subsample(self, n=10, num_samples=100, num_classes=2):

        clf = SklearnClassifier(ConfidenceEnhancedLinearSVC(), num_classes)

        dataset = SklearnDataset(np.random.rand(num_samples, 10),
                                 np.random.randint(0, high=2, size=num_samples))

        indices_labeled = np.random.choice(np.arange(num_samples), size=10, replace=False)
        indices_unlabeled = np.array([i for i in np.arange(num_samples) if i not in set(indices_labeled)])
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        clf.fit(dataset[indices_labeled])

        query_strategy = ClassBalancer(RandomSampling(), subsample_size=32)

        with mock.patch.object(query_strategy,
                               '_query_class_balanced',
                               wraps=query_strategy._query_class_balanced) as query_class_balanced:
            query_strategy.query(clf, dataset, indices_unlabeled, indices_labeled, y, n=n)

        query_class_balanced.assert_called_once_with(
            clf,
            dataset,
            AnyNumpyArrayOfShape((32,)),
            AnyNumpyArrayOfShape((10,)),
            y,
            n
        )


class ClassBalancerTest(unittest.TestCase):

    # TODO: base tests and test _validate_query_input()
    def test_query(self, n=10, num_samples=100, num_classes=4):

        clf = SklearnClassifier(ConfidenceEnhancedLinearSVC(), num_classes)

        dataset = SklearnDataset(np.random.rand(num_samples, 10),
                                 np.random.randint(0, high=2, size=num_samples))

        indices_labeled = np.random.choice(np.arange(num_samples), size=10, replace=False)
        indices_unlabeled = np.array([i for i in np.arange(num_samples) if i not in set(indices_labeled)])
        y = np.array([0, 1, 1, 1, 1, 1, 3, 2, 3, 2])

        clf.fit(dataset[indices_labeled])

        query_strategy = ClassBalancer(RandomSampling(), subsample_size=32)

        with mock.patch.object(clf, 'predict') as predict_mock:
            predict_mock.return_value = np.array([0] * 8 + [1] * 8 + [2] * 8 + [3] * 8)
            query_strategy.query(clf, dataset, indices_unlabeled, indices_labeled, y, n=n)

    def test_class_balancer_str(self):
        strategy = ClassBalancer(RandomSampling(), ignored_classes=[2, 3], subsample_size=4096)
        expected_str = 'ClassBalancer(base_query_strategy=RandomSampling(), ' \
                       'ignored_classes=[2, 3], subsample_size=4096)'
        self.assertEqual(expected_str, str(strategy))
