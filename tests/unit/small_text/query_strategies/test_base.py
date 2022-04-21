import unittest

import scipy
import numpy as np

from small_text.classifiers.classification import SklearnClassifier, ConfidenceEnhancedLinearSVC
from small_text.query_strategies.base import ClassificationType, constraints
from small_text.query_strategies.strategies import RandomSampling

from tests.utils.datasets import random_sklearn_dataset


@constraints(classification_type='single-label')
class FakeSingleLabelQueryStrategy(RandomSampling):

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        return super().query(clf, dataset, indices_unlabeled, indices_labeled, y, n=n)


@constraints(classification_type='multi-label')
class FakeMultiLabelQueryStrategy(RandomSampling):

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        return super().query(clf, dataset, indices_unlabeled, indices_labeled, y, n=n)


class ClassificationTypeTest(unittest.TestCase):

    def test_from_str(self):
        self.assertEqual(ClassificationType.SINGLE_LABEL,
                         ClassificationType.from_str('single-label'))
        self.assertEqual(ClassificationType.MULTI_LABEL,
                         ClassificationType.from_str('multi-label'))

    def test_from_str_invalid(self):
        with self.assertRaises(ValueError):
            ClassificationType.from_str('does-not-exist')


class ConstraintTest(unittest.TestCase):

    def test_without_constraint(self):
        sls = RandomSampling()
        self._test_query_strategy(sls)

    def test_with_valid_single_label_constraint(self):
        sls = FakeSingleLabelQueryStrategy()
        self._test_query_strategy(sls)

    def test_with_invalid_single_label_constraint(self):
        sls = FakeSingleLabelQueryStrategy()

        with self.assertRaisesRegex(RuntimeError,
                                    'Invalid configuration: This query strategy requires '):
            self._test_query_strategy(sls, multi_label=True)

    def test_with_valid_multi_label_constraint(self):
        sls = FakeMultiLabelQueryStrategy()
        self._test_query_strategy(sls, multi_label=True)

    def test_with_invalid_multi_label_constraint(self):
        sls = FakeMultiLabelQueryStrategy()

        with self.assertRaisesRegex(RuntimeError,
                                    'Invalid configuration: This query strategy requires '):
            self._test_query_strategy(sls)

    def _test_query_strategy(self, query_strategy, multi_label=False, num_classes=3):

        clf = SklearnClassifier(ConfidenceEnhancedLinearSVC(), num_classes)
        ds = random_sklearn_dataset(num_samples=100)

        indices_all = np.arange(len(ds))
        indices_labeled = np.random.choice(indices_all, 10, replace=False)
        indices_unlabeled = np.delete(indices_all, indices_labeled)

        if multi_label:
            y = scipy.sparse.random(100, num_classes, density=0.5, format='csr')
            y.data[np.s_[:]] = 1
            y = y.astype(int)
        else:
            y = np.random.randint(num_classes, size=indices_labeled.shape[0])

        query_strategy.query(clf, ds, indices_unlabeled, indices_labeled, y)
