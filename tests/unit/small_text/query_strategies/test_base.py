import unittest

import scipy
import numpy as np

from small_text.classifiers.classification import SklearnClassifier, ConfidenceEnhancedLinearSVC
from small_text.query_strategies.base import ClassificationType, argselect, constraints
from small_text.query_strategies.strategies import RandomSampling

from numpy.testing import assert_array_almost_equal
from tests.utils.datasets import random_sklearn_dataset


@constraints(classification_type=ClassificationType.SINGLE_LABEL)
class FakeSingleLabelQueryStrategy(RandomSampling):

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        return super().query(clf, dataset, indices_unlabeled, indices_labeled, y, n=n)


@constraints(classification_type='single-label')
class FakeSingleLabelQueryStrategyStringKwarg(RandomSampling):

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        return super().query(clf, dataset, indices_unlabeled, indices_labeled, y, n=n)


@constraints(classification_type=ClassificationType.MULTI_LABEL)
class FakeMultiLabelQueryStrategy(RandomSampling):

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        return super().query(clf, dataset, indices_unlabeled, indices_labeled, y, n=n)


@constraints(classification_type='multi-label')
class FakeMultiLabelQueryStrategyStringKwarg(RandomSampling):

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

    def test_with_valid_single_label_constraint_string_kwarg(self):
        sls = FakeSingleLabelQueryStrategyStringKwarg()
        self._test_query_strategy(sls)

    def test_with_invalid_single_label_constraint(self):
        sls = FakeSingleLabelQueryStrategy()

        with self.assertRaisesRegex(RuntimeError,
                                    'Invalid configuration: This query strategy requires'):
            self._test_query_strategy(sls, multi_label=True)

    def test_with_invalid_single_label_constraint_string_kwarg(self):
        sls = FakeSingleLabelQueryStrategyStringKwarg()

        with self.assertRaisesRegex(RuntimeError,
                                    'Invalid configuration: This query strategy requires'):
            self._test_query_strategy(sls, multi_label=True)

    def test_with_valid_multi_label_constraint(self):
        sls = FakeMultiLabelQueryStrategy()
        self._test_query_strategy(sls, multi_label=True)

    def test_with_valid_multi_label_constraint_string_kwarg(self):
        sls = FakeMultiLabelQueryStrategyStringKwarg()
        self._test_query_strategy(sls, multi_label=True)

    def test_with_invalid_multi_label_constraint(self):
        sls = FakeMultiLabelQueryStrategy()

        with self.assertRaisesRegex(RuntimeError,
                                    'Invalid configuration: This query strategy requires '):
            self._test_query_strategy(sls)

    def test_with_invalid_multi_label_constraint_string_kwarg(self):
        sls = FakeMultiLabelQueryStrategyStringKwarg()

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


class ArgselectTest(unittest.TestCase):

    def test_argselect_without_ties_maximum(self, n=5):

        arr = np.arange(0.1, 1.1, step=0.1)
        np.random.shuffle(arr)

        indices = argselect(arr, n)
        assert_array_almost_equal(np.array([0.6, 0.7, 0.8, 0.9, 1.0]), np.sort(arr[indices]))

    def test_argselect_without_ties_minimum(self, n=5):

        arr = np.arange(0.1, 1.1, step=0.1)
        np.random.shuffle(arr)

        indices = argselect(arr, n, maximum=False)
        assert_array_almost_equal(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), np.sort(arr[indices]))

    def test_argselect_with_ties_within_window_minimum(self, n=5):
        arr = np.array([0.1, 0.2, 0.3, 0.3, 0.3,
                        0.4, 0.4, 0.4, 0.4, 0.4])
        np.random.shuffle(arr)

        indices = argselect(arr, n, maximum=False)
        assert_array_almost_equal(np.array([0.1, 0.2, 0.3, 0.3, 0.3]), np.sort(arr[indices]))

    def test_argselect_with_ties_outside_window_minimum(self, n=5):
        arr = np.array([0.1, 0.2, 0.3, 0.3, 0.3,
                        0.3, 0.3, 0.3, 0.3, 0.3,
                        0.3, 0.3, 0.3, 0.3, 0.5])
        np.random.shuffle(arr)

        indices = argselect(arr, n, maximum=False)
        assert_array_almost_equal(np.array([0.1, 0.2, 0.3, 0.3, 0.3]), np.sort(arr[indices]))

    def test_argselect_with_ties_within_window_maximum(self, n=5):
        arr = np.array([0.1, 0.2, 0.3, 0.3, 0.3,
                        0.4, 0.5, 0.5, 0.5, 0.6])
        np.random.shuffle(arr)

        indices = argselect(arr, n, maximum=True)
        assert_array_almost_equal(np.array([0.4, 0.5, 0.5, 0.5, 0.6]), np.sort(arr[indices]))

    def test_argselect_with_ties_outside_window_maximum(self, n=5):
        arr = np.array([0.1, 0.2, 0.3, 0.3, 0.3,
                        0.3, 0.3, 0.3, 0.3, 0.3,
                        0.3, 0.3, 0.3, 0.3, 0.5])
        np.random.shuffle(arr)

        indices = argselect(arr, n, maximum=True)
        assert_array_almost_equal(np.array([0.3, 0.3, 0.3, 0.3, 0.5]), np.sort(arr[indices]))

    def test_argselect_n_equals_input_size(self):
        arr = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.9])
        np.random.shuffle(arr)

        indices = argselect(arr, arr.shape[0])
        assert_array_almost_equal(arr, arr[indices])

        indices = argselect(arr, arr.shape[0], maximum=False)
        assert_array_almost_equal(arr, arr[indices])

    def test_argselect_n_greater_input_size(self):

        arr = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.9])

        with self.assertRaisesRegex(ValueError, r'n=11 out of bounds of array with shape \(10,\)'):
            argselect(arr, arr.shape[0]+1)

        with self.assertRaisesRegex(ValueError, r'n=11 out of bounds of array with shape \(10,\)'):
            argselect(arr, arr.shape[0]+1, maximum=False)
