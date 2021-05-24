import unittest
import numpy as np

from unittest.mock import patch, Mock, ANY
from numpy.testing import assert_array_equal

from active_learning.active_learner import (PoolBasedActiveLearner,
    ActiveLearner, AbstractPoolBasedActiveLearner)
from active_learning.exceptions import LearnerNotInitializedException
from active_learning.classifiers import ConfidenceEnhancedLinearSVC
from active_learning.classifiers.factories import SklearnClassifierFactory
from active_learning.data import SklearnDataSet
from active_learning.query_strategies import RandomSampling

from tests.utils.datasets import random_matrix_data


class ActiveLearnerTest(unittest.TestCase):

    def test_abstract_active_learner(self):
        with self.assertRaises(TypeError):
            ActiveLearner()


class AbstractPoolBasedActiveLearnerTest(unittest.TestCase):

    def test_abstract_active_learner(self):
        with self.assertRaises(TypeError):
            AbstractPoolBasedActiveLearner()


class PoolBasedActiveLearnerTest(unittest.TestCase):

    dataset_num_samples = 100

    def _set_up_active_learner(self):
        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())
        query_strategy = RandomSampling()
        x = SklearnDataSet(*random_matrix_data('dense', self.dataset_num_samples))
        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)
        x_indices_initial = np.random.choice(np.arange(100), size=10)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        active_learner.initialize_data(x_indices_initial, y_initial)

        return active_learner

    def test_init(self):

        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())
        query_strategy = RandomSampling()
        x = np.random.rand(100, 10)

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)

        self.assertEqual(None, active_learner.classifier)
        self.assertEqual(query_strategy, active_learner.query_strategy)
        self.assertFalse(active_learner.incremental_training)

    def test_initialize_data(self):
        self._test_initialize_data(retrain=True)
        self._test_initialize_data(retrain=False)

    def test_initialize_data_with_validation_set_given(self):
        self._test_initialize_data(retrain=True, x_indices_validation='random')
        self._test_initialize_data(retrain=False, x_indices_validation='random')

    def _test_initialize_data(self, retrain=True, x_indices_validation=None):

        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())
        clf_mock = Mock(clf_factory.new())
        clf_factory.new = Mock(return_value=clf_mock)
        query_strategy = RandomSampling()
        x = np.random.rand(100, 10)

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)
        self.assertIsNone(active_learner._label_to_position)
        assert_array_equal(np.empty(0, dtype=np.int64), active_learner.x_indices_labeled)
        assert_array_equal(np.empty(0, dtype=np.int64), active_learner.y)

        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        if x_indices_validation == 'random':
            x_indices_labeled_indices = list(range(len(x_indices_initial)))
            x_indices_validation = np.random.choice(x_indices_labeled_indices, size=1, replace=False)

        active_learner.initialize_data(x_indices_initial, y_initial, retrain=retrain,
                                       x_indices_validation=x_indices_validation)

        self.assertIsNotNone(active_learner._label_to_position)
        assert_array_equal(y_initial, active_learner.y)
        assert_array_equal(x_indices_initial, active_learner.x_indices_labeled)

        if retrain:
            clf_factory.new.assert_called_with()
            if x_indices_validation is not None:
                self.assertEqual((9, 10), clf_mock.fit.call_args[0][0].shape)
            else:
                self.assertEqual((10, 10), clf_mock.fit.call_args[0][0].shape)
        else:
            clf_factory.new.assert_not_called()
            clf_mock.fit.assert_not_called()

    def test_query_without_prior_initialize_data(self):
        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())
        query_strategy_mock = Mock()
        x = np.random.rand(100, 10)

        with self.assertRaises(LearnerNotInitializedException):
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy_mock, x)
            active_learner.query(num_samples=5)

    def test_query(self):

        num_samples = 5

        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())
        query_strategy_mock = Mock()
        x = SklearnDataSet(*random_matrix_data('dense', self.dataset_num_samples))

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy_mock, x)

        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        active_learner.initialize_data(x_indices_initial, y_initial)

        active_learner.query(num_samples=num_samples)

        query_strategy_mock.query.assert_called_with(ANY, x, ANY, x_indices_initial, y_initial,
                                                     n=num_samples)

    def test_query_with_custom_representation(self):

        num_samples = 5

        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())
        query_strategy_mock = Mock()
        x = SklearnDataSet(*random_matrix_data('dense', self.dataset_num_samples))

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy_mock, x)

        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        active_learner.initialize_data(x_indices_initial, y_initial)

        custom_x = np.random.rand(100, 3)
        active_learner.query(num_samples=num_samples, x=custom_x)

        query_strategy_mock.query.assert_called_with(ANY, custom_x, ANY, x_indices_initial, y_initial,
                                                     n=num_samples)

    def test_query_with_custom_representation_invalid(self):

        num_samples = 5

        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())
        query_strategy_mock = Mock()
        x = SklearnDataSet(*random_matrix_data('dense', self.dataset_num_samples))

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy_mock, x)

        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        active_learner.initialize_data(x_indices_initial, y_initial)

        custom_x = np.random.rand(98, 3)  # <-- representation's first dimension differs from x
        assert custom_x.shape[0] != len(x)

        with self.assertRaises(ValueError):
            active_learner.query(num_samples=num_samples, x=custom_x)

    def test_query_with_kwargs(self):

        num_samples = 5
        kwargs = {'argx_x': 1, 'arg_y': 2}

        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())
        query_strategy_mock = Mock()
        x = SklearnDataSet(*random_matrix_data('dense', self.dataset_num_samples))

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy_mock, x)

        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        active_learner.initialize_data(x_indices_initial, y_initial)

        active_learner.query(num_samples=num_samples, query_strategy_kwargs=kwargs)

        query_strategy_mock.query.assert_called_with(ANY, x, ANY, x_indices_initial, y_initial,
                                                     n=num_samples, **kwargs)

    def test_update(self):
        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())
        query_strategy = RandomSampling()
        x = np.random.rand(100, 10)

        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)

            x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
            y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            active_learner.initialize_data(x_indices_initial, y_initial, len(np.unique(y_initial)))

            assert_array_equal(x_indices_initial, active_learner.x_indices_labeled)
            assert_array_equal(y_initial, active_learner.y)

            indices_new = active_learner.query(num_samples=5)
            y_new = np.random.randint(0, 1, size=5)
            active_learner.update(y_new)
            assert_array_equal(np.concatenate([x_indices_initial, indices_new]),
                               active_learner.x_indices_labeled)
            assert_array_equal(np.concatenate([y_initial, y_new]), active_learner.y)

        retrain_mock.assert_called_with(x_indices_validation=None)

    def test_update_with_validation_set_given(self):
        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())
        query_strategy = RandomSampling()
        x = np.random.rand(100, 10)

        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)

            x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
            y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            active_learner.initialize_data(x_indices_initial, y_initial, len(np.unique(y_initial)))

            x_indices_validation = np.random.choice(x_indices_initial, size=10, replace=False)

            assert_array_equal(x_indices_initial, active_learner.x_indices_labeled)
            assert_array_equal(y_initial, active_learner.y)

            indices_new = active_learner.query(num_samples=5)
            y_new = np.random.randint(0, 1, size=5)
            active_learner.update(y_new, x_indices_validation=x_indices_validation)
            assert_array_equal(np.concatenate([x_indices_initial, indices_new]),
                               active_learner.x_indices_labeled)
            assert_array_equal(np.concatenate([y_initial, y_new]), active_learner.y)

        retrain_mock.assert_called_with(x_indices_validation=x_indices_validation)

    def test_update_incremental_lerner(self):
        query_strategy = RandomSampling()
        x = np.random.rand(100, 10)

        clf_mock = ConfidenceEnhancedLinearSVC()
        clf_mock.fit = Mock()
        clf_factory_mock = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())
        clf_factory_mock.new = Mock()
        clf_factory_mock.new.side_effect = [clf_mock, None]

        active_learner = PoolBasedActiveLearner(clf_factory_mock, query_strategy, x, incremental_training=True)

        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        active_learner.initialize_data(x_indices_initial, y_initial, len(np.unique(y_initial)))

        assert_array_equal(x_indices_initial, active_learner.x_indices_labeled)
        assert_array_equal(y_initial, active_learner.y)

        for _ in range(2):
            active_learner.query(num_samples=5)
            y_new = np.random.randint(0, 1, size=5)
            active_learner.update(y_new)

        clf_factory_mock.new.assert_called_once_with()
        self.assertEqual(active_learner._clf, clf_mock)
        active_learner._clf.fit.assert_called()

    def test_query_update_mismatch(self):

        active_learner = self._set_up_active_learner()

        indices = active_learner.query(num_samples=5)
        self.assertEqual(5, len(indices))

        y_new = np.random.randint(0, 1, size=6)
        with self.assertRaises(ValueError):
            active_learner.update(y_new)

    def test_update_duplicate_indices_in_labeled_pool(self):
        query_strategy = RandomSampling()
        x = np.random.rand(100, 10)

        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)
        active_learner._retrain = Mock()

        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        active_learner.initialize_data(x_indices_initial, y_initial, len(np.unique(y_initial)))

        with self.assertRaises(ValueError):
            y_new = np.random.randint(0, 1, size=5)

            active_learner.query(num_samples=5)
            active_learner.queried_indices = x_indices_initial
            active_learner.update(y_new)

        active_learner._retrain.assert_called_once()

    def test_update_label_at(self):
        self._test_update_label_at()

    def test_update_label_at_with_retrain_true(self):
        self._test_update_label_at(retrain=True)

    def test_update_label_at_with_validation_set_given(self):
        self._test_update_label_at(x_indices_validation='random')

    def _test_update_label_at(self, retrain=False, x_indices_validation=None):
        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())
        query_strategy = RandomSampling()
        x = np.random.rand(100, 10)
        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)

            x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
            y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            active_learner.initialize_data(x_indices_initial, y_initial, len(np.unique(y_initial)))

            # query & update
            indices_new = active_learner.query(num_samples=5)
            y_new = [0, 0, 0, 1, 1]
            if x_indices_validation == 'random':
                x_indices_validation = np.random.choice(x_indices_initial, size=10, replace=False)
            active_learner.update(y_new, x_indices_validation=x_indices_validation)

            active_learner.update_label_at(indices_new[0], 1, retrain=retrain)
            assert_array_equal([1, 0, 0, 1, 1], active_learner.y[-5:])
        retrain_mock.assert_called_with(x_indices_validation=x_indices_validation)

        if retrain:
            retrain_mock.assert_called_with(x_indices_validation=x_indices_validation)

    def test_remove_label_at(self):
        self._test_remove_label_at(x_index=-2)
        self._test_remove_label_at(x_index=-1)

    def test_remove_label_at_with_validation_set_given(self):
        self._test_remove_label_at(x_index=-1, retrain=True, x_indices_validation='random')

    def _test_remove_label_at(self, x_index=-5, retrain=False, x_indices_validation=None):
        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())
        query_strategy = RandomSampling()
        x = np.random.rand(100, 10)
        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)

            x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
            y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            active_learner.initialize_data(x_indices_initial, y_initial, len(np.unique(y_initial)))

            # query & update
            active_learner.query(num_samples=5)
            y_new = [0, 0, 0, 1, 1]
            if x_indices_validation == 'random':
                x_indices_validation = np.random.choice(x_indices_initial, size=10, replace=False)
            active_learner.update(y_new)

            active_learner.remove_label_at(active_learner.x_indices_labeled[x_index],
                                           retrain=retrain, x_indices_validation=x_indices_validation)
            assert_array_equal([0, 0, 0, 1], active_learner.y[-4:])

            self.assertEqual(14, len(active_learner.y))
            self.assertEqual(14, len(active_learner.x_indices_labeled))

        retrain_mock.assert_called_with(x_indices_validation=x_indices_validation)
