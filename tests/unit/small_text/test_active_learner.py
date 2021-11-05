import unittest
import numpy as np

from unittest.mock import call, patch, Mock, ANY
from numpy.testing import assert_array_equal

from small_text.active_learner import (PoolBasedActiveLearner,
                                       ActiveLearner, AbstractPoolBasedActiveLearner)
from small_text.exceptions import LearnerNotInitializedException
from small_text.classifiers import ConfidenceEnhancedLinearSVC
from small_text.classifiers.factories import SklearnClassifierFactory
from small_text.data import SklearnDataset
from small_text.query_strategies import RandomSampling

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

    def _get_classifier_factory(self):
        return SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())

    def _set_up_active_learner(self):
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        x = SklearnDataset(*random_matrix_data('dense', self.dataset_num_samples))
        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)
        x_indices_initial = np.random.choice(np.arange(100), size=10)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        active_learner.initialize_data(x_indices_initial, y_initial)

        return active_learner

    def test_init(self):

        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        x = np.random.rand(100, 10)

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)

        self.assertEqual(None, active_learner.classifier)
        self.assertEqual(query_strategy, active_learner.query_strategy)
        self.assertFalse(active_learner.incremental_training)

    def test_initialize_data(self):
        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        self._test_initialize_data(x_indices_initial, retrain=True)
        self._test_initialize_data(x_indices_initial, retrain=False)

    def test_initialize_data_with_indices_ignored(self):
        x_indices_random = np.random.choice(np.arange(100), size=15, replace=False)
        self._test_initialize_data(x_indices_random[:10], x_indices_ignored=x_indices_random[10:15], retrain=True)
        self._test_initialize_data(x_indices_random[:10], x_indices_ignored=x_indices_random[10:15], retrain=False)

    def test_initialize_data_with_validation_set_given(self):
        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        self._test_initialize_data(x_indices_initial, retrain=True, x_indices_validation='random')
        self._test_initialize_data(x_indices_initial, retrain=False, x_indices_validation='random')

    def _test_initialize_data(self, x_indices_initial, retrain=True, x_indices_ignored=None, x_indices_validation=None):

        clf_factory = self._get_classifier_factory()
        clf_mock = Mock(clf_factory.new())
        clf_factory.new = Mock(return_value=clf_mock)
        query_strategy = RandomSampling()
        x = np.random.rand(100, 10)

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)
        self.assertIsNone(active_learner._label_to_position)
        assert_array_equal(np.empty(0, dtype=np.int64), active_learner.x_indices_labeled)
        assert_array_equal(np.empty(0, dtype=np.int64), active_learner.y)

        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        if x_indices_validation == 'random':
            x_indices_labeled_indices = list(range(len(x_indices_initial)))
            x_indices_validation = np.random.choice(x_indices_labeled_indices, size=1, replace=False)

        active_learner.initialize_data(x_indices_initial, y_initial, retrain=retrain,
                                       x_indices_ignored=x_indices_ignored,
                                       x_indices_validation=x_indices_validation)

        self.assertIsNotNone(active_learner._label_to_position)
        assert_array_equal(y_initial, active_learner.y)
        assert_array_equal(x_indices_initial, active_learner.x_indices_labeled)
        if x_indices_ignored is not None:
            assert_array_equal(x_indices_ignored, active_learner.x_indices_ignored)
        else:
            assert_array_equal(np.array([]), active_learner.x_indices_ignored)

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
        clf_factory = self._get_classifier_factory()
        query_strategy_mock = Mock()
        x = np.random.rand(100, 10)

        with self.assertRaises(LearnerNotInitializedException):
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy_mock, x)
            active_learner.query(num_samples=5)

    def test_query(self, num_samples=5):

        clf_factory = self._get_classifier_factory()
        query_strategy_mock = Mock()
        x = SklearnDataset(*random_matrix_data('dense', self.dataset_num_samples))

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy_mock, x)

        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        active_learner.initialize_data(x_indices_initial, y_initial)

        active_learner.query(num_samples=num_samples)

        query_strategy_mock.query.assert_called_with(ANY, x, ANY, x_indices_initial, y_initial,
                                                     n=num_samples)

    def test_query_with_custom_representation(self, num_samples=5):

        clf_factory = self._get_classifier_factory()
        query_strategy_mock = Mock()
        x = SklearnDataset(*random_matrix_data('dense', self.dataset_num_samples))

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy_mock, x)

        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        active_learner.initialize_data(x_indices_initial, y_initial)

        custom_x = np.random.rand(100, 3)
        active_learner.query(num_samples=num_samples, x=custom_x)

        query_strategy_mock.query.assert_called_with(ANY, custom_x, ANY, x_indices_initial, y_initial,
                                                     n=num_samples)

    def test_query_with_custom_representation_invalid(self, num_samples=5):

        clf_factory = self._get_classifier_factory()
        query_strategy_mock = Mock()
        x = SklearnDataset(*random_matrix_data('dense', self.dataset_num_samples))

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

        clf_factory = self._get_classifier_factory()
        query_strategy_mock = Mock()
        x = SklearnDataset(*random_matrix_data('dense', self.dataset_num_samples))

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy_mock, x)

        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        active_learner.initialize_data(x_indices_initial, y_initial)

        active_learner.query(num_samples=num_samples, query_strategy_kwargs=kwargs)

        query_strategy_mock.query.assert_called_with(ANY, x, ANY, x_indices_initial, y_initial,
                                                     n=num_samples, **kwargs)

    def test_update(self):
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        x = np.random.rand(100, 10)

        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)

            x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
            y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            active_learner.initialize_data(x_indices_initial, y_initial)

            assert_array_equal(x_indices_initial, active_learner.x_indices_labeled)
            assert_array_equal(y_initial, active_learner.y)

            indices_new = active_learner.query(num_samples=5)
            y_new = np.random.randint(0, 1, size=5)
            active_learner.update(y_new)
            assert_array_equal(np.concatenate([x_indices_initial, indices_new]),
                               active_learner.x_indices_labeled)
            assert_array_equal(np.concatenate([y_initial, y_new]), active_learner.y)

        retrain_mock.assert_called_with(x_indices_validation=None)

    def test_update_with_ignored_samples(self):
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        x = np.random.rand(100, 10)

        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)

            x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
            y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            active_learner.initialize_data(x_indices_initial, y_initial)

            assert_array_equal(x_indices_initial, active_learner.x_indices_labeled)
            assert_array_equal(y_initial, active_learner.y)

            indices_new = active_learner.query(num_samples=5)
            y_new = np.array([None, 1, None, 0, None])  # ignore samples 0, 2, and 4
            active_learner.update(y_new)
            assert_array_equal(np.concatenate([x_indices_initial, [indices_new[1], indices_new[3]]]),
                               active_learner.x_indices_labeled)
            assert_array_equal(np.concatenate([y_initial, [y_new[1], y_new[3]]]), active_learner.y)

        retrain_mock.assert_called_with(x_indices_validation=None)

    def test_update_with_only_ignored_samples(self, query_size=5):
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        x = np.random.rand(100, 10)

        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)

            x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
            y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            active_learner.initialize_data(x_indices_initial, y_initial)

            assert_array_equal(x_indices_initial, active_learner.x_indices_labeled)
            assert_array_equal(y_initial, active_learner.y)

            active_learner.query(num_samples=query_size)
            y_new = np.array([None]*query_size)  # ignore all samples
            active_learner.update(y_new)

            assert_array_equal(x_indices_initial, active_learner.x_indices_labeled)
            assert_array_equal(y_initial, active_learner.y)

        retrain_mock.assert_has_calls([call(x_indices_validation=None)])

    def test_update_with_validation_set_given(self):
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        x = np.random.rand(100, 10)

        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)

            x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
            y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            x_indices_validation = np.random.choice(x_indices_initial, size=10, replace=False)

            active_learner.initialize_data(x_indices_initial, y_initial, x_indices_validation=x_indices_validation)

            assert_array_equal(x_indices_initial, active_learner.x_indices_labeled)
            assert_array_equal(y_initial, active_learner.y)

            indices_new = active_learner.query(num_samples=5)
            y_new = np.random.randint(0, 1, size=5)
            active_learner.update(y_new, x_indices_validation=x_indices_validation)
            assert_array_equal(np.concatenate([x_indices_initial, indices_new]),
                               active_learner.x_indices_labeled)
            assert_array_equal(np.concatenate([y_initial, y_new]), active_learner.y)

        retrain_mock.assert_has_calls([call(x_indices_validation=x_indices_validation),
                                       call(x_indices_validation=x_indices_validation)])

    def test_update_incremental_training(self):
        query_strategy = RandomSampling()
        x = np.random.rand(100, 10)

        clf_mock = ConfidenceEnhancedLinearSVC()
        clf_mock.fit = Mock()
        clf_factory_mock = self._get_classifier_factory()
        clf_factory_mock.new = Mock()
        clf_factory_mock.new.side_effect = [clf_mock, None]

        active_learner = PoolBasedActiveLearner(clf_factory_mock, query_strategy, x, incremental_training=True)

        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        active_learner.initialize_data(x_indices_initial, y_initial)

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

        active_learner.initialize_data(x_indices_initial, y_initial)

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
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        x = np.random.rand(100, 10)
        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)

            x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
            y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            active_learner.initialize_data(x_indices_initial, y_initial)

            # query & update
            indices_new = active_learner.query(num_samples=5)
            y_new = [0, 0, 0, 1, 1]
            if x_indices_validation == 'random':
                x_indices_validation = np.random.choice(x_indices_initial, size=10, replace=False)
            active_learner.update(y_new, x_indices_validation=x_indices_validation)

            active_learner.update_label_at(indices_new[0], 1, retrain=retrain)
            assert_array_equal([1, 0, 0, 1, 1], active_learner.y[-5:])
        if retrain:
            retrain_mock.assert_has_calls([call(x_indices_validation=x_indices_validation),
                                           call(x_indices_validation=x_indices_validation)])

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
            active_learner.initialize_data(x_indices_initial, y_initial)

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

    def test_ignore_sample_at(self):
        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        unlabeled_indices = self._get_unlabeled_indices(x_indices_initial)
        self._test_ignore_sample_without_label_change(x_indices_initial, y_initial, unlabeled_indices[0])

    def test_ignore_sample_at_with_retrain_true(self):
        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        unlabeled_indices = self._get_unlabeled_indices(x_indices_initial)
        self._test_ignore_sample_without_label_change(x_indices_initial, y_initial, unlabeled_indices[0], retrain=True)

    def test_ignore_sample_at_with_validation_set_given(self):
        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        unlabeled_indices = self._get_unlabeled_indices(x_indices_initial)
        x_indices_validation = np.random.choice(x_indices_initial, size=3, replace=False)
        self._test_ignore_sample_without_label_change(x_indices_initial, y_initial, unlabeled_indices[0],
                                                      x_indices_validation=x_indices_validation)

    def test_ignore_sample_at_with_label_removed(self):
        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self._test_ignore_sample_with_label_change(x_indices_initial, y_initial, x_indices_initial[0])

    def test_ignore_sample_at_with_label_removed_and_retrain_true(self):
        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self._test_ignore_sample_with_label_change(x_indices_initial, y_initial, x_indices_initial[0], retrain=True)

    def test_ignore_sample_at_with_label_removed_and_retrain_true_and_validation_set_given(self):
        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self._test_ignore_sample_with_label_change(x_indices_initial, y_initial, x_indices_initial[0],
                                                   retrain=True, x_indices_validation=x_indices_initial[1:3])

    def _get_unlabeled_indices(self, x_indices_initial, num_samples=100):

        mask = np.ones(num_samples, bool)
        mask[x_indices_initial] = False
        indices = np.arange(num_samples)

        return indices[mask]

    def _test_ignore_sample_without_label_change(self, x_indices_initial, y_initial, x_index_to_ignore,
                                                 num_samples=100, retrain=False, x_indices_validation=None):
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        x = np.random.rand(num_samples, 10)
        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)

            active_learner.initialize_data(x_indices_initial, y_initial, x_indices_validation=x_indices_validation)
            active_learner.ignore_sample_at(x_index_to_ignore, retrain=retrain)

        retrain_mock.assert_called_with(x_indices_validation=x_indices_validation)
        assert_array_equal(np.array(x_index_to_ignore), active_learner.x_indices_ignored)

        # whenever no labels change only one _retrain call is expected (regardless of the retrain kwarg)
        self.assertEqual(1, retrain_mock.call_count)
        retrain_mock.assert_has_calls([call(x_indices_validation=x_indices_validation)])

    def _test_ignore_sample_with_label_change(self, x_indices_initial, y_initial, x_index_to_ignore,
                                                 num_samples=100, retrain=False, x_indices_validation=None):
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        x = np.random.rand(num_samples, 10)
        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)
            active_learner.initialize_data(x_indices_initial, y_initial, x_indices_validation=x_indices_validation)
            self.assertEqual(x_indices_initial.shape[0], active_learner.x_indices_labeled.shape[0])
            self.assertEqual(x_indices_initial.shape[0], active_learner.y.shape[0])
            assert_array_equal(np.array([]), active_learner.x_indices_ignored)

            active_learner.ignore_sample_at(x_index_to_ignore, retrain=retrain,
                                            x_indices_validation=x_indices_validation)

        self.assertEqual(x_indices_initial.shape[0]-1, active_learner.x_indices_labeled.shape[0])
        self.assertEqual(x_indices_initial.shape[0] - 1, active_learner.y.shape[0])

        assert_array_equal(np.array(x_index_to_ignore), active_learner.x_indices_ignored)

        if not retrain:
            self.assertEqual(1, retrain_mock.call_count)
            retrain_mock.assert_has_calls([call(x_indices_validation=x_indices_validation)])
        else:
            self.assertEqual(2, retrain_mock.call_count)
            retrain_mock.assert_has_calls([call(x_indices_validation=x_indices_validation),
                                           call(x_indices_validation=x_indices_validation)])
