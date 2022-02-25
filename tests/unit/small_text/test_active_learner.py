import unittest
import numpy as np

from unittest.mock import call, patch, Mock, ANY
from numpy.testing import assert_array_equal
from scipy import sparse
from scipy.sparse import csr_matrix, vstack

from small_text.active_learner import (
    AbstractPoolBasedActiveLearner,
    ActiveLearner,
    PoolBasedActiveLearner
)
from small_text.base import LABEL_IGNORED
from small_text.exceptions import LearnerNotInitializedException
from small_text.classifiers import ConfidenceEnhancedLinearSVC
from small_text.classifiers.factories import SklearnClassifierFactory
from small_text.data import SklearnDataset
from small_text.query_strategies import RandomSampling

from tests.utils.datasets import random_matrix_data
from tests.utils.testing import assert_csr_matrix_equal, assert_labels_equal


class ActiveLearnerTest(unittest.TestCase):

    def test_abstract_active_learner(self):
        with self.assertRaises(TypeError):
            ActiveLearner()


class AbstractPoolBasedActiveLearnerTest(unittest.TestCase):

    def test_abstract_active_learner(self):
        with self.assertRaises(TypeError):
            AbstractPoolBasedActiveLearner()


class _PoolBasedActiveLearnerTest(object):

    NUM_CLASSES = 5

    NUM_SAMPLES = 100

    def _get_classifier_factory(self):
        return SklearnClassifierFactory(ConfidenceEnhancedLinearSVC(),
                                        self.NUM_CLASSES,
                                        kwargs=dict({'multi_label': self.multi_label}))

    def _set_up_active_learner(self):
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        x = self._get_random_dataset()

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)
        x_indices_initial = np.random.choice(np.arange(100), size=10)
        y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        active_learner.initialize_data(x_indices_initial, y_initial)

        return active_learner

    def _get_random_dataset(self, num_samples=100):
        if self.multi_label:
            return SklearnDataset(*random_matrix_data('dense', 'sparse', self.NUM_SAMPLES))
        else:
            return SklearnDataset(*random_matrix_data('dense', 'dense', self.NUM_SAMPLES))

    def _get_random_labels(self, num_samples=100, num_labels=2):
        if self.multi_label:
            return sparse.random(num_samples, num_labels, density=0.5, format='csr')
        else:
            return np.random.randint(0, high=self.NUM_CLASSES, size=num_samples)

    def _get_y_initial(self):
        if self.multi_label:
            return csr_matrix(np.array([
                [0, 0], [0, 1], [1, 0], [1, 0], [0, 1],
                [1, 1], [0, 0], [1, 0], [0, 0], [1, 0]
            ]))
        else:
            return np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    def _get_y_new(self):
        if self.multi_label:
            _, y_new = random_matrix_data('dense', 'sparse', 5)
        else:
            y_new = np.random.randint(0, 1, size=5)
        return y_new

    def test_init(self):

        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()

        x = self._get_random_dataset()

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x)

        self.assertIsNone(active_learner.classifier)
        self.assertEqual(query_strategy, active_learner.query_strategy)
        self.assertEqual(clf_factory, active_learner._clf_factory)
        self.assertIsNone(active_learner.y)
        self.assertFalse(active_learner.reuse_model)

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
        self.assertIsNone(active_learner._index_to_position)
        assert_array_equal(np.empty(0, dtype=np.int64), active_learner.indices_labeled)
        assert_array_equal(np.empty(0, dtype=np.int64), active_learner.y)

        y_initial = self._get_y_initial()
        if x_indices_validation == 'random':
            x_indices_labeled_indices = list(range(len(x_indices_initial)))
            x_indices_validation = np.random.choice(x_indices_labeled_indices, size=1, replace=False)

        active_learner.initialize_data(x_indices_initial, y_initial, retrain=retrain,
                                       indices_ignored=x_indices_ignored,
                                       indices_validation=x_indices_validation)

        self.assertIsNotNone(active_learner._index_to_position)
        if self.multi_label:
            assert_csr_matrix_equal(y_initial, active_learner.y)
        else:
            assert_array_equal(y_initial, active_learner.y)
        assert_array_equal(x_indices_initial, active_learner.indices_labeled)
        if x_indices_ignored is not None:
            assert_array_equal(x_indices_ignored, active_learner.indices_ignored)
        else:
            assert_array_equal(np.array([]), active_learner.indices_ignored)

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
        ds = self._get_random_dataset()

        with self.assertRaises(LearnerNotInitializedException):
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy_mock, ds)
            active_learner.query(num_samples=5)

    def test_query(self, num_samples=5):

        clf_factory = self._get_classifier_factory()
        query_strategy_mock = Mock()
        ds = self._get_random_dataset()

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy_mock, ds)

        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = ds.y[x_indices_initial]
        active_learner.initialize_data(x_indices_initial, y_initial)

        active_learner.query(num_samples=num_samples)

        query_strategy_mock.query.assert_called_with(ANY, ds, ANY, x_indices_initial, y_initial,
                                                     n=num_samples)

    def test_query_with_custom_representation(self, num_samples=5):

        clf_factory = self._get_classifier_factory()
        query_strategy_mock = Mock()
        ds = self._get_random_dataset()

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy_mock, ds)

        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = ds.y[x_indices_initial]
        active_learner.initialize_data(x_indices_initial, y_initial)

        custom_x = np.random.rand(100, 3)
        active_learner.query(num_samples=num_samples, representation=custom_x)

        query_strategy_mock.query.assert_called_with(ANY, custom_x, ANY, x_indices_initial, y_initial,
                                                     n=num_samples)

    def test_query_with_custom_representation_invalid(self, num_samples=5):

        clf_factory = self._get_classifier_factory()
        query_strategy_mock = Mock()
        ds = self._get_random_dataset()

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy_mock, ds)

        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = ds.y[x_indices_initial]
        active_learner.initialize_data(x_indices_initial, y_initial)

        custom_x = np.random.rand(98, 3)  # <-- representation's first dimension differs from x
        assert custom_x.shape[0] != len(ds)

        with self.assertRaises(ValueError):
            active_learner.query(num_samples=num_samples, representation=custom_x)

    def test_query_with_kwargs(self):

        num_samples = 5
        kwargs = {'argx_x': 1, 'arg_y': 2}

        clf_factory = self._get_classifier_factory()
        query_strategy_mock = Mock()
        ds = self._get_random_dataset()

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy_mock, ds)

        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = ds.y[x_indices_initial]
        active_learner.initialize_data(x_indices_initial, y_initial)

        active_learner.query(num_samples=num_samples, query_strategy_kwargs=kwargs)

        query_strategy_mock.query.assert_called_with(ANY, ds, ANY, x_indices_initial, y_initial,
                                                     n=num_samples, **kwargs)

    def test_update(self):
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        dataset = self._get_random_dataset()

        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, dataset)

            indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
            y_initial = self._get_y_initial()
            active_learner.initialize_data(indices_initial, y_initial)

            assert_array_equal(indices_initial, active_learner.indices_labeled)
            if self.multi_label:
                assert_csr_matrix_equal(y_initial, active_learner.y)
            else:
                assert_array_equal(y_initial, active_learner.y)

            indices_new = active_learner.query(num_samples=5)
            y_new = self._get_y_new()
            active_learner.update(y_new)

            assert_array_equal(np.concatenate([indices_initial, indices_new]),
                               active_learner.indices_labeled)
            if self.multi_label:
                assert_csr_matrix_equal(vstack([y_initial, y_new]), active_learner.y)
            else:
                assert_array_equal(np.concatenate([y_initial, y_new]), active_learner.y)

        retrain_mock.assert_called_with(indices_validation=None)

    def test_update_with_ignored_samples(self):
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        dataset = self._get_random_dataset()

        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, dataset)

            indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
            y_initial = self._get_y_initial()
            active_learner.initialize_data(indices_initial, y_initial)

            assert_array_equal(indices_initial, active_learner.indices_labeled)
            if self.multi_label:
                assert_csr_matrix_equal(y_initial, active_learner.y)
            else:
                assert_array_equal(y_initial, active_learner.y)

            indices_new = active_learner.query(num_samples=5)
            if self.multi_label:
                y_new = np.array([
                    [LABEL_IGNORED, LABEL_IGNORED],
                    [0, 1],
                    [LABEL_IGNORED, LABEL_IGNORED],
                    [0, 0],
                    [LABEL_IGNORED, LABEL_IGNORED],
                ])
                y_new = csr_matrix(y_new)
            else:
                y_new = np.array([LABEL_IGNORED, 1, LABEL_IGNORED, 0, LABEL_IGNORED])  # ignore samples 0, 2, and 4
            active_learner.update(y_new)
            assert_array_equal(np.concatenate([indices_initial, [indices_new[1], indices_new[3]]]),
                               active_learner.indices_labeled)

            if self.multi_label:
                assert_csr_matrix_equal(vstack([y_initial, y_new[[1, 3], :]]), active_learner.y)
            else:
                assert_array_equal(np.concatenate([y_initial, y_new[[1, 3]]]), active_learner.y)

        retrain_mock.assert_called_with(indices_validation=None)

    def test_update_with_only_ignored_samples(self, query_size=5):
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        dataset = self._get_random_dataset()

        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, dataset)

            indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
            y_initial = dataset.y[indices_initial]
            active_learner.initialize_data(indices_initial, y_initial)

            assert_array_equal(indices_initial, active_learner.indices_labeled)
            if self.multi_label:
                assert_csr_matrix_equal(y_initial, active_learner.y)
            else:
                assert_array_equal(y_initial, active_learner.y)

            active_learner.query(num_samples=query_size)
            y_new = np.array([LABEL_IGNORED]*query_size)  # ignore all samples
            active_learner.update(y_new)

            assert_array_equal(indices_initial, active_learner.indices_labeled)
            assert_labels_equal(y_initial, active_learner.y)

        retrain_mock.assert_has_calls([call(indices_validation=None)])

    def test_update_with_validation_set_given(self):
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        dataset = self._get_random_dataset()

        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, dataset)

            indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
            y_initial = self._get_y_initial()
            indices_validation = np.random.choice(indices_initial, size=10, replace=False)

            active_learner.initialize_data(indices_initial, y_initial,
                                           indices_validation=indices_validation)

            assert_array_equal(indices_initial, active_learner.indices_labeled)
            if self.multi_label:
                assert_csr_matrix_equal(y_initial, active_learner.y)
            else:
                assert_array_equal(y_initial, active_learner.y)

            indices_new = active_learner.query(num_samples=5)
            y_new = self._get_y_new()
            active_learner.update(y_new, indices_validation=indices_validation)
            assert_array_equal(np.concatenate([indices_initial, indices_new]),
                               active_learner.indices_labeled)
            if self.multi_label:
                assert_csr_matrix_equal(vstack([y_initial, y_new]), active_learner.y)
            else:
                assert_array_equal(np.concatenate([y_initial, y_new]), active_learner.y)

        retrain_mock.assert_has_calls([call(indices_validation=indices_validation),
                                       call(indices_validation=indices_validation)])

    def test_update_reuse_model(self):
        query_strategy = RandomSampling()
        dataset = self._get_random_dataset()

        clf_mock = ConfidenceEnhancedLinearSVC()
        clf_mock.fit = Mock()
        clf_factory_mock = self._get_classifier_factory()
        clf_factory_mock.new = Mock()
        clf_factory_mock.new.side_effect = [clf_mock, None]

        active_learner = PoolBasedActiveLearner(clf_factory_mock, query_strategy, dataset,
                                                reuse_model=True)

        indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = self._get_y_initial()
        active_learner.initialize_data(indices_initial, y_initial)

        assert_array_equal(indices_initial, active_learner.indices_labeled)
        if self.multi_label:
            assert_csr_matrix_equal(y_initial, active_learner.y)
        else:
            assert_array_equal(y_initial, active_learner.y)

        for _ in range(2):
            active_learner.query(num_samples=5)
            y_new = self._get_y_new()
            active_learner.update(y_new)

        clf_factory_mock.new.assert_called_once_with()
        self.assertEqual(active_learner._clf, clf_mock)
        active_learner._clf.fit.assert_called()

    def test_query_update_mismatch(self):
        active_learner = self._set_up_active_learner()

        indices = active_learner.query(num_samples=5)
        self.assertEqual(5, len(indices))
        if self.multi_label:
            _, y_new = random_matrix_data('dense', 'sparse', 6)
        else:
            y_new = np.random.randint(0, 1, size=6)

        with self.assertRaises(ValueError):
            active_learner.update(y_new)

    def test_update_duplicate_indices_in_labeled_pool(self):
        query_strategy = RandomSampling()
        dataset = self._get_random_dataset()

        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC(), self.NUM_CLASSES)

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, dataset)
        active_learner._retrain = Mock()

        indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = self._get_y_initial()

        active_learner.initialize_data(indices_initial, y_initial)

        with self.assertRaises(ValueError):
            y_new = np.random.randint(0, 1, size=5)

            active_learner.query(num_samples=5)
            active_learner.indices_queried = indices_initial
            active_learner.update(y_new)

        active_learner._retrain.assert_called_once()

    def test_update_label_at(self):
        self._test_update_label_at()

    def test_update_label_at_with_retrain_true(self):
        self._test_update_label_at(retrain=True)

    def test_update_label_at_with_validation_set_given(self):
        self._test_update_label_at(indices_validation='random')

    def _test_update_label_at(self, retrain=False, indices_validation=None):
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        dataset = self._get_random_dataset()

        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, dataset)

            indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
            y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            active_learner.initialize_data(indices_initial, y_initial)

            # query & update
            indices_new = active_learner.query(num_samples=5)
            y_new = np.array([0, 0, 0, 1, 1])
            if indices_validation == 'random':
                indices_validation = np.random.choice(indices_initial, size=10, replace=False)
            active_learner.update(y_new, indices_validation=indices_validation)

            active_learner.update_label_at(indices_new[0], 1, retrain=retrain)
            assert_array_equal([1, 0, 0, 1, 1], active_learner.y[-5:])
        if retrain:
            retrain_mock.assert_has_calls([call(indices_validation=indices_validation),
                                           call(indices_validation=indices_validation)])

    def test_remove_label_at(self):
        self._test_remove_label_at(index_to_remove=-2)
        self._test_remove_label_at(index_to_remove=-1)

    def test_remove_label_at_with_validation_set_given(self):
        self._test_remove_label_at(index_to_remove=-1, retrain=True, indices_validation='random')

    def _test_remove_label_at(self, index_to_remove=-5, retrain=False, indices_validation=None):
        dataset = self._get_random_dataset(num_samples=100)
        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC(), self.NUM_CLASSES)
        query_strategy = RandomSampling()
        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, dataset)

            indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
            y_initial = dataset.y[indices_initial]
            active_learner.initialize_data(indices_initial, y_initial)

            # query & update
            indices = active_learner.query(num_samples=5)
            y_new = dataset.y[indices]
            active_learner.update(y_new)

            if indices_validation == 'random':
                indices_validation = np.random.choice(indices_initial, size=3, replace=False)

            active_learner.remove_label_at(active_learner.indices_labeled[index_to_remove],
                                           retrain=retrain, x_indices_validation=indices_validation)

            if isinstance(y_new, csr_matrix):
                assert_csr_matrix_equal(dataset.y[active_learner.indices_labeled[-4:]], active_learner.y[-4:])
                assert_csr_matrix_equal(dataset.y[active_learner.indices_labeled[:11]], active_learner.y[:11])
            else:
                assert_array_equal(dataset.y[active_learner.indices_labeled[-4:]], active_learner.y[-4:])
                assert_array_equal(dataset.y[active_learner.indices_labeled[:11]], active_learner.y[:11])

            self.assertEqual(14, active_learner.y.shape[0])
            self.assertEqual(14, active_learner.indices_labeled.shape[0])

        retrain_mock.assert_called_with(indices_validation=indices_validation)

    def test_ignore_sample_at(self):
        ds = self._get_random_dataset(num_samples=100)
        indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = self._get_y_initial()
        unlabeled_indices = self._get_unlabeled_indices(indices_initial)
        self._test_ignore_sample_without_label_change(ds, indices_initial, y_initial, unlabeled_indices[0])

    def test_ignore_sample_at_with_retrain_true(self):
        ds = self._get_random_dataset(num_samples=100)
        indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = self._get_y_initial()
        unlabeled_indices = self._get_unlabeled_indices(indices_initial)
        self._test_ignore_sample_without_label_change(ds,
                                                      indices_initial,
                                                      y_initial,
                                                      unlabeled_indices[0],
                                                      retrain=True)

    def test_ignore_sample_at_with_validation_set_given(self):
        ds = self._get_random_dataset(num_samples=100)
        indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = self._get_y_initial()
        unlabeled_indices = self._get_unlabeled_indices(indices_initial)
        x_indices_validation = np.random.choice(indices_initial, size=3, replace=False)
        self._test_ignore_sample_without_label_change(ds, indices_initial, y_initial, unlabeled_indices[0],
                                                      indices_validation=x_indices_validation)

    def test_ignore_sample_at_with_label_removed(self):
        ds = self._get_random_dataset(num_samples=100)
        indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = self._get_y_initial()
        self._test_ignore_sample_with_label_change(ds, indices_initial, y_initial, indices_initial[0])

    def test_ignore_sample_at_with_label_removed_and_retrain_true(self):
        ds = self._get_random_dataset(num_samples=100)
        x_indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = self._get_y_initial()
        self._test_ignore_sample_with_label_change(ds, x_indices_initial, y_initial, x_indices_initial[0], retrain=True)

    def test_ignore_sample_at_with_label_removed_and_retrain_true_and_validation_set_given(self):
        ds = self._get_random_dataset(num_samples=100)
        indices_initial = np.random.choice(np.arange(100), size=10, replace=False)
        y_initial = ds.y[indices_initial]
        self._test_ignore_sample_with_label_change(ds, indices_initial, y_initial, indices_initial[0],
                                                   retrain=True, indices_validation=indices_initial[1:3])

    def _get_unlabeled_indices(self, indices_initial, num_samples=100):

        mask = np.ones(num_samples, bool)
        mask[indices_initial] = False
        indices = np.arange(num_samples)

        return indices[mask]

    def _test_ignore_sample_without_label_change(self, dataset, indices_initial, y_initial, index_to_ignore,
                                                 retrain=False, indices_validation=None):
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, dataset)

            active_learner.initialize_data(indices_initial, y_initial, indices_validation=indices_validation)
            active_learner.ignore_sample_at(index_to_ignore, retrain=retrain)

        retrain_mock.assert_called_with(indices_validation=indices_validation)
        assert_array_equal(np.array(index_to_ignore), active_learner.indices_ignored)

        # whenever no labels change only one _retrain call is expected (regardless of the retrain kwarg)
        self.assertEqual(1, retrain_mock.call_count)
        retrain_mock.assert_has_calls([call(indices_validation=indices_validation)])

    def _test_ignore_sample_with_label_change(self, dataset, indices_initial, y_initial, index_to_ignore,
                                              retrain=False, indices_validation=None):
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        with patch.object(PoolBasedActiveLearner, '_retrain', return_value=None) as retrain_mock:
            active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, dataset)
            active_learner.initialize_data(indices_initial, y_initial, indices_validation=indices_validation)
            self.assertEqual(indices_initial.shape[0], active_learner.indices_labeled.shape[0])
            self.assertEqual(indices_initial.shape[0], active_learner.y.shape[0])
            assert_array_equal(np.array([]), active_learner.indices_ignored)

            active_learner.ignore_sample_at(index_to_ignore, retrain=retrain,
                                            indices_validation=indices_validation)

        self.assertEqual(indices_initial.shape[0] - 1, active_learner.indices_labeled.shape[0])
        self.assertEqual(indices_initial.shape[0] - 1, active_learner.y.shape[0])

        assert_array_equal(np.array(index_to_ignore), active_learner.indices_ignored)

        if not retrain:
            self.assertEqual(1, retrain_mock.call_count)
            retrain_mock.assert_has_calls([call(indices_validation=indices_validation)])
        else:
            self.assertEqual(2, retrain_mock.call_count)
            retrain_mock.assert_has_calls([call(indices_validation=indices_validation),
                                           call(indices_validation=indices_validation)])


class PoolBasedActiveLearnerSingleLabelTest(unittest.TestCase, _PoolBasedActiveLearnerTest):

    def setUp(self):
        self.multi_label = False

    def _set_up_active_learner(self):
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        ds = self._get_random_dataset(num_samples=self.NUM_SAMPLES)
        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, ds)
        x_indices_initial = np.random.choice(np.arange(100), size=10)
        y_initial = ds.y[x_indices_initial]
        active_learner.initialize_data(x_indices_initial, y_initial)

        return active_learner


class PoolBasedActiveLearnerMultiLabelTest(unittest.TestCase, _PoolBasedActiveLearnerTest):

    def setUp(self):
        self.multi_label = True

    def _set_up_active_learner(self):
        clf_factory = self._get_classifier_factory()
        query_strategy = RandomSampling()
        ds = SklearnDataset(*random_matrix_data('dense', 'sparse', self.NUM_SAMPLES))
        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, ds)
        x_indices_initial = np.random.choice(np.arange(100), size=10)
        y_initial = ds.y[x_indices_initial]
        active_learner.initialize_data(x_indices_initial, y_initial)

        return active_learner
