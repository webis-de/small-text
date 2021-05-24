import unittest
import tempfile

import numpy as np

from pathlib import Path
from numpy.testing import assert_array_equal

from active_learning.active_learner import PoolBasedActiveLearner
from active_learning.classifiers import ConfidenceEnhancedLinearSVC, SklearnClassifierFactory
from active_learning.query_strategies import RandomSampling

from tests.utils.datasets import random_sklearn_dataset
from tests.utils.object_factory import get_initialized_active_learner


class SerializationTest(unittest.TestCase):

    def test_save_and_load_with_file_str(self):
        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())
        query_strategy = RandomSampling()

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            file_str = tmp_dir_name + '/active_learner.pkl'
            active_learner, ind_initial, ind_queried = self._write(file_str, query_strategy, clf_factory)
            self._load(file_str, query_strategy, ind_initial, ind_queried)

    def test_save_and_load_with_path(self):
        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())
        query_strategy = RandomSampling()

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            file_path = Path(tmp_dir_name + '/active_learner.pkl')
            active_learner, ind_initial, ind_queried = self._write(file_path, query_strategy, clf_factory)
            self._load(file_path, query_strategy, ind_initial, ind_queried)

    def test_save_and_load_with_file_handle(self):
        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC())
        query_strategy = RandomSampling()

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            file_str = tmp_dir_name + '/active_learner.pkl'
            with open(file_str, 'wb') as f:
                active_learner, ind_initial, ind_queried = self._write(f, query_strategy, clf_factory)
            with open(file_str, 'rb') as f:
                self._load(f, query_strategy, ind_initial, ind_queried)

    def _write(self, file, query_strategy, clf_factory):

        x = random_sklearn_dataset(20, 100)
        active_learner = get_initialized_active_learner(clf_factory, query_strategy, x)
        ind_initial = active_learner.x_indices_labeled
        ind_queried = active_learner.query()

        active_learner.update(np.random.randint(2, size=10))
        active_learner.save(file)

        return active_learner, ind_initial, ind_queried

    def _load(self, file, query_strategy, ind_initial, ind_queried):
        active_learner = PoolBasedActiveLearner.load(file)
        self.assertIsNotNone(active_learner)

        assert_array_equal(np.concatenate([ind_initial, ind_queried]), active_learner.x_indices_labeled)
        self.assertIsNotNone(active_learner.classifier)
        self.assertEqual(query_strategy.__class__, active_learner.query_strategy.__class__)
