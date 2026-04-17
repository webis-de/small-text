import json
import unittest
import tempfile

import numpy as np

from pathlib import Path
from numpy.testing import assert_array_equal

from small_text.active_learner import ACTIVE_LEARNER_CONFIG_FILE, PoolBasedActiveLearner
from small_text.classifiers import ConfidenceEnhancedLinearSVC, SklearnClassifierFactory
from small_text.exceptions import SerializationException
from small_text.query_strategies import RandomSampling

from tests.utils.datasets import random_sklearn_dataset
from tests.utils.object_factory import get_initialized_active_learner


class SerializationTest(unittest.TestCase):

    def test_save_and_load_with_folder_str(self):
        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC(), 2)
        query_strategy = RandomSampling()

        with tempfile.TemporaryDirectory() as folder:
            active_learner, ind_initial, ind_queried = self._write(folder, query_strategy, clf_factory)
            self._load(folder, query_strategy, ind_initial, ind_queried)

    def test_save_and_load_with_path(self):
        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC(), 2)
        query_strategy = RandomSampling()

        with tempfile.TemporaryDirectory() as folder:
            active_learner, ind_initial, ind_queried = self._write(Path(folder), query_strategy, clf_factory)
            self._load(folder, query_strategy, ind_initial, ind_queried)

    def test_save_and_load_with_path_that_does_nox_exist(self):
        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC(), 2)
        query_strategy = RandomSampling()

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder) / 'non-existent-subfolder'
            with self.assertRaises(FileNotFoundError):
                self._write(Path(folder), query_strategy, clf_factory)

    def test_save_and_load_with_path_that_does_nox_exist_and_create_folder_true(self):
        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC(), 2)
        query_strategy = RandomSampling()

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder) / 'non-existent-subfolder'
            active_learner, ind_initial, ind_queried = self._write(Path(folder),
                                                                   query_strategy,
                                                                   clf_factory,
                                                                   create_folder=True)
            self._load(folder, query_strategy, ind_initial, ind_queried)

    def test_save_and_load_with_missing_key(self):
        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC(), 2)
        query_strategy = RandomSampling()

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            active_learner, ind_initial, ind_queried =  self._write(folder, query_strategy, clf_factory)

            config_file = folder / ACTIVE_LEARNER_CONFIG_FILE
            with open(config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            del config_dict['small_text_version']

            with open(config_file, 'w+', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4, sort_keys=True)

            with self.assertRaisesRegex(SerializationException,
                                        f'Invalid {ACTIVE_LEARNER_CONFIG_FILE} file'):
                self._load(folder, query_strategy, ind_initial, ind_queried)

    def test_save_and_load_with_mismatched_version(self):
        clf_factory = SklearnClassifierFactory(ConfidenceEnhancedLinearSVC(), 2)
        query_strategy = RandomSampling()

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            active_learner, ind_initial, ind_queried =  self._write(folder, query_strategy, clf_factory)

            config_file = folder / ACTIVE_LEARNER_CONFIG_FILE
            with open(config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            config_dict['small_text_version'] = '100.0.0'

            with open(config_file, 'w+', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4, sort_keys=True)

            with self.assertRaisesRegex(SerializationException, 'Version mismatch:'):
                self._load(folder, query_strategy, ind_initial, ind_queried)

    def _write(self, folder, query_strategy, clf_factory, create_folder=False):

        x = random_sklearn_dataset(20, 100)
        active_learner = get_initialized_active_learner(clf_factory, query_strategy, x)
        ind_initial = active_learner.indices_labeled
        ind_queried = active_learner.query()

        labels = np.random.randint(2, size=10)
        # trigger an update to guarantee the existence of all fields
        labels[0:2] = [0, 1]
        active_learner.update(labels)

        active_learner.save(folder, create_folder=create_folder)

        return active_learner, ind_initial, ind_queried

    def _load(self, folder, query_strategy, ind_initial, ind_queried):
        active_learner = PoolBasedActiveLearner.load(folder)
        self.assertIsNotNone(active_learner)

        assert_array_equal(np.concatenate([ind_initial, ind_queried]), active_learner.indices_labeled)
        self.assertIsNotNone(active_learner.classifier)
        self.assertEqual(query_strategy.__class__, active_learner.query_strategy.__class__)
