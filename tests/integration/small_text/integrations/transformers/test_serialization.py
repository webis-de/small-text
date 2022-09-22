import unittest
import tempfile

import pytest

import numpy as np

from numpy.testing import assert_array_equal

from small_text.active_learner import PoolBasedActiveLearner
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.query_strategies import RandomSampling

from small_text.integrations.transformers.datasets import TransformersDataset
from tests.utils.object_factory import get_initialized_active_learner

try:
    from small_text.integrations.transformers import TransformerModelArguments
    from small_text.integrations.transformers.classifiers import TransformerBasedClassificationFactory
    from tests.utils.datasets import twenty_news_transformers
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class SerializationTest(unittest.TestCase):

    def test_and_load_with_file_str(self):
        num_labels = 2

        dataset = twenty_news_transformers(20, num_labels=num_labels)
        self.assertFalse(dataset.x[TransformersDataset.INDEX_TEXT].is_cuda)

        clf_factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
            num_labels,
            kwargs={'device': 'cuda'})

        query_strategy = RandomSampling()

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            file_str = tmp_dir_name + 'active_learner.ser'

            active_learner = get_initialized_active_learner(clf_factory, query_strategy, dataset)
            ind_initial = active_learner.indices_labeled
            ind = active_learner.query(num_samples=5)

            self.assertTrue(next(active_learner.classifier.model.parameters()).is_cuda)

            active_learner.update(np.random.randint(2, size=5))
            weights_before = list(active_learner.classifier.model.parameters())

            active_learner.save(file_str)
            del active_learner

            active_learner = PoolBasedActiveLearner.load(file_str)
            self.assertIsNotNone(active_learner)
            assert_array_equal(np.concatenate([ind_initial, ind]), active_learner.indices_labeled)

            weights_after = list(active_learner.classifier.model.parameters())
            self.assertEqual(len(weights_before), len(weights_after))
            for i in range(len(weights_before)):
                assert_array_equal(weights_before[i].cpu().detach().numpy(), weights_after[i].cpu().detach().numpy())

            self.assertIsNotNone(active_learner.classifier)
            self.assertEqual(query_strategy.__class__, active_learner.query_strategy.__class__)
