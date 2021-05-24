import unittest
import tempfile

import pytest

import numpy as np

from numpy.testing import assert_array_equal

from active_learning.active_learner import PoolBasedActiveLearner
from active_learning.integrations.pytorch.exceptions import PytorchNotFoundError
from active_learning.query_strategies import RandomSampling

from tests.utils.datasets import trec_dataset
from tests.utils.object_factory import get_initialized_active_learner

try:
    import torch
    from active_learning.integrations.pytorch.classifiers import PytorchClassifier, KimCNNFactory
    from active_learning.integrations.pytorch.datasets import PytorchTextClassificationDataset
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class SerializationTest(unittest.TestCase):

    def test_and_load_with_file_str(self):
        dataset, _ = trec_dataset()
        self.assertFalse(dataset._data[0][PytorchTextClassificationDataset.INDEX_TEXT].is_cuda)

        clf_factory = KimCNNFactory('kimcnn',
                                    {'embedding_matrix': torch.rand(len(dataset.vocab), 100)})
        query_strategy = RandomSampling()

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            file_str = tmp_dir_name + 'active_learner.ser'

            active_learner = get_initialized_active_learner(clf_factory, query_strategy, dataset)
            ind_initial = active_learner.x_indices_labeled
            ind = active_learner.query()

            # TODO: reconsider if this makes sense
            #self.assertTrue(next(active_learner.classifier.model.parameters()).is_cuda)

            active_learner.update(np.random.randint(2, size=10))
            weights_before = list(active_learner.classifier.model.parameters())

            active_learner.save(file_str)
            del active_learner

            active_learner = PoolBasedActiveLearner.load(file_str)
            self.assertIsNotNone(active_learner)
            assert_array_equal(np.concatenate([ind_initial, ind]), active_learner.x_indices_labeled)

            weights_after = list(active_learner.classifier.model.parameters())
            self.assertEqual(len(weights_before), len(weights_after))
            for i in range(len(weights_before)):
                assert_array_equal(weights_before[i].cpu().detach().numpy(), weights_after[i].cpu().detach().numpy())

            self.assertIsNotNone(active_learner.classifier)
            self.assertEqual(query_strategy.__class__, active_learner.query_strategy.__class__)
