import unittest

import pytest

import numpy as np

from active_learning.integrations.pytorch.exceptions import PytorchNotFoundError

from tests.utils.datasets import trec_dataset
from tests.utils.object_factory import get_initialized_active_learner

try:
    import torch
    from active_learning.integrations.pytorch.classifiers import PytorchClassifier, KimCNNFactory
    from active_learning.integrations.pytorch.datasets import PytorchTextClassificationDataset
    from active_learning.integrations.pytorch.query_strategies import ExpectedGradientLength, \
        ExpectedGradientLengthMaxWord, ExpectedGradientLengthLayer, BADGE
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class QueryStrategiesTest(unittest.TestCase):

    def test_expected_gradient_length(self):
        query_strategy = ExpectedGradientLength(2)
        self._simple_active_learning_test(query_strategy)

    def test_expected_gradient_length_max_word(self):
        query_strategy = ExpectedGradientLengthMaxWord(2, 'embedding')
        self._simple_active_learning_test(query_strategy)

    def test_expected_gradient_length_layer(self):
        query_strategy = ExpectedGradientLengthLayer(2, 'fc')
        self._simple_active_learning_test(query_strategy)

    def test_badge(self):
        query_strategy = BADGE()
        self._simple_active_learning_test(query_strategy)

    def _simple_active_learning_test(self, query_strategy, query_size=10, num_initial=10):
        dataset, _ = trec_dataset()
        dataset = dataset[0:50]

        self.assertFalse(dataset[0].x[PytorchTextClassificationDataset.INDEX_TEXT].is_cuda)
        clf_factory = KimCNNFactory('kimcnn',
                                    {'embedding_matrix': torch.rand(len(dataset.vocab), 100)})

        active_learner = get_initialized_active_learner(clf_factory, query_strategy, dataset)

        for _ in range(2):
            active_learner.query()
            active_learner.update(np.random.randint(2, size=query_size))

        self.assertEqual(query_size*2 + num_initial, active_learner.x_indices_labeled.shape[0])
