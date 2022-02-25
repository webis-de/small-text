import unittest

import pytest

import numpy as np

from unittest import mock
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

from tests.utils.object_factory import get_initialized_active_learner

try:
    import torch
    from small_text.integrations.pytorch.classifiers import KimCNNFactory
    from small_text.integrations.pytorch.classifiers.factories import KimCNNClassifier
    from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset
    from small_text.integrations.pytorch.query_strategies import ExpectedGradientLength, \
        ExpectedGradientLengthMaxWord, ExpectedGradientLengthLayer, BADGE

    from tests.utils.datasets import random_text_classification_dataset
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class QueryStrategiesTest(unittest.TestCase):

    def test_expected_gradient_length(self):
        query_strategy = ExpectedGradientLength(2)
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=2)

    def test_expected_gradient_length_max_word(self):
        query_strategy = ExpectedGradientLengthMaxWord(2, 'embedding')
        self._simple_exhaustive_active_learning_test(query_strategy)

    def test_expected_gradient_length_layer(self):
        query_strategy = ExpectedGradientLengthLayer(2, 'fc')
        self._simple_exhaustive_active_learning_test(query_strategy)

    def test_badge(self):
        query_strategy = BADGE(2)
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=2)

    def test_badge_multiclass(self):

        query_strategy = BADGE(6)
        self._simple_exhaustive_active_learning_test(query_strategy)

    def test_badge_with_classifier_that_does_not_return_embeddings_proba(self):
        # fake_embed return random embeddings and does not return probabilities
        def fake_embed(data_set, module_selector=lambda x: x['fc'], pbar='tqdm'):
            return np.random.rand(len(data_set), 10)

        query_strategy = BADGE(6)
        with mock.patch.object(KimCNNClassifier,
                               'embed',
                               wraps=fake_embed):

            self._simple_exhaustive_active_learning_test(query_strategy)

    def _simple_exhaustive_active_learning_test(self, query_strategy, query_size=10,
                                                num_classes=6, num_initial=30):
        dataset = random_text_classification_dataset(num_samples=200, max_length=10, num_classes=num_classes)

        self.assertFalse(dataset[0].x[PytorchTextClassificationDataset.INDEX_TEXT].is_cuda)
        clf_factory = KimCNNFactory('kimcnn',
                                    num_classes,
                                    {'embedding_matrix': torch.rand(len(dataset.vocab), 20),
                                     'num_epochs': 2})

        active_learner = get_initialized_active_learner(clf_factory, query_strategy, dataset,
                                                        initial_indices=num_initial)

        for _ in range(3):
            active_learner.query()
            active_learner.update(np.random.randint(2, size=query_size))

        self.assertEqual(query_size * 3 + num_initial, active_learner.indices_labeled.shape[0])
