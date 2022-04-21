import unittest
import pytest

import numpy as np

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.query_strategies import DiscriminativeActiveLearning

try:
    from small_text.integrations.transformers import TransformerModelArguments
    from small_text.integrations.transformers.classifiers import (
        TransformerBasedClassificationFactory,
        TransformerBasedClassification,
    )

    from tests.utils.datasets import random_transformer_dataset
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class DiscriminativeActiveLearningTest(unittest.TestCase):

    def test_query_with_transformer_models(self):
        transformer_model = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        strategy = DiscriminativeActiveLearning(
            TransformerBasedClassificationFactory(transformer_model, 2),
            3
        )

        clf = TransformerBasedClassification(transformer_model, 2)

        dataset = random_transformer_dataset(100)

        indices_labeled = np.random.choice(np.arange(10), size=10, replace=False)
        indices_unlabeled = np.array(
            [i for i in np.arange(100) if i not in set(indices_labeled)])
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        indices = strategy.query(clf, dataset, indices_unlabeled, indices_labeled, y)
        self.assertEqual(10, indices.shape[0])
        self.assertEqual(10, np.unique(indices).shape[0])
