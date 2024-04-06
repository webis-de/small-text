import unittest

import pytest

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

from tests.integration.small_text.query_strategies.test_query_strategies import (
    QueryStrategiesExhaustiveIntegrationTest
)

try:
    import torch
    from small_text.integrations.pytorch.classifiers.base import AMPArguments
    from small_text.integrations.pytorch.query_strategies import (
        DiscriminativeRepresentationLearning,
    )

    from small_text.integrations.transformers import TransformerModelArguments
    from small_text.integrations.transformers.classifiers.classification import TransformerBasedClassification
    from small_text.integrations.transformers.classifiers.factories import TransformerBasedClassificationFactory

    from tests.utils.datasets import random_transformer_dataset
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class QueryStrategiesTest(QueryStrategiesExhaustiveIntegrationTest, unittest.TestCase):

    def _get_dataset(self, num_classes, multi_label=False):
        return random_transformer_dataset(num_samples=60, max_length=10,
                                          multi_label=multi_label, num_classes=num_classes)

    def _get_factory(self, num_classes, multi_label=False):
        transformer_model = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        return TransformerBasedClassificationFactory(transformer_model, 6)

    def test_discriminative_representation_learning(self):
        query_strategy = DiscriminativeRepresentationLearning()
        self._simple_exhaustive_active_learning_test(query_strategy)

    def test_discriminative_representation_learning_amp(self):
        query_strategy = DiscriminativeRepresentationLearning(amp_args=AMPArguments(use_amp=True, device_type='cuda'),
                                                              device='cuda')
        self._simple_exhaustive_active_learning_test(query_strategy)
