import unittest

import pytest

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.query_strategies import (
    BreakingTies,
    CategoryVectorInconsistencyAndRanking,
    ContrastiveActiveLearning,
    DiscriminativeActiveLearning,
    EmbeddingKMeans,
    GreedyCoreset,
    LeastConfidence,
    LightweightCoreset,
    SEALS,
    AnchorSubsampling,
    SubsamplingQueryStrategy
)

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
class QueryStrategiesTestTransformer(QueryStrategiesExhaustiveIntegrationTest, unittest.TestCase):

    def _get_dataset(self, num_classes, multi_label=False):
        return random_transformer_dataset(num_samples=60, max_length=10,
                                          multi_label=multi_label, num_classes=num_classes)

    def _get_factory(self, num_classes, multi_label=False):
        transformer_model = TransformerModelArguments('small-text/tiny-distilroberta-base')
        return TransformerBasedClassificationFactory(transformer_model,
                                                     6,
                                                     classification_kwargs={'multi_label': multi_label})

    # --- query strategies from the small-text core functionality  ---

    def test_breaking_ties(self):
        query_strategy = BreakingTies()
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=6)

    def test_category_vector_inconsistency_and_ranking(self):
        query_strategy = CategoryVectorInconsistencyAndRanking()
        self._simple_exhaustive_active_learning_test(query_strategy, multi_label=True,
                                                     num_classes=6)

    def test_contrastive_active_learning(self):
        query_strategy = ContrastiveActiveLearning()
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=6)

    @pytest.mark.skip(reason='not yet supported')
    def test_discriminative_active_learning(self):
        transformer_model = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier_factory = TransformerBasedClassificationFactory(transformer_model, 6)
        query_strategy = DiscriminativeActiveLearning(classifier_factory, 2)
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=6)

    def test_embedding_kmeans(self):
        query_strategy = EmbeddingKMeans()
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=6)

    def test_greedy_coreset(self):
        query_strategy = GreedyCoreset()
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=6)

    def test_lightweight_coreset(self):
        query_strategy = LightweightCoreset()
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=6)

    @pytest.mark.optional
    def test_seals(self):
        query_strategy = SEALS(LeastConfidence(), k=5)
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=6)

    @pytest.mark.optional
    def test_anchor_subsampling(self):
        query_strategy = AnchorSubsampling(LeastConfidence(), k=5)
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=6)

    def test_subsampling_query_strategy(self):
        query_strategy = SubsamplingQueryStrategy(LeastConfidence(), subsample_size=10)
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=6)

    # --- additional query strategies from the integration package ---

    def test_discriminative_representation_learning(self):
        query_strategy = DiscriminativeRepresentationLearning()
        self._simple_exhaustive_active_learning_test(query_strategy)

    def test_discriminative_representation_learning_amp(self):
        query_strategy = DiscriminativeRepresentationLearning(amp_args=AMPArguments(use_amp=True, device_type='cuda'),
                                                              device='cuda')
        self._simple_exhaustive_active_learning_test(query_strategy)
