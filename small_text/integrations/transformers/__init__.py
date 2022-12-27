from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from small_text.integrations.transformers.classifiers.base import ModelLoadingStrategy
    from small_text.integrations.transformers.classifiers.classification import (
        transformers_collate_fn,
        FineTuningArguments,
        TransformerModelArguments,
        TransformerBasedClassification,
        TransformerBasedEmbeddingMixin
    )
    from small_text.integrations.transformers.classifiers.setfit import (
        SetFitClassification,
        SetFitModelArguments,
        SetFitClassificationEmbeddingMixin
    )
    from small_text.integrations.transformers.classifiers.factories import (
        SetFitClassificationFactory,
        TransformerBasedClassificationFactory
    )
    from small_text.integrations.transformers.datasets import (
        TransformersDataset,
        TransformersDatasetView
    )
    __all__ = [
        'ModelLoadingStrategy',
        'transformers_collate_fn',
        'FineTuningArguments',
        'TransformerModelArguments',
        'TransformerBasedClassification',
        'TransformerBasedEmbeddingMixin',
        'SetFitClassification',
        'SetFitModelArguments',
        'SetFitClassificationEmbeddingMixin',
        'TransformerBasedClassificationFactory',
        'SetFitClassificationFactory',
        'TransformersDataset',
        'TransformersDatasetView'
    ]
except PytorchNotFoundError:
    __all__ = []
