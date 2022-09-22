from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from small_text.integrations.pytorch.classifiers.base import (
        PytorchModelSelectionMixin,
        PytorchClassifier
    )
    from small_text.integrations.pytorch.classifiers.factories import (
        AbstractClassifierFactory,
        KimCNNFactory
    )
    from small_text.integrations.pytorch.classifiers.kimcnn import (
        kimcnn_collate_fn,
        KimCNNEmbeddingMixin,
        KimCNNClassifier
    )
    from small_text.integrations.pytorch.datasets import (
        PytorchDataset,
        PytorchDatasetView,
        PytorchTextClassificationDataset,
        PytorchTextClassificationDatasetView
    )
    from small_text.integrations.pytorch.models.kimcnn import KimCNN
    from small_text.integrations.pytorch.query_strategies.strategies import (
        ExpectedGradientLength,
        ExpectedGradientLengthLayer,
        ExpectedGradientLengthMaxWord
    )

    __all__ = [
        'PytorchModelSelectionMixin',
        'PytorchClassifier',
        'AbstractClassifierFactory',
        'KimCNNFactory',
        'KimCNN',
        'kimcnn_collate_fn',
        'KimCNNEmbeddingMixin',
        'KimCNNClassifier',
        'ExpectedGradientLength',
        'ExpectedGradientLengthLayer',
        'ExpectedGradientLengthMaxWord',
        'PytorchNotFoundError',
        'PytorchDataset',
        'PytorchDatasetView',
        'PytorchTextClassificationDataset',
        'PytorchTextClassificationDatasetView'
    ]

except PytorchNotFoundError:
    __all__ = []
