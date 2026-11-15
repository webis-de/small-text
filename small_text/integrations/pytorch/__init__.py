from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from small_text.integrations.pytorch.classifiers.base import (
        AMPArguments,
        PytorchModelSelectionMixin,
        PytorchClassifier
    )
    from small_text.integrations.pytorch.classifiers.factories import (
        AbstractClassifierFactory,
        KimCNNClassifierFactory,
        LSTMClassifierFactory
    )
    from small_text.integrations.pytorch.classifiers.kimcnn import (
        kimcnn_collate_fn,
        KimCNNEmbeddingMixin,
        KimCNNClassifier
    )
    from small_text.integrations.pytorch.classifiers.lstm import (
        lstm_collate_fn,
        LSTMEmbeddingMixin,
        LSTMClassifier
    )
    from small_text.integrations.pytorch.datasets import (
        PytorchDataset,
        PytorchDatasetView,
        PytorchTextClassificationDataset,
        PytorchTextClassificationDatasetView
    )
    from small_text.integrations.pytorch.models.kimcnn import KimCNN
    from small_text.integrations.pytorch.models.lstm import LSTM
    from small_text.integrations.pytorch.query_strategies.strategies import (
        DiscriminativeRepresentationLearning,
        ExpectedGradientLength,
        ExpectedGradientLengthLayer,
        ExpectedGradientLengthMaxWord
    )

    __all__ = [
        'AMPArguments',
        'PytorchModelSelectionMixin',
        'PytorchClassifier',
        'AbstractClassifierFactory',
        'KimCNNClassifierFactory',
        'LSTMClassifierFactory',
        'KimCNN',
        'LSTM',
        'kimcnn_collate_fn',
        'KimCNNEmbeddingMixin',
        'KimCNNClassifier',
        'lstm_collate_fn',
        'LSTMEmbeddingMixin',
        'LSTMClassifier',
        'DiscriminativeRepresentationLearning',
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
