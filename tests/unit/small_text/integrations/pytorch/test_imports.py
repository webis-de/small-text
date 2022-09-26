import unittest
import pytest


@pytest.mark.pytorch
class ImportTest(unittest.TestCase):

    def test_import_integration_module(self):
        from small_text import PytorchNotFoundError

    def test_import_base_module(self):
        from small_text import PytorchModelSelectionMixin
        from small_text import PytorchClassifier

    def test_import_classifiers_module(self):
        from small_text import kimcnn_collate_fn
        from small_text import KimCNNEmbeddingMixin
        from small_text import KimCNNClassifier
        from small_text import ExpectedGradientLength
        from small_text import ExpectedGradientLengthLayer
        from small_text import ExpectedGradientLengthMaxWord

    def test_import_factories_module(self):
        from small_text import AbstractClassifierFactory
        from small_text import KimCNNFactory

    def test_import_datasets_module(self):
        from small_text import PytorchDataset
        from small_text import PytorchDatasetView
        from small_text import PytorchTextClassificationDataset
        from small_text import PytorchTextClassificationDatasetView

    def test_import_models_module(self):
        from small_text import KimCNN

    def test_import_query_strategies_module(self):
        from small_text import ExpectedGradientLength
        from small_text import ExpectedGradientLengthLayer
        from small_text import ExpectedGradientLengthMaxWord
