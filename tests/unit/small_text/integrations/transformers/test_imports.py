import unittest
import pytest


@pytest.mark.pytorch
class ImportsTest(unittest.TestCase):

    def test_import_datasets_modules(self):
        from small_text import TransformersDataset
        from small_text import TransformersDatasetView

    def test_import_classifier_base_modules(self):
        from small_text import ModelLoadingStrategy

    def test_import_classifiers_modules(self):
        from small_text import transformers_collate_fn
        from small_text import FineTuningArguments
        from small_text import TransformerModelArguments
        from small_text import TransformerBasedEmbeddingMixin
        from small_text import TransformerBasedClassification

    def test_import_setfit_modules(self):
        from small_text import SetFitClassification
        from small_text import SetFitModelArguments
        from small_text import SetFitClassificationEmbeddingMixin

    def test_import_classifier_factories_modules(self):
        from small_text import TransformerBasedClassificationFactory
        from small_text import SetFitClassificationFactory
