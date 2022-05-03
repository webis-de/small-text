import unittest
import pytest

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from small_text.integrations.transformers.classifiers.classification import \
        TransformerBasedClassification, TransformerModelArguments
    from small_text.integrations.transformers.classifiers.factories import TransformerBasedClassificationFactory
except (PytorchNotFoundError, ModuleNotFoundError):
    pass


@pytest.mark.pytorch
class TransformerBasedClassificationFactoryTest(unittest.TestCase):

    def test_factory_new(self):
        num_classes = 2
        factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
            num_classes,
            kwargs={'device': 'cpu'})
        clf = factory.new()
        self.assertTrue(isinstance(clf, TransformerBasedClassification))
        self.assertEqual(num_classes, clf.num_classes)
