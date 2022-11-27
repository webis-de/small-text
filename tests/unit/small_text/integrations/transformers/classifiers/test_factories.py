import unittest
import pytest

from small_text.exceptions import MissingOptionalDependencyError
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from small_text.integrations.transformers.classifiers.classification import \
        TransformerBasedClassification, TransformerModelArguments
    from small_text.integrations.transformers.classifiers.factories import (
        SetFitClassificationFactory,
        TransformerBasedClassificationFactory
    )
    from small_text.integrations.transformers.classifiers.setfit import (
        SetFitClassification,
        SetFitModelArguments
    )

except (PytorchNotFoundError, ModuleNotFoundError):
    pass


@pytest.mark.pytorch
class TransformerBasedClassificationFactoryTest(unittest.TestCase):

    def test_factory_new(self):
        num_classes = 2
        kwargs = {'lr': 0.123}
        factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
            num_classes,
            kwargs=kwargs)
        clf = factory.new()
        self.assertTrue(isinstance(clf, TransformerBasedClassification))
        self.assertEqual(num_classes, clf.num_classes)
        self.assertEqual(kwargs['lr'], clf.lr)


@pytest.mark.pytorch
class SetFitClassificationFactoryTest(unittest.TestCase):

    def test_factory_new_with_missing_optional_dependency(self):
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2')
        num_classes = 4
        classification_kwargs = {}
        factory = SetFitClassificationFactory(setfit_model_args,
                                              num_classes,
                                              classification_kwargs=classification_kwargs)

        with self.assertRaisesRegex(MissingOptionalDependencyError,
                                    'The optional dependency \'setfit\''):
            factory.new()

    @pytest.mark.optional
    def test_factory_new(self):
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2')
        num_classes = 4
        classification_kwargs = {}
        factory = SetFitClassificationFactory(setfit_model_args,
                                              num_classes,
                                              classification_kwargs=classification_kwargs)

        clf = factory.new()
        self.assertTrue(isinstance(clf, SetFitClassification))
        self.assertEqual(setfit_model_args, clf.setfit_model_args)
        self.assertEqual(num_classes, clf.num_classes)
