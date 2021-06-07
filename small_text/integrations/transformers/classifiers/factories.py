from small_text.classifiers.factories import AbstractClassifierFactory
from small_text.integrations.transformers.classifiers.classification import \
    TransformerBasedClassification


class TransformerBasedClassificationFactory(AbstractClassifierFactory):

    def __init__(self, transformer_model, kwargs={}):
        self.transformer_model = transformer_model
        self.kwargs = kwargs

    def new(self):
        return TransformerBasedClassification(self.transformer_model, **self.kwargs)
