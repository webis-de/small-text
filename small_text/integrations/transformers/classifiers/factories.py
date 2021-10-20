from small_text.classifiers.factories import AbstractClassifierFactory
from small_text.integrations.transformers.classifiers.classification import \
    TransformerBasedClassification


class TransformerBasedClassificationFactory(AbstractClassifierFactory):

    def __init__(self, transformer_model, num_classes, kwargs={}):
        self.transformer_model = transformer_model
        self.num_classes = num_classes
        self.kwargs = kwargs

    def new(self):
        return TransformerBasedClassification(self.transformer_model,
                                              self.num_classes,
                                              **self.kwargs)
