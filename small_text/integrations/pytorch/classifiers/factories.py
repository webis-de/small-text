from small_text.classifiers.factories import AbstractClassifierFactory
from small_text.integrations.pytorch.classifiers.kimcnn import KimCNNClassifier


class KimCNNFactory(AbstractClassifierFactory):

    def __init__(self, classifier_name, kwargs={}):
        self.classifier_name = classifier_name
        self.kwargs = kwargs

    def new(self):
        return KimCNNClassifier(**self.kwargs)
