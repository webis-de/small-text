from active_learning.classifiers.factories import AbstractClassifierFactory
from active_learning.integrations.pytorch.classifiers.kimcnn import KimCNNClassifier


class KimCNNFactory(AbstractClassifierFactory):

    def __init__(self, classifier_name, kwargs={}):
        self.classifier_name = classifier_name
        self.kwargs = kwargs

    def new(self):
        return KimCNNClassifier(**self.kwargs)
