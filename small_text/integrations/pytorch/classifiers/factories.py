from small_text.classifiers.factories import AbstractClassifierFactory
from small_text.integrations.pytorch.classifiers.kimcnn import KimCNNClassifier


class KimCNNFactory(AbstractClassifierFactory):

    def __init__(self, classifier_name, num_classes, kwargs={}):
        self.classifier_name = classifier_name
        self.num_classes = num_classes
        self.kwargs = kwargs

    def new(self):
        return KimCNNClassifier(self.num_classes, **self.kwargs)
