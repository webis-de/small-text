from small_text.classifiers.factories import AbstractClassifierFactory
from small_text.integrations.pytorch.classifiers.kimcnn import KimCNNClassifier


class KimCNNFactory(AbstractClassifierFactory):

    def __init__(self, classifier_name, num_classes, kwargs={}):
        """
        classifier_name : str
            Obsolete. Do not use any more.
        num_classes : int
            Number of classes.
        kwargs : dict
            Keyword arguments that are passed to the constructor of each classifier that is built by the factory.
        """
        self.classifier_name = classifier_name
        self.num_classes = num_classes
        self.kwargs = kwargs

    def new(self):
        """Creates a new KimCNNClassifier instance.

        Returns
        -------
        classifier : KimCNNClassifier
            A new instance of KimCNNClassifier which is initialized with the given keyword args `kwargs`.
        """
        return KimCNNClassifier(self.num_classes, **self.kwargs)
