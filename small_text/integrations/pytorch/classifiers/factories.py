from small_text.classifiers.factories import AbstractClassifierFactory
from small_text.integrations.pytorch.classifiers.kimcnn import KimCNNClassifier
from small_text.integrations.pytorch.classifiers.lstm import LSTMClassifier


class KimCNNClassifierFactory(AbstractClassifierFactory):

    def __init__(self, num_classes: int, classification_kwargs: dict = {}):
        """
        num_classes : int
            Number of classes.
        kwargs : dict, default={}
            Keyword arguments that are passed to the constructor of each classifier that is built by the factory.
        """
        self.num_classes = num_classes
        self.classification_kwargs = classification_kwargs

    def new(self) -> KimCNNClassifier:
        """Creates a new KimCNNClassifier instance.

        Returns
        -------
        classifier : KimCNNClassifier
            A new instance of KimCNNClassifier which is initialized with the given keyword args `kwargs`.
        """
        return KimCNNClassifier(self.num_classes, **self.classification_kwargs)


class LSTMClassifierFactory(AbstractClassifierFactory):

    def __init__(self, num_classes: int, classification_kwargs: dict = {}):
        """
        num_classes : int
            Number of classes.
        kwargs : dict, default={}
            Keyword arguments that are passed to the constructor of each classifier that is built by the factory.
        """
        self.num_classes = num_classes
        self.classification_kwargs = classification_kwargs

    def new(self) -> LSTMClassifier:
        """Creates a new LSTMClassifier instance.

        Returns
        -------
        classifier : LSTMClassifier
            A new instance of LSTMClassifier which is initialized with the given keyword args `kwargs`.
        """
        return LSTMClassifier(self.num_classes, **self.classification_kwargs)
