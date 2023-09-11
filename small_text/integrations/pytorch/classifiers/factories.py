from small_text.classifiers.factories import AbstractClassifierFactory
from small_text.integrations.pytorch.classifiers.kimcnn import KimCNNClassifier


class KimCNNFactory(AbstractClassifierFactory):

    def __init__(self, num_classes: int, kwargs: dict = {}):
        """
        num_classes : int
            Number of classes.
        kwargs : dict, default={}
            Keyword arguments that are passed to the constructor of each classifier that is built by the factory.
        """
        self.num_classes = num_classes
        self.kwargs = kwargs

    def new(self) -> KimCNNClassifier:
        """Creates a new KimCNNClassifier instance.

        Returns
        -------
        classifier : KimCNNClassifier
            A new instance of KimCNNClassifier which is initialized with the given keyword args `kwargs`.
        """
        return KimCNNClassifier(self.num_classes, **self.kwargs)
