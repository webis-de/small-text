from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.base import clone

from small_text.classifiers.classification import SklearnClassifier


class AbstractClassifierFactory(ABC):

    @abstractmethod
    def new(self):
        pass


class SklearnClassifierFactory(AbstractClassifierFactory):

    def __init__(self, base_estimator, num_classes, kwargs={}):
        if not issubclass(type(base_estimator), BaseEstimator):
            raise ValueError(
                'Given classifier template must be a subclass of '
                'sklearn.base.BaseEstimator. Encountered class was: {}.'
                .format(str(base_estimator.__class__))
            )

        self.base_estimator = base_estimator
        self.num_classes = num_classes
        self.kwargs = kwargs

    def new(self):
        return SklearnClassifier(clone(self.base_estimator), self.num_classes, **self.kwargs)

    def __str__(self):
        return f'SklearnClassifierFactory(base_estimator={type(self.base_estimator).__name__}, ' \
               f'num_classes={self.num_classes}, kwargs={self.kwargs})'
