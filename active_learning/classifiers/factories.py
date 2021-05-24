from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator

from active_learning.classifiers.classification import SklearnClassifier


class AbstractClassifierFactory(ABC):

    @abstractmethod
    def new(self):
        pass


class SklearnClassifierFactory(AbstractClassifierFactory):

    def __init__(self, clf_template, kwargs={}):
        if not issubclass(type(clf_template), BaseEstimator):
            raise ValueError('Given classifier template must be a subclass of '
                             'sklearn.base.BaseEstimator. Encountered class was: {}.'.format(
                str(clf_template.__class__)
            ))

        self.base_estimator_class = clf_template.__class__
        self.kwargs = kwargs

    def new(self):
        return SklearnClassifier(self.base_estimator_class(**self.kwargs))
