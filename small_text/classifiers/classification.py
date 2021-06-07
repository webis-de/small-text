from abc import ABC, abstractmethod

import numpy as np

from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC


class Classifier(ABC):
    """Abstract base blass for classifiers that can be used with the active learning components."""

    @abstractmethod
    def fit(self, train_set):
        pass

    @abstractmethod
    def predict(self, test_set, return_proba=False):
        pass

    @abstractmethod
    def predict_proba(self, test_set):
        pass


class SklearnClassifier(Classifier):
    """An adapter for using scikit-learn estimators."""

    def __init__(self, clf):
        """
        Parameters
        ----------
        clf : sklearn.base.BaseEstimator
            A scikit-learn estimator that implements `fit` and `predict_proba`.
        """
        self.clf = clf

    def fit(self, train_set):
        """
        Trains the model using the given `train_set`.

        Parameters
        ----------
        train_set : SklearnDataSet
            The dataset used for training the model.

        Returns
        -------
        clf : SklearnClassifier
            Returns the fitted classifier.
        """
        self.clf.fit(train_set.x, train_set.y)
        return self

    def predict(self, test_set, return_proba=False):
        """

        Returns
        -------
        predictions : np.ndarray[int]
            List of predictions.
        probas : np.ndarray[float] (optional)
            List of probabilities (or confidence estimates) if `return_proba` is True.
        """

        proba = self.predict_proba(test_set)
        predictions = np.argmax(proba, axis=1)

        if return_proba:
            return predictions, proba

        return predictions

    def predict_proba(self, test_set):
        return self.clf.predict_proba(test_set.x)


class ConfidenceEnhancedLinearSVC(LinearSVC):

    def __init__(self, linearsvc_kwargs=None):
        """
        Extends scikit-learn's LinearSVC class to provide confidence estimates.

        Parameters
        ----------
        linearsvc_kwargs : dict
            Kwargs for the LinearSVC superclass.
        """
        linearsvc_kwargs = dict() if linearsvc_kwargs is None else linearsvc_kwargs
        super().__init__(**linearsvc_kwargs)

    def predict(self, data_set, return_proba=False):

        if return_proba:
            proba = self.predict_proba(data_set)

            target_class = np.argmax(proba, axis=1)
            return target_class, proba
        else:
            return super().predict(data_set)

    def predict_proba(self, data_set):

        scores = self.decision_function(data_set)
        if len(scores.shape) == 1:
            proba = np.zeros((scores.shape[0], 2))
            scores = np.apply_along_axis(self._sigmoid, -1, scores)
            target = np.array([0 if score <= 0.5 else 1 for score in scores])
            scores = np.array([0.5+(0.5-score) if score <= 0.5 else 0.5+(score-0.5) for score in scores])
            for i, score in enumerate(scores):
                proba[i, target[i]] = score
                proba[i, target[i]-1] = 1-score
            proba = normalize(proba, norm='l1')
            return proba
        else:
            proba = np.apply_along_axis(self._sigmoid, -1, scores)
            proba = normalize(proba, norm='l1')
            return proba

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class EmbeddingMixin(ABC):

    @abstractmethod
    def embed(self, data_set):
        """
        Parameters
        ----------
        data_set : DataSet
            A dataset for which each instance is used to compute its embedding vector.
        """
        pass
