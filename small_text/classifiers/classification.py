from abc import ABC, abstractmethod

import numpy as np
from typing import TYPE_CHECKING

from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.utils.multiclass import is_multilabel

from small_text.data.datasets import Dataset
from small_text.utils.classification import empty_result, prediction_result
from small_text.utils.data import check_training_data

if TYPE_CHECKING:
    import numpy.typing as npt

    from typing import Union, Tuple
    from scipy.sparse import csr_matrix


class Classifier(ABC):
    """Abstract base class for classifiers that can be used with the active learning components.
    """

    @abstractmethod
    def fit(self, train_set: Dataset, weights: "Union[npt.NDArray[np.double], None]" = None) -> "Classifier":
        """Train the model using the given train set.

        Parameters
        ----------
        train_set : Dataset
            The dataset used for training the model.
        weights : np.ndarray[np.double] or None, default=None
            Sample weights or None.
        """
        pass

    @abstractmethod
    def predict(self,
                data_set: Dataset,
                return_proba: bool = False,
                multi_label_threshold: float = 0.5,
                **kwargs) \
            -> "Union[npt.NDArray[np.uint], Tuple[npt.NDArray[np.uint], npt.NDArray[np.double]]," \
               "csr_matrix, Tuple[csr_matrix, csr_matrix]]":
        """Predicts the labels for each sample in the given dataset.

        Parameters
        ----------
        data_set : Dataset
            A dataset for which the labels are to be predicted.
        return_proba : bool, default=False
            If `True`, also returns a probability-like class distribution.
        multi_label_threshold : float, default=0.5
            In multi-label classification, a label is predicted for a sample only if the respective probability value
            is greater than `multi_label_threshold`. Must be between 0.0 and 1.0. Ignored when `multi_label` is False.
        """
        pass

    @abstractmethod
    def predict_proba(self,
                      data_set: Dataset,
                      multi_label_threshold: float = 0.5,
                      **kwargs) -> "npt.NDArray[np.double]":
        """Predicts the label distribution for each sample in the given dataset.

        Parameters
        ----------
        data_set : Dataset
            A dataset for which the labels are to be predicted.
        multi_label_threshold : float, default=0.5
            In multi-label classification, a label is predicted for a sample only if the respective probability value
            is greater than `multi_label_threshold`. Must be between 0.0 and 1.0. Ignored when `multi_label` is False.
        """
        pass


class SklearnClassifier(Classifier):
    """An adapter for using scikit-learn estimators.

    Notes
    -----
    The multi-label settings currently assumes that the underlying classifer returns a sparse
    matrix if trained on sparse data.
    """

    def __init__(self, model: BaseEstimator, num_classes, multi_label=False):
        """
        Parameters
        ----------
        model : BaseEstimator
            A scikit-learn estimator that implements `fit` and `predict_proba`.
        num_classes : int
            Number of classes which are to be trained and predicted.
        multi_label : bool, default=False
            If `False`, the classes are mutually exclusive, i.e. the prediction step results in
            exactly one predicted label per instance.
        """
        if multi_label:
            self.model = OneVsRestClassifier(model)
        else:
            self.model = model
        self.num_classes = num_classes
        self.multi_label = multi_label

    def fit(self, train_set, weights=None):
        """Trains the model using the given train set.

        Parameters
        ----------
        train_set : SklearnDataset
            The dataset used for training the model.
        weights : np.ndarray[np.double] or None, default=None
            Sample weights or None.

        Returns
        -------
        clf : SklearnClassifier
            Returns the current classifier with a fitted model.
        """
        check_training_data(train_set, None, weights=weights)
        if self.multi_label and weights is not None:
            raise ValueError('Sample weights are not supported for multi-label SklearnClassifier.')

        y = train_set.y
        if self.multi_label and not is_multilabel(y):
            raise ValueError('Invalid input: Given labeling must be recognized as '
                             'multi-label according to sklearn.utils.multilabel.is_multilabel(y)')
        elif not self.multi_label and is_multilabel(y):
            raise ValueError('Invalid input: Given labeling is recognized as multi-label labeling '
                             'but the classifier is set to single-label mode')

        fit_kwargs = dict() if self.multi_label else dict({'sample_weight': weights})
        self.model.fit(train_set.x, y, **fit_kwargs)
        return self

    def predict(self, data_set: Dataset, return_proba=False, multi_label_threshold: float = 0.5):
        """
        Predicts the labels for the given dataset.

        Parameters
        ----------
        data_set : SklearnDataset
            A dataset for which the labels are to be predicted.
        return_proba : bool, default=False
            If `True`, also returns a probability-like class distribution.
        multi_label_threshold : float, default=0.5
            In multi-label classification, a label is predicted for a sample only if the respective probability value
            is greater than `multi_label_threshold`. Must be between 0.0 and 1.0. Ignored when `multi_label` is False.

        Returns
        -------
        predictions : np.ndarray[np.uint] or csr_matrix
            List of predictions if the classifier was fitted on multi-label data,
            otherwise a sparse matrix of predictions.
        probas : np.ndarray[np.double]
            List of probabilities (or confidence estimates) if `return_proba` is True.
        """
        if len(data_set) == 0:
            return empty_result(self.multi_label, self.num_classes, return_prediction=True,
                                return_proba=return_proba)

        proba = self.model.predict_proba(data_set.x)

        return prediction_result(proba,
                                 self.multi_label,
                                 self.num_classes,
                                 return_proba=return_proba,
                                 multi_label_threshold=multi_label_threshold)

    def predict_proba(self, data_set: Dataset, multi_label_threshold: float = 0.5):
        """Predicts the label distribution for each sample in the given dataset.

        Parameters
        ----------
        data_set : SklearnDataset
            A dataset for which the labels are to be predicted.
        multi_label_threshold : float, default=0.5
            In multi-label classification, a label is predicted for a sample only if the respective probability value
            is greater than `multi_label_threshold`. Must be between 0.0 and 1.0. Ignored when `multi_label` is False.
        """
        if len(data_set) == 0:
            return empty_result(self.multi_label, self.num_classes, return_prediction=False, return_proba=True)

        return self.model.predict_proba(data_set.x)


class ConfidenceEnhancedLinearSVC(LinearSVC):
    """Extends scikit-learn's LinearSVC class to provide confidence estimates.
    """

    def __init__(self, linearsvc_kwargs: dict = None):
        """
        Parameters
        ----------
        linearsvc_kwargs : dict, default=None
            Kwargs for the LinearSVC superclass.
        """
        self.linearsvc_kwargs = dict() if linearsvc_kwargs is None else linearsvc_kwargs
        super().__init__(**self.linearsvc_kwargs)

    def predict(self, x, return_proba=False):

        if return_proba:
            proba = self.predict_proba(x)

            target_class = np.argmax(proba, axis=1)
            return target_class, proba
        else:
            return super().predict(x)

    def predict_proba(self, x):

        scores = self.decision_function(x)
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
    def embed(self, data_set, return_proba=False, multi_label_threshold: float = 0.5, **kwargs) \
            -> "Union[npt.NDArray[np.double]]":
        """
        Parameters
        ----------
        data_set : Dataset
            A dataset for which each instance is used to compute its embedding vector.
        return_proba : bool, default=False
            If `True`, also returns a probability-like class distribution.
        multi_label_threshold : float, default=0.5
            In multi-label classification, a label is predicted for a sample only if the respective probability value
            is greater than `multi_label_threshold`. Must be between 0.0 and 1.0. Ignored when `multi_label` is False.
        """
        pass
