import numpy as np

from abc import ABC, abstractmethod


class StoppingCriterion(ABC):

    @abstractmethod
    def stop(self, active_learner=None, predictions=None, proba=None, indices_stopping=None):
        """
        Parameters
        ----------
        active_learner : small_text.active_learner.PoolBasedActiveLearner
            An active learner instance.
        predictions : np.ndarray[int]
            Predictions for a fixed subset (usually the full train set).
        proba : np.ndarray[float]
            Probability distribution over the possible classes for a fixed subset. This is expected
            to have the same length as `predictions` unless one of `predictions` and `proba`
            is `None`.
        indices_stopping : np.ndarray[int]
            Uses the given indices to select a subset for stopping from either `predictions`
            or `proba` if not `None`. The indices are relative to `predictions` and `proba`.
        """
        pass


def check_window_based_predictions(predictions, last_predictions):
    if predictions is None or np.all([predictions[i] is None for i in range(predictions.shape[0])]):
        raise ValueError('Predictions must not be None!')
    elif last_predictions is not None and predictions.shape[0] != last_predictions.shape[0]:
        raise ValueError(f'Predictions must not differ in size: '
                         f'Current {predictions.shape[0]} / Previous {last_predictions.shape[0]}')


class DeltaFScore(StoppingCriterion):
    """A stopping criterion which stops if the predicted change of the F-score falls below
    a threshold [AB19]_.

    .. note:: This criterion is only applicable for binary classification."""
    def __init__(self, num_classes, window_size=3, threshold=0.05):
        """
        num_classes : int
            Number of classes.
        window_size : int, default=3
            Defines number of iterations for which the predictions are taken into account, i.e.
            this stopping criterion only sees the last `window_size`-many states of the prediction
            array passed to `stop()`.
        threshold : float, default=0.05
            The criterion stops when the predicted F-score falls below this threshold.
        """
        self.num_classes = num_classes

        if num_classes != 2:
            raise ValueError('DeltaFScore is only applicable for binary classifications '
                             '(requires num_class=2)')

        self.window_size = window_size
        self.threshold = threshold

        self.last_predictions = None
        self.delta_history = []

    def stop(self, active_learner=None, predictions=None, proba=None, indices_stopping=None):
        check_window_based_predictions(predictions, self.last_predictions)

        if self.last_predictions is None:
            self.last_predictions = predictions
            return False
        else:
            agreement = (predictions == self.last_predictions).astype(int).sum()
            disagreement_old_positive = ((self.last_predictions == 1) & (predictions == 0)).astype(int).sum()
            disagreement_new_positive = ((self.last_predictions == 0) & (predictions == 1)).astype(int).sum()

            denominator = (2 * agreement + disagreement_old_positive + disagreement_new_positive)
            delta_f = 1 - 2 * agreement / denominator

            self.delta_history.append(delta_f)
            self.last_predictions = predictions

            if len(self.delta_history) < self.window_size:
                return False

            self.delta_history = self.delta_history[-self.window_size:]

            if np.all(np.array(self.delta_history) < self.threshold):
                return True
            else:
                return False
