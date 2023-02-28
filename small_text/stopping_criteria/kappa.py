import warnings
import numpy as np

from sklearn.metrics import cohen_kappa_score

from small_text.stopping_criteria.base import StoppingCriterion, check_window_based_predictions


class KappaAverage(StoppingCriterion):
    """A stopping criterion which measures the agreement between sets of predictions [BV09]_.
    """
    def __init__(self, num_classes, window_size=3, kappa=0.99):
        """
        num_classes : int
            Number of classes.
        window_size : int, default=3
            Defines number of iterations for which the predictions are taken into account, i.e.
            this stopping criterion only sees the last `window_size`-many states of the prediction
            array passed to `stop()`.
        kappa : float, default=0.99
            The criterion stops when the agreement between two consecutive predictions within
            the window falls below this threshold.
        """
        self.num_classes = num_classes
        self.window_size = window_size
        self.kappa = kappa

        self.last_predictions = None
        self.kappa_history = []

    def stop(self, active_learner=None, predictions=None, proba=None, indices_stopping=None):
        check_window_based_predictions(predictions, self.last_predictions)

        if self.last_predictions is None:
            self.last_predictions = predictions
            return False
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                labels = np.arange(self.num_classes)
                cohens_kappa = cohen_kappa_score(predictions, self.last_predictions, labels=labels)

            self.kappa_history.append(cohens_kappa)
            self.last_predictions = predictions

            if len(self.kappa_history) < self.window_size:
                return False

            self.kappa_history = self.kappa_history[-self.window_size:]
            deltas = np.abs([a - b for a, b in zip(self.kappa_history, self.kappa_history[1:])])

            if all(np.isnan(deltas)):
                warnings.warn('Nan encountered within the list of kappa values', RuntimeWarning)
                return True

            if np.mean(deltas) < (1 - self.kappa):
                return True
            else:
                return False
