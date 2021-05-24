import warnings
import numpy as np

from sklearn.metrics import cohen_kappa_score


class KappaAverage(object):
    """
    A stopping criterion which measures the agreement between sets of predictions.

    References
    ----------
    .. [BV09] M. Bloodgood and K. Vijay-Shanker. 2009.
       A method for stopping active learning based on stabilizing predictions and the need for user-adjustable stopping.
       In Proceedings of the Thirteenth Conference on Computational Natural Language Learning (CoNLL '09).
       Association for Computational Linguistics, USA, 39–47.
    """
    def __init__(self, kappa=0.99, window_size=3, labels=[0, 1]):
        self.last_predictions = None
        self.kappa_history = []
        self.kappa = kappa
        self.window_size = window_size
        self.labels = labels

    def evaluate(self, predictions):

        if self.last_predictions is None:
            self.last_predictions = predictions
            return False
        else:
            # TODO: labels
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                cohens_kappa = cohen_kappa_score(predictions, self.last_predictions, labels=self.labels)

            self.kappa_history.append(cohens_kappa)
            self.last_predictions = predictions

            if len(self.kappa_history) < self.window_size:
                return False

            self.kappa_history = self.kappa_history[-(self.window_size):]
            deltas = np.abs([a - b for a, b in zip(self.kappa_history, self.kappa_history[1:])])

            if all(np.isnan(deltas)):
                warnings.warn('Nan encountered within the list of kappa values', RuntimeWarning)
                return True

            if np.mean(deltas) < (1 - self.kappa):
                return True
            else:
                return False
