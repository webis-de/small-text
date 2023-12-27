import numpy as np

from sklearn.metrics import cohen_kappa_score

from small_text.stopping_criteria.base import StoppingCriterion, check_window_based_predictions


def _adapted_cohen_kappa_score(y1, y2, *, labels=None, weights=None, sample_weight=None):
    """Extends cohen's kappa by intercepting the special case of a perfect agreement, which results in a
    division by zero when adhering to the original formula. In case of a perfect agreement `1.0` is returned, otherwise
    the call is delegated to the `cohen_kappa_score()` implementation in scikit-learn.

    .. seealso::
       Dcumentation of the underlying `cohen_kappa_score()` method in scikit-learn.
           https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
    """
    if np.array_equal(y1, y2):
        return 1.0
    else:
        return cohen_kappa_score(
            y1,
            y2,
            labels=labels,
            weights=weights,
            sample_weight=sample_weight
        )


class KappaAverage(StoppingCriterion):
    """A stopping criterion which measures the agreement between sets of predictions [BV09]_.

    .. versionchanged:: 1.3.3
       The previous implementation, which was flawed, has been corrected.

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
            labels = np.arange(self.num_classes)
            cohens_kappa = _adapted_cohen_kappa_score(predictions, self.last_predictions, labels=labels)

            self.kappa_history.append(cohens_kappa)
            self.last_predictions = predictions

            if len(self.kappa_history) < self.window_size:
                return False

            self.kappa_history = self.kappa_history[-self.window_size:]

            if np.mean(self.kappa_history) >= self.kappa:
                return True
            else:
                return False
