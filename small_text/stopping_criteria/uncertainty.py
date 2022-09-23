import numpy as np

from scipy.stats import entropy
from small_text.stopping_criteria.base import StoppingCriterion


class OverallUncertainty(StoppingCriterion):
    """A stopping criterion which stops as soon as the average overall uncertainty falls
    below a given threshold [ZWH08]_.

    As a measure of uncertainty, normalized prediction entropy is used. In order to reproduce
    the original implementation pass the unlabeled set `indices_stopping` to `stop()` method.

    .. versionadded:: 1.1.0
    """
    def __init__(self, num_classes, threshold=0.05):
        """
        num_classes : int
            Number of classes.
        threshold : float, default=0.05
            A normalized entropy value below which the criterion indicates to stop.
        """
        if threshold < 0 or threshold >= 1:
            raise ValueError(f'Threshold must be between 0 (inclusive) and 1 (exclusive), '
                             f'but got {threshold}.')

        self.num_classes = num_classes
        self.threshold = threshold

        self.last_predictions = None

    def stop(self, active_learner=None, predictions=None, proba=None, indices_stopping=None):
        if indices_stopping is None:
            raise ValueError('indices_stopping must not be None')

        prediction_entropy = np.apply_along_axis(lambda x: entropy(x), 1, proba[indices_stopping])
        normalized_prediction_entropy = prediction_entropy / np.log(self.num_classes)
        normalized_prediction_entropy = np.mean(normalized_prediction_entropy)

        if normalized_prediction_entropy < self.threshold:
            return True

        return False
