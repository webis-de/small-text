import numpy as np

from small_text.stopping_criteria.base import StoppingCriterion, check_window_based_predictions


class ClassificationChange(StoppingCriterion):
    """A stopping criterion which stops as soon as the predictions do not change during two
    subsequent checks [ZWH08]_.

    Compared to the original paper, this implementation offers a threshold parameter which lessens
    the stopping requirement so that a percentage of samples are allowed to change.
    The default setting (:code:`threshold=0.0`) will result in the original algorithm.

    .. versionadded:: 1.1.0
    """
    def __init__(self, num_classes, threshold=0.0):
        """
        num_classes : int
            Number of classes.
        threshold : float, default=0.0
            A percentage threshold of how many samples that are allowed to change.
        """
        if threshold < 0 or threshold > 1:
            raise ValueError(f'Threshold must be between 0 and 1 inclusive, but got {threshold}.')

        self.num_classes = num_classes
        self.threshold = threshold

        self.last_predictions = None

    def stop(self, active_learner=None, predictions=None, proba=None, indices_stopping=None):
        check_window_based_predictions(predictions, self.last_predictions)

        if self.last_predictions is None:
            self.last_predictions = predictions
            return False
        else:
            unchanged = np.equal(self.last_predictions, predictions)
            self.last_predictions = predictions

            if unchanged.sum() >= predictions.shape[0] * (1 - self.threshold):
                return True

            return False
