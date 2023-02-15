import numpy as np

from scipy.sparse import issparse
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
            if issparse(predictions):
                def compare_rows(mat_one, mat_two, row):
                    row_one = get_row(mat_one, row)
                    row_two = get_row(mat_two, row)
                    same_shape = row_one.shape == row_two.shape
                    return same_shape and (row_one == row_two).all()

                def get_row(mat, row):
                    item = slice(mat.indptr[row], mat.indptr[row+1])
                    return mat.indices[item]

                unchanged = np.array([
                    compare_rows(self.last_predictions, predictions, i)
                    for i in range(self.last_predictions.shape[0])
                ], dtype=bool)
            else:
                unchanged = np.equal(self.last_predictions, predictions)
            self.last_predictions = predictions

            if unchanged.sum() >= predictions.shape[0] * (1 - self.threshold):
                return True

            return False
