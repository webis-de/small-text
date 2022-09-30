from small_text.stopping_criteria.base import StoppingCriterion


class MaxIterations(StoppingCriterion):
    """Stops after a fixed number of iterations.

    .. versionadded:: 1.1.0
    """
    def __init__(self, max_iterations):
        """
        max_iterations : int
            Number of iterations after which the criterion will indicate to stop.
        """
        if max_iterations < 1:
            raise ValueError(f'Argument max_iterations must be greater or equal 1. '
                             f'Encountered: {max_iterations}.')

        self.max_iterations = max_iterations
        self.current_iteration = 0

    def stop(self, active_learner=None, predictions=None, proba=None, indices_stopping=None):
        if self.current_iteration+1 >= self.max_iterations:
            return True

        self.current_iteration += 1
        return False
