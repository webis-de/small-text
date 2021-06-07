class ActiveLearnerException(Exception):
    """Base class for active learner exceptions."""
    pass


class LearnerNotInitializedException(ActiveLearnerException):
    """Thrown when the learner is queried without being initialized."""
    pass
