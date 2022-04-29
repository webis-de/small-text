class ActiveLearnerException(Exception):
    """Base class for active learner exceptions."""
    pass


class LearnerNotInitializedException(ActiveLearnerException):
    """Raised when the learner is queried without being initialized."""
    pass


class ConstraintViolationError(RuntimeError):
    """Raised when there is a mismatch between the passed arguments and the capabilities of
    a query strategy.

    This means a query strategy is not capable of handling the passed arguments, for example
    when you pass multi-label data to a query strategy that requires single-label data.
    """
    pass


class MissingOptionalDependencyError(RuntimeError):
    """Raised when an optional dependency is required but cannot be imported.
    """
    pass
