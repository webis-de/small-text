from small_text.exceptions import ActiveLearnerException


class QueryException(ActiveLearnerException):
    """Exceptions thrown during executing a query strategy."""
    pass


class EmptyPoolException(QueryException):
    """Thrown when there are zero unlabeled examples."""
    pass


class PoolExhaustedException(QueryException):
    """Thrown when there is an insufficient amount of unlabeled examples."""
