import logging

from contextlib import contextmanager
from logging import Logger, NOTSET


VERBOSITY_QUIET = 0
VERBOSITY_VERBOSE = 10
VERBOSITY_MORE_VERBOSE = 20
VERBOSITY_ALL = 100


class VerbosityLogger(Logger):

    def __init__(self, name, level=NOTSET, verbosity=VERBOSITY_VERBOSE):
        super().__init__(name, level=level)

        self.verbosity = verbosity

    def debug(self, msg, *args, verbosity=VERBOSITY_VERBOSE, **kwargs):
        if self.verbosity >= verbosity:
            super().debug(msg, *args, **kwargs)

    def info(self, msg, *args, verbosity=VERBOSITY_VERBOSE, **kwargs):
        if self.verbosity >= verbosity:
            super().info(msg, *args, **kwargs)

    def warning(self, msg, *args, verbosity=VERBOSITY_VERBOSE, **kwargs):
        if self.verbosity >= verbosity:
            super().warning(msg, *args, **kwargs)

    def warn(self, msg, *args, verbosity=VERBOSITY_VERBOSE, **kwargs):
        if self.verbosity >= verbosity:
            super().warn(msg, *args, **kwargs)

    def error(self, msg, *args, verbosity=VERBOSITY_VERBOSE, **kwargs):
        if self.verbosity >= verbosity:
            super().error(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, verbosity=VERBOSITY_VERBOSE, **kwargs):
        if self.verbosity >= verbosity:
            super().exception(msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg, *args, verbosity=VERBOSITY_VERBOSE, **kwargs):
        if self.verbosity >= verbosity:
            super().critical(msg, *args, **kwargs)

    def log(self, level, msg, *args, verbosity=VERBOSITY_VERBOSE, **kwargs):
        if self.verbosity >= verbosity:
            super().log(level, msg, *args, **kwargs)


@contextmanager
def verbosity_logger():
    logger_class = logging.getLoggerClass()
    logging.setLoggerClass(VerbosityLogger)
    yield
    logging.setLoggerClass(logger_class)
