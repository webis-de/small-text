import logging
import unittest

from logging import Logger, INFO, DEBUG
from unittest.mock import patch

from small_text import VerbosityLogger, verbosity_logger, VERBOSITY_QUIET, VERBOSITY_ALL


class VerbosityLoggerTest(unittest.TestCase):

    @patch('small_text.utils.logging.Logger.debug')
    def test_debug(self, debug_mock):
        logger = VerbosityLogger('logger1')

        msg = 'something happened'
        logger.debug(msg)

        debug_mock.assert_called_with(msg)

    @patch('small_text.utils.logging.Logger.debug')
    def test_debug_suppressed_by_logger_verbosity(self, debug_mock):
        logger = VerbosityLogger('logger1', verbosity=VERBOSITY_QUIET)

        msg = 'something happened'
        logger.debug(msg)

        debug_mock.assert_not_called()

    @patch('small_text.utils.logging.Logger.debug')
    def test_debug_suppressed_by_call_verbosity(self, debug_mock):
        logger = VerbosityLogger('logger1')

        msg = 'something happened'
        logger.debug(msg, verbosity=VERBOSITY_ALL)

        debug_mock.assert_not_called()

    @patch('small_text.utils.logging.Logger.info')
    def test_info(self, info_mock):
        logger = VerbosityLogger('logger1')

        msg = 'something happened'
        logger.info(msg)

        info_mock.assert_called_with(msg)

    @patch('small_text.utils.logging.Logger.info')
    def test_info_suppressed_by_logger_verbosity(self, info_mock):
        logger = VerbosityLogger('logger1', verbosity=VERBOSITY_QUIET)

        msg = 'something happened'
        logger.info(msg)

        info_mock.assert_not_called()

    @patch('small_text.utils.logging.Logger.info')
    def test_info_suppressed_by_call_verbosity(self, info_mock):
        logger = VerbosityLogger('logger1')

        msg = 'something happened'
        logger.debug(msg, verbosity=VERBOSITY_ALL)

        info_mock.assert_not_called()

    @patch('small_text.utils.logging.Logger.warning')
    def test_warning(self, warning_mock):
        logger = VerbosityLogger('logger1')

        msg = 'something happened'
        logger.warning(msg)

        warning_mock.assert_called_with(msg)

    @patch('small_text.utils.logging.Logger.warning')
    def test_warning_suppressed_by_logger_verbosity(self, warning_mock):
        logger = VerbosityLogger('logger1', verbosity=VERBOSITY_QUIET)

        msg = 'something happened'
        logger.warning(msg)

        warning_mock.assert_not_called()

    @patch('small_text.utils.logging.Logger.warning')
    def test_warning_suppressed_by_call_verbosity(self, warning_mock):
        logger = VerbosityLogger('logger1')

        msg = 'something happened'
        logger.warning(msg, verbosity=VERBOSITY_ALL)

        warning_mock.assert_not_called()

    @patch('small_text.utils.logging.Logger.warn')
    def test_warn(self, warn_mock):
        logger = VerbosityLogger('logger1')

        msg = 'something happened'
        logger.warn(msg)

        warn_mock.assert_called_with(msg)

    @patch('small_text.utils.logging.Logger.warn')
    def test_warn_suppressed_by_logger_verbosity(self, warn_mock):
        logger = VerbosityLogger('logger1', verbosity=VERBOSITY_QUIET)

        msg = 'something happened'
        logger.warn(msg)

        warn_mock.assert_not_called()

    @patch('small_text.utils.logging.Logger.warning')
    def test_warn_suppressed_by_call_verbosity(self, warn_mock):
        logger = VerbosityLogger('logger1')

        msg = 'something happened'
        logger.warn(msg, verbosity=VERBOSITY_ALL)

        warn_mock.assert_not_called()

    @patch('small_text.utils.logging.Logger.error')
    def test_error(self, error_mock):
        logger = VerbosityLogger('logger1')

        msg = 'something happened'
        logger.error(msg)

        error_mock.assert_called_with(msg)

    @patch('small_text.utils.logging.Logger.error')
    def test_error_suppressed_by_logger_verbosity(self, error_mock):
        logger = VerbosityLogger('logger1', verbosity=VERBOSITY_QUIET)

        msg = 'something happened'
        logger.error(msg)

        error_mock.assert_not_called()

    @patch('small_text.utils.logging.Logger.warning')
    def test_error_suppressed_by_call_verbosity(self, error_mock):
        logger = VerbosityLogger('logger1')

        msg = 'something happened'
        logger.error(msg, verbosity=VERBOSITY_ALL)

        error_mock.assert_not_called()

    @patch('small_text.utils.logging.Logger.exception')
    def test_exception_with_exc_info(self, exception_mock):
        logger = VerbosityLogger('logger1')

        try:
            raise ValueError('for unit testing')
        except ValueError:
            msg = 'something happened'
            logger.exception(msg)
        finally:
            exception_mock.assert_called_with(msg, exc_info=True)

    @patch('small_text.utils.logging.Logger.exception')
    def test_exception_without_exc_info(self, exception_mock):
        logger = VerbosityLogger('logger1')

        try:
            raise ValueError('for unit testing')
        except ValueError:
            msg = 'something happened'
            logger.exception(msg, exc_info=False)
        finally:
            exception_mock.assert_called_with(msg, exc_info=False)

    @patch('small_text.utils.logging.Logger.exception')
    def test_exception_suppressed_by_logger_verbosity(self, exception_mock):
        logger = VerbosityLogger('logger1', verbosity=VERBOSITY_QUIET)

        try:
            raise ValueError('for unit testing')
        except ValueError:
            msg = 'something happened'
            logger.exception(msg)
        finally:
            exception_mock.assert_not_called()

    @patch('small_text.utils.logging.Logger.exception')
    def test_exception_suppressed_by_call_verbosity(self, exception_mock):
        logger = VerbosityLogger('logger1')

        try:
            raise ValueError('for unit testing')
        except ValueError:
            msg = 'something happened'
            logger.exception(msg, verbosity=VERBOSITY_ALL)
        finally:
            exception_mock.assert_not_called()

    @patch('small_text.utils.logging.Logger.critical')
    def test_critical(self, critical_mock):
        logger = VerbosityLogger('logger1')

        msg = 'something happened'
        logger.critical(msg)

        critical_mock.assert_called_with(msg)

    @patch('small_text.utils.logging.Logger.critical')
    def test_critical_suppressed_by_logger_verbosity(self, critical_mock):
        logger = VerbosityLogger('logger1', verbosity=VERBOSITY_QUIET)

        msg = 'something happened'
        logger.critical(msg)

        critical_mock.assert_not_called()

    @patch('small_text.utils.logging.Logger.critical')
    def test_critical_suppressed_by_call_verbosity(self, critical_mock):
        logger = VerbosityLogger('logger1')

        msg = 'something happened'
        logger.critical(msg, verbosity=VERBOSITY_ALL)

        critical_mock.assert_not_called()

    @patch('small_text.utils.logging.Logger.log')
    def test_log_info(self, log_mock):
        logger = VerbosityLogger('logger1')

        msg = 'something happened'
        logger.log(INFO, msg)

        log_mock.assert_called_with(INFO, msg)

    @patch('small_text.utils.logging.Logger.log')
    def test_log_info_suppressed_by_call_verbosity(self, log_mock):
        logger = VerbosityLogger('logger1')

        msg = 'something happened'
        logger.log(DEBUG, msg)

        log_mock.assert_called_with(DEBUG, msg)

    @patch('small_text.utils.logging.Logger.log')
    def test_log_suppressed_by_logger_verbosity(self, log_mock):
        logger = VerbosityLogger('logger1', verbosity=VERBOSITY_QUIET)

        msg = 'something happened'
        logger.critical(INFO, msg)

        log_mock.assert_not_called()

    @patch('small_text.utils.logging.Logger.log')
    def test_log_suppressed_by_call_verbosity(self, log_mock):
        logger = VerbosityLogger('logger1')

        msg = 'something happened'
        logger.log(INFO, msg, verbosity=VERBOSITY_ALL)

        log_mock.assert_not_called()


class VerbosityLoggerContextManagerTest(unittest.TestCase):

    def test_get_logger(self):
        first_logger = logging.getLogger('logger1')
        self.assertTrue(isinstance(first_logger, Logger))

        with verbosity_logger():
            second_logger = logging.getLogger('logger2')
            self.assertTrue(isinstance(second_logger, VerbosityLogger))
