import logging
import numpy as np

from abc import ABC


class EarlyStoppingHandler(ABC):

    def check_early_stop(self, epoch, measured_values):
        """Checks if the training should be stopped early. The decision is made based on
        the masured values of one or more quantitative metrics over time.

        Parameters
        ----------
        epoch : int
            The number of the current epoch. Multiple checks per epoch are allowed.
        measure_values : dict of str to float
            A dictionary of measured values.
        """
        pass


class NoopEarlyStopping(EarlyStoppingHandler):
    """A no-operation early stopping handler which never stops. This is for developer
    convenience only, you will likely not need this in an application setting.

    .. versionadded:: 1.1.0
    """

    def check_early_stop(self, epoch, measured_values):
        """Checks if the training should be stopped early. The decision is made based on
        the masured values of one or more quantitative metrics over time.

        Parameters
        ----------
        epoch : int
            The number of the current epoch (1-indexed). Multiple checks per epoch are allowed.
        measured_values : dict of str to float
            A dictionary of measured values.
        """
        _unused = epoch, measured_values  # noqa:F841
        return False


class EarlyStopping(EarlyStoppingHandler):
    """A default early stopping implementation which supports stopping based on thresholds
    or based on patience-based improvement.

    .. versionadded:: 1.1.0
    """
    def __init__(self, metric, min_delta=1e-14, patience=5, threshold=0.0):
        """
        Parameters
        ----------
        metric : small_text.training.metrics.Metric
            The measured training metric which will be monitored for early stopping.
        min_delta : float, default=1e-14
            The minimum absolute value to consider a change in the masured value as an
            improvement.
        patience : int, default=5
            The maximum number of steps (i.e. calls to `check_early_stop()`) which can yield no
            improvement. Disable patience-based improvement monitoring by setting patience to
            a value less than zero.
        threshold : float, default=0.0
            If greater zero, then early stopping is triggered as soon as the current measured value
            crosses ('valid_acc', 'train_acc') or falls below ('valid_loss', 'train_loss')
            the given threshold. Disable threshold-based stopping by setting the threshold to
            a value lesser than or equal zero.
        """
        self._validate_arguments(metric, min_delta, patience, threshold)

        self._dtype = {
            'names': ['epoch', 'count', 'train_acc', 'train_loss', 'val_acc', 'val_loss'],
            'formats': [int, int, float, float, float, float]
        }

        self.metric = metric
        self.min_delta = min_delta
        self.patience = patience
        self.threshold = threshold

        self._index_best = -1
        self._history = np.empty((0,), dtype=self._dtype)

    def _validate_arguments(self, metric, min_delta, patience, threshold):
        if min_delta < 0:
            raise ValueError('Invalid value encountered: '
                             '"min_delta" needs to be greater than zero.')

        if patience < 0 and threshold <= 0:
            raise ValueError('Invalid configuration encountered: '
                             'Either "patience" or "threshold" must be enabled.')

        if '_acc' in metric.name and (threshold < 0.0 or threshold > 1.0):
            raise ValueError('Invalid value encountered: '
                             '"threshold" needs to be within the interval [0, 1] '
                             'for accuracy metrics.')

    def check_early_stop(self, epoch, measured_values):
        """Checks if the training should be stopped early. The decision is made based on
        the masured values of one or more quantitative metrics over time.

        1. Returns `True` if the threshold is crossed/undercut (for accuracy/loss respectively).
        2. Checks for an improvement and returns `True` if patience has been execeeded.
        3. Otherwise, return `False`.

        Parameters
        ----------
        epoch : int
            The number of the current epoch (1-indexed). Multiple checks per epoch are allowed.
        measured_values : dict of str to float
            A dictionary of measured values.
        """
        if epoch <= 0:
            raise ValueError('Argument "epoch" must be greater than zero.')

        self._history = self.add_to_history(epoch, measured_values)

        metric_sign = -1 if self.metric.lower_is_better else 1

        measured_value = measured_values.get(self.metric.name, None)
        has_crossed_threshold = measured_value is not None and \
            np.sign(measured_value - self.threshold) == metric_sign
        if self.threshold > 0 and has_crossed_threshold:
            logging.debug(f'Early stopping: Threshold exceeded. '
                          f'[value={measured_values[self.metric.name]}, '
                          f'threshold={self.threshold}]')
            return True
        elif measured_value is None:
            return False

        if len(self._history) == 1:
            self._index_best = 0
            return False

        if self.patience < 0:
            return False
        else:
            return self._check_for_improvement(measured_values, metric_sign)

    def _check_for_improvement(self, measured_values, metric_sign):
        previous_best = self._history[self.metric.name][self._index_best]
        index_last = self._history.shape[0] - 1

        delta = measured_values[self.metric.name] - previous_best
        delta_sign = np.sign(delta)

        if self.min_delta > 0:
            improvement = delta_sign == metric_sign and np.abs(delta) >= self.min_delta
        else:
            improvement = delta_sign == metric_sign

        if improvement:
            self._index_best = index_last
            return False
        else:
            history_since_previous_best = self._history[self._index_best + 1:][self.metric.name]
            rows_not_nan = np.logical_not(np.isnan(history_since_previous_best))
            if rows_not_nan.sum() > self.patience:
                logging.debug(f'Early stopping: Patience exceeded.'
                              f'{{value={index_last-self._index_best}, patience={self.patience}}}')
                return True
            return False

    def add_to_history(self, epoch, measured_values):
        count = (self._history['epoch'] == epoch).sum()
        tuple_measured_values = (measured_values.get('train_acc', None),
                                 measured_values.get('train_loss', None),
                                 measured_values.get('val_acc', None),
                                 measured_values.get('val_loss', None))
        return np.append(self._history,
                         np.array((epoch, count) + tuple_measured_values, dtype=self._dtype))


class EarlyStoppingOrCondition(EarlyStoppingHandler):
    """A sequential early stopping handler which bases its response on a list of sub handlers.
    As long as one early stopping handler returns `True` the aggregated response will be `True`,
    i.e. the answer is the combination of single answers aggregated by a logical or.

    .. versionadded:: 1.1.0
    """
    def __init__(self, early_stopping_handlers):
        """
        Parameters
        ----------
        early_stopping_handlers : list of EarlyStoppingHandler
            A list of early stopping (sub-)handlers.
        """
        self.early_stopping_handlers = early_stopping_handlers

    def check_early_stop(self, epoch, measured_values):
        """Checks if the training should be stopped early. The decision is made based on
        the masured values of one or more quantitative metrics over time.

        Parameters
        ----------
        epoch : int
            The number of the current epoch (1-indexed). Multiple checks per epoch are allowed.
        measured_values : dict of str to float
            A dictionary of measured values.
        """
        results = []
        for early_stopping_handler in self.early_stopping_handlers:
            results.append(early_stopping_handler.check_early_stop(epoch, measured_values))
        return np.any(results)


class EarlyStoppingAndCondition(EarlyStoppingHandler):
    """A sequential early stopping handler which bases its response on a list of sub handlers.
    Whenever all sub early stopping handler return `True` the aggregated response will be `True`,
    i.e. the answer is the combination of single answers aggregated by a logical and.

    .. versionadded:: 1.1.0
    """
    def __init__(self, early_stopping_handlers):
        """
        Parameters
        ----------
        early_stopping_handlers : list of EarlyStoppingHandler
            A list of early stopping (sub-)handlers.
        """
        self.early_stopping_handlers = early_stopping_handlers

    def check_early_stop(self, epoch, measured_values):
        """Checks if the training should be stopped early. The decision is made based on
        the masured values of one or more quantitative metrics over time.

        Parameters
        ----------
        epoch : int
            The number of the current epoch (1-indexed). Multiple checks per epoch are allowed.
        measured_values : dict of str to float
            A dictionary of measured values.
        """
        results = []
        for early_stopping_handler in self.early_stopping_handlers:
            results.append(early_stopping_handler.check_early_stop(epoch, measured_values))
        return np.any(results)
