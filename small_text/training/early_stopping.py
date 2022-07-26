import numpy as np


class EarlyStoppingHandler(object):
    pass


class NoopStoppingHandler(EarlyStoppingHandler):

    def __init__(self):
        self.reset()

    def check_early_stop(self, epoch, monitorables, epoch_finished=True):
        _unused = epoch, monitorables, epoch_finished
        return False

    def reset(self):
        pass


class DefaultEarlyStoppingHandler(EarlyStoppingHandler):

    def __init__(self, monitor, min_delta=0, patience=5, threshold=0,
                 multiple_monitorings_per_epoch=False):
        if patience <= 0:
            raise ValueError('Invalid value encountered: '
                             '"patience" needs to be greater or equal 1.')

        self.dtype = {
            'names': ['epoch', 'count', 'train_acc', 'train_loss', 'val_acc', 'val_loss'],
            'formats': [int, int, float, float, float, float]
        }

        self.index_best = -1
        self.history = np.empty((0,), dtype=self.dtype)

        self.monitor = monitor ## TODO: check for train_* / val_*
        self.min_delta = min_delta
        self.patience = patience
        self.threshold = threshold
        self.multiple_monitorings_per_epoch = multiple_monitorings_per_epoch

        self.reset()

    def check_early_stop(self, epoch, monitorables):

        self.history = self.add_to_history(epoch, monitorables)

        greater_is_better = '_acc' in self.monitor
        monitor_sign = 1 if greater_is_better else -1

        if self.threshold > 0 and np.sign(monitorables[self.monitor] - self.threshold) == monitor_sign:
            return True

        if len(self.history) == 1:
            self.index_best = 0
            return False

        previous_best = self.history[self.monitor][self.index_best]
        index_last = self.history.shape[0] - 1

        delta = monitorables[self.monitor] - previous_best
        delta_sign = np.sign(delta)

        if self.min_delta > 0:
            improvement = delta_sign == monitor_sign and np.abs(delta) > self.min_delta
        else:
            improvement = delta_sign == monitor_sign

        if improvement:
            self.index_best = index_last
            return False
        else:
            if index_last - self.index_best >= self.patience:
                return True
            return False

    def add_to_history(self, epoch, monitorables):
        count = (self.history['epoch'] == epoch).sum()
        tuple_monitorables = (monitorables.get('train_acc', None),
                              monitorables.get('train_loss', None),
                              monitorables.get('val_acc', None),
                              monitorables.get('val_loss', None))
        return np.append(self.history,
                         np.array((epoch, count) + tuple_monitorables, dtype=self.dtype))

    def reset(self):
        self.index_best = -1
        self.history = np.empty((0,), dtype=self.dtype)
