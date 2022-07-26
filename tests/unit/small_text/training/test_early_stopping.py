import unittest
import numpy as np

from small_text.training.early_stopping import DefaultEarlyStoppingHandler, NoopStoppingHandler


class NoopStoppingHandlerTest(unittest.TestCase):

    def test_stopping_handler_all(self):
        stopping_handler = NoopStoppingHandler()
        self.assertFalse(stopping_handler.check_early_stop(0, dict(), epoch_finished=False))
        self.assertFalse(stopping_handler.check_early_stop(0, dict()))
        self.assertFalse(stopping_handler.check_early_stop(1, dict()))

        stopping_handler.reset()


class LossBasedEarlyStoppingHandlerTest(object):

    def get_monitor(self):
        raise NotImplementedError('monitor must be implemented')

    def test_stopping_handler_init_default(self):
        stopping_handler = DefaultEarlyStoppingHandler(self.get_monitor())
        self.assertIsNotNone(stopping_handler.history)
        self.assertEqual((0,), stopping_handler.history.shape)
        self.assertEqual(self.get_monitor(), stopping_handler.monitor)
        self.assertEqual(0, stopping_handler.min_delta)
        self.assertEqual(5, stopping_handler.patience)
        self.assertEqual(0, stopping_handler.threshold)
        self.assertFalse(stopping_handler.multiple_monitorings_per_epoch)

    def test_stopping_handler_stop_val_loss_no_patience(self):
        with self.assertRaisesRegex(ValueError,
                                    "Invalid value encountered: \"patience\" needs to be"):
            DefaultEarlyStoppingHandler(self.get_monitor(), min_delta=0.01, patience=0)

    def test_stopping_handler_stop_val_loss_threshold(self):
        stopping_handler = DefaultEarlyStoppingHandler(self.get_monitor(), threshold=0.1, patience=2)

        self.assertTrue(stopping_handler.check_early_stop(0, {self.get_monitor(): 0.05}))
        stopping_handler.reset()

        self.assertFalse(stopping_handler.check_early_stop(0, {self.get_monitor(): 0.2}))
        self.assertTrue(stopping_handler.check_early_stop(1, {self.get_monitor(): 0.009}))

    def test_stopping_handler_stop_val_loss_patience(self):
        stopping_handler = DefaultEarlyStoppingHandler(self.get_monitor(), min_delta=0.01, patience=2)

        self.assertFalse(stopping_handler.check_early_stop(0, {self.get_monitor(): 0.04}))
        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_monitor(): 0.031}))
        self.assertTrue(stopping_handler.check_early_stop(2, {self.get_monitor(): 0.033}))

    def test_stopping_handler_stop_val_acc_patience_and_stop(self):
        stopping_handler = DefaultEarlyStoppingHandler(self.get_monitor(), min_delta=0.01, patience=2)

        self.assertFalse(stopping_handler.check_early_stop(0, {self.get_monitor(): 0.065}))
        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_monitor(): 0.068}))
        self.assertTrue(stopping_handler.check_early_stop(2, {self.get_monitor(): 0.066}))

    def test_stopping_handler_stop_val_acc_patience_and_dont_stop(self):
        stopping_handler = DefaultEarlyStoppingHandler(self.get_monitor(), min_delta=0.01, patience=3)

        self.assertFalse(stopping_handler.check_early_stop(0, {self.get_monitor(): 0.065}))
        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_monitor(): 0.065}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_monitor(): 0.066}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_monitor(): 0.055}))

    def test_stopping_handler_stop_val_acc_patience_and_dont_stop_extended(self):
        """Checks if an update of the best index is recognized, i.e. compares to the previous test
        this would stop but does not if a new best index is correctly recognized in epoch 1."""
        stopping_handler = DefaultEarlyStoppingHandler(self.get_monitor(), min_delta=0.01, patience=3)

        self.assertFalse(stopping_handler.check_early_stop(0, {self.get_monitor(): 0.680}))
        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_monitor(): 0.693}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_monitor(): 0.660}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_monitor(): 0.670}))

    def test_stopping_handler_reset(self):
        stopping_handler = DefaultEarlyStoppingHandler(self.get_monitor(), min_delta=0.01, patience=2)

        self.assertFalse(stopping_handler.check_early_stop(0, {self.get_monitor(): 0.004}))
        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_monitor(): 0.003}))
        self.assertIsNotNone(stopping_handler.history)
        self.assertEqual((2,), stopping_handler.history.shape)

        stopping_handler.reset()
        self.assertEqual((0,), stopping_handler.history.shape)


class DefaultEarlyStoppingHandlerValLossTest(unittest.TestCase, LossBasedEarlyStoppingHandlerTest):

    def get_monitor(self):
        return 'val_loss'


class DefaultEarlyStoppingHandlerTrainLossTest(unittest.TestCase, LossBasedEarlyStoppingHandlerTest):

    def get_monitor(self):
        return 'train_loss'
