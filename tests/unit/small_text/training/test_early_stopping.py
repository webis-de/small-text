import unittest

from small_text.training.early_stopping import (
    EarlyStopping,
    NoopEarlyStopping,
    EarlyStoppingAndCondition,
    EarlyStoppingOrCondition
)
from small_text.training.metrics import Metric


class NoopStoppingHandlerTest(unittest.TestCase):

    def test_stopping_handler_all(self):
        stopping_handler = NoopEarlyStopping()
        self.assertFalse(stopping_handler.check_early_stop(0, dict()))
        self.assertFalse(stopping_handler.check_early_stop(0, dict()))
        self.assertFalse(stopping_handler.check_early_stop(1, dict()))


class EarlyStoppingTest(unittest.TestCase):

    def test_check_early_stop_with_varying_metrics(self):
        stopping_handler = EarlyStopping(Metric('train_acc'), patience=2)
        check_early_stop = stopping_handler.check_early_stop
        self.assertFalse(check_early_stop(1, {'valid_loss': 0.35, 'train_acc': 0.80}))
        self.assertFalse(check_early_stop(1, {'valid_loss': 0.34}))
        self.assertFalse(check_early_stop(1, {'valid_loss': 0.33}))
        self.assertFalse(check_early_stop(2, {'valid_loss': 0.35, 'train_acc': 0.81}))
        self.assertFalse(check_early_stop(2, {'valid_loss': 0.34}))
        self.assertFalse(check_early_stop(2, {'valid_loss': 0.35}))


class GeneralEarlyStoppingTest(object):

    def test_init_invalid_config(self):
        with self.assertRaisesRegex(ValueError,
                                    'Invalid configuration encountered: '
                                    'Either "patience" or "threshold" must be enabled.'):
            EarlyStopping(self.get_metric(), patience=-1, threshold=-1)

    def test_init_invalid_min_delta(self):
        with self.assertRaisesRegex(ValueError,
                                    'Invalid value encountered: "min_delta" needs to be'):
            EarlyStopping(self.get_metric(), min_delta=-0.01)

    def test_check_early_stop_invalid_epoch(self):
        stopping_handler = EarlyStopping(self.get_metric())
        with self.assertRaisesRegex(ValueError,
                                    'Argument "epoch" must be greater'):
            stopping_handler.check_early_stop(0, {self.get_metric().name: 0.25})


class LossBasedEarlyStoppingTest(object):

    def get_metric(self):
        raise NotImplementedError('get_metric() must be implemented')

    def test_init_default(self):
        stopping_handler = EarlyStopping(self.get_metric())
        self.assertIsNotNone(stopping_handler._history)
        self.assertEqual((0,), stopping_handler._history.shape)
        self.assertEqual(self.get_metric().name, stopping_handler.metric.name)
        self.assertEqual(1e-14, stopping_handler.min_delta)
        self.assertEqual(5, stopping_handler.patience)
        self.assertEqual(0, stopping_handler.threshold)

    def test_check_early_stop_loss_threshold(self):
        stopping_handler = EarlyStopping(self.get_metric(), threshold=0.1,
                                         patience=2)
        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.1}))
        self.assertTrue(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.05}))

    def test_check_early_stop_loss_threshold_zero(self):
        stopping_handler = EarlyStopping(self.get_metric(), threshold=0)

        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.2}))

        stopping_handler = EarlyStopping(self.get_metric(), threshold=0)

        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name:  0.2}))

    def test_check_early_stop_loss_patience_and_stop(self):
        stopping_handler = EarlyStopping(self.get_metric(), patience=2)

        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.065}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.068}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_metric().name: 0.067}))
        self.assertTrue(stopping_handler.check_early_stop(4, {self.get_metric().name: 0.066}))

    def test_check_early_stop_loss_patience_zero(self):
        stopping_handler = EarlyStopping(self.get_metric(), patience=0)

        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.065}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.064}))
        self.assertTrue(stopping_handler.check_early_stop(3, {self.get_metric().name: 0.068}))

    def test_check_early_stop_loss_min_delta(self):
        stopping_handler = EarlyStopping(self.get_metric(), patience=2, min_delta=0)
        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.04}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.035}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_metric().name: 0.033}))
        self.assertFalse(stopping_handler.check_early_stop(4, {self.get_metric().name: 0.031}))

        stopping_handler = EarlyStopping(self.get_metric(), patience=2, min_delta=0.01)
        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.04}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.031}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_metric().name: 0.032}))
        self.assertTrue(stopping_handler.check_early_stop(4, {self.get_metric().name: 0.031}))

    def test_check_early_stop_loss_patience_and_dont_stop(self):
        stopping_handler = EarlyStopping(self.get_metric(), patience=2)

        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.04}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.031}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_metric().name: 0.029}))

    def test_check_early_stop_loss_patience_and_delta_dont_stop(self):
        stopping_handler = EarlyStopping(self.get_metric(), patience=3)

        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.065}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.065}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_metric().name: 0.066}))
        self.assertFalse(stopping_handler.check_early_stop(4, {self.get_metric().name: 0.055}))

    def test_check_early_stop_loss_patience_and_dont_stop_extended(self):
        """Checks if an update of the best index is recognized, i.e. compares to the previous test
        this would stop but does not if a new best index is correctly recognized in epoch 1."""
        stopping_handler = EarlyStopping(self.get_metric(), patience=3)

        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.180}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.193}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_metric().name: 0.160}))
        self.assertFalse(stopping_handler.check_early_stop(4, {self.get_metric().name: 0.170}))

    def test_check_early_stop_with_none_values_in_between(self):
        stopping_handler = EarlyStopping(self.get_metric(), patience=2)
        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.35}))
        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: None}))
        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: None}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.35}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: None}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: None}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.35}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: None}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: None}))
        self.assertTrue(stopping_handler.check_early_stop(3, {self.get_metric().name: 0.35}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_metric().name: None}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_metric().name: None}))


class EarlyStoppingValLossTest(unittest.TestCase,
                               GeneralEarlyStoppingTest,
                               LossBasedEarlyStoppingTest):

    def get_metric(self):
        return Metric('val_loss')


class EarlyStoppingTrainLossTest(unittest.TestCase,
                                 GeneralEarlyStoppingTest,
                                 LossBasedEarlyStoppingTest):

    def get_metric(self):
        return Metric('train_loss')


class AccuracyBasedEarlyStoppingTest(object):

    def get_metric(self):
        raise NotImplementedError('monitor must be implemented')

    def test_init_default(self):
        stopping_handler = EarlyStopping(self.get_metric())
        self.assertIsNotNone(stopping_handler._history)
        self.assertEqual((0,), stopping_handler._history.shape)
        self.assertEqual(self.get_metric().name, stopping_handler.metric.name)
        self.assertEqual(1e-14, stopping_handler.min_delta)
        self.assertEqual(5, stopping_handler.patience)
        self.assertEqual(0, stopping_handler.threshold)

    def test_init_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError,
                                    'Invalid value encountered: \"threshold\" needs to be'):
            EarlyStopping(self.get_metric(), threshold=-0.01)

        with self.assertRaisesRegex(ValueError,
                                    'Invalid value encountered: \"threshold\" needs to be'):
            EarlyStopping(self.get_metric(), threshold=1.01)

    def test_check_early_stop_acc_threshold(self):
        stopping_handler = EarlyStopping(self.get_metric(), threshold=0.9,
                                         patience=2)

        self.assertTrue(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.91}))

        stopping_handler = EarlyStopping(self.get_metric(), threshold=0.9,
                                         patience=2)
        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.80}))
        self.assertTrue(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.91}))

    def test_check_early_stop_acc_threshold_zero(self):
        stopping_handler = EarlyStopping(self.get_metric(), threshold=0)

        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.91}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.91}))

    def test_check_early_stop_acc_patience_and_stop(self):
        stopping_handler = EarlyStopping(self.get_metric(), patience=2)

        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.65}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.64}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_metric().name: 0.63}))
        self.assertTrue(stopping_handler.check_early_stop(4, {self.get_metric().name: 0.65}))

    def test_check_early_stop_acc_patience_zero(self):
        stopping_handler = EarlyStopping(self.get_metric(), patience=0)

        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.65}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.66}))
        self.assertTrue(stopping_handler.check_early_stop(3, {self.get_metric().name: 0.64}))

    def test_check_early_stop_loss_min_delta(self):
        stopping_handler = EarlyStopping(self.get_metric(), patience=2, min_delta=0)
        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.80}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.82}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_metric().name: 0.85}))
        self.assertFalse(stopping_handler.check_early_stop(4, {self.get_metric().name: 0.90}))

        stopping_handler = EarlyStopping(self.get_metric(), patience=2, min_delta=0.01)
        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.89}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.89}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_metric().name: 0.89}))
        self.assertTrue(stopping_handler.check_early_stop(4, {self.get_metric().name: 0.899}))

    def test_check_early_stop_acc_patience_and_dont_stop(self):
        stopping_handler = EarlyStopping(self.get_metric(), patience=2)

        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.70}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.68}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_metric().name: 0.71}))

    def test_check_early_stop_val_acc_patience_and_dont_stop(self):
        stopping_handler = EarlyStopping(self.get_metric(), patience=3)

        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.65}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.65}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_metric().name: 0.66}))
        self.assertFalse(stopping_handler.check_early_stop(4, {self.get_metric().name: 0.55}))

    def test_check_early_stop_val_acc_patience_and_dont_stop_extended(self):
        """Checks if an update of the best index is recognized, i.e. compares to the previous test
        this would stop but does not if a new best index is correctly recognized in epoch 1."""
        stopping_handler = EarlyStopping(self.get_metric(), patience=3)

        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.680}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.693}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_metric().name: 0.660}))
        self.assertFalse(stopping_handler.check_early_stop(4, {self.get_metric().name: 0.670}))

    def test_check_early_stop_with_none_values_in_between(self):
        stopping_handler = EarlyStopping(self.get_metric(), patience=2)
        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: 0.68}))
        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: None}))
        self.assertFalse(stopping_handler.check_early_stop(1, {self.get_metric().name: None}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.68}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: None}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: None}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: 0.68}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: None}))
        self.assertFalse(stopping_handler.check_early_stop(2, {self.get_metric().name: None}))
        self.assertTrue(stopping_handler.check_early_stop(3, {self.get_metric().name: 0.68}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_metric().name: None}))
        self.assertFalse(stopping_handler.check_early_stop(3, {self.get_metric().name: None}))


class EarlyStoppingValAccTest(unittest.TestCase,
                              GeneralEarlyStoppingTest,
                              AccuracyBasedEarlyStoppingTest):

    def get_metric(self):
        return Metric('val_acc', lower_is_better=False)


class EarlyStoppingTrainAccTest(unittest.TestCase,
                                GeneralEarlyStoppingTest,
                                AccuracyBasedEarlyStoppingTest):

    def get_metric(self):
        return Metric('train_acc', lower_is_better=False)


class EarlyStoppingOrConditionTest(unittest.TestCase):

    def test_check_early_stop(self):
        stopping_handler = EarlyStoppingOrCondition([
            EarlyStopping(Metric('val_loss'), patience=2),
            EarlyStopping(Metric('train_acc', lower_is_better=False), patience=5)
        ])

        check_early_stop = stopping_handler.check_early_stop
        self.assertFalse(check_early_stop(1, {'val_loss': 0.07, 'train_acc': 0.68}))
        self.assertFalse(check_early_stop(2, {'val_loss': 0.08, 'train_acc': 0.69}))
        self.assertFalse(check_early_stop(3, {'val_loss': 0.07, 'train_acc': 0.70}))
        self.assertTrue(check_early_stop(4, {'val_loss': 0.07, 'train_acc': 0.70}))

    def test_check_early_stop_no_stop(self):
        stopping_handler = EarlyStoppingOrCondition([
            EarlyStopping(Metric('val_loss'), patience=5),
            EarlyStopping(Metric('train_acc', lower_is_better=False), patience=5)
        ])

        check_early_stop = stopping_handler.check_early_stop
        self.assertFalse(check_early_stop(1, {'val_loss': 0.07, 'train_acc': 0.68}))
        self.assertFalse(check_early_stop(2, {'val_loss': 0.08, 'train_acc': 0.69}))
        self.assertFalse(check_early_stop(3, {'val_loss': 0.07, 'train_acc': 0.70}))
        self.assertFalse(check_early_stop(4, {'val_loss': 0.07, 'train_acc': 0.70}))


class EarlyStoppingAndConditionTest(unittest.TestCase):

    def test_check_early_stop(self):
        stopping_handler = EarlyStoppingAndCondition([
            EarlyStopping(Metric('val_loss'), patience=2),
            EarlyStopping(Metric('train_acc', lower_is_better=False), threshold=0.7)
        ])

        check_early_stop = stopping_handler.check_early_stop
        self.assertFalse(check_early_stop(1, {'val_loss': 0.07, 'train_acc': 0.68}))
        self.assertFalse(check_early_stop(2, {'val_loss': 0.08, 'train_acc': 0.69}))
        self.assertFalse(check_early_stop(3, {'val_loss': 0.07, 'train_acc': 0.70}))
        self.assertTrue(check_early_stop(4, {'val_loss': 0.07, 'train_acc': 0.71}))

    def test_check_early_stop_no_stop(self):
        stopping_handler = EarlyStoppingAndCondition([
            EarlyStopping(Metric('val_loss'), patience=5),
            EarlyStopping(Metric('train_acc', lower_is_better=False), threshold=0.7)
        ])

        check_early_stop = stopping_handler.check_early_stop
        self.assertFalse(check_early_stop(1, {'val_loss': 0.07, 'train_acc': 0.68}))
        self.assertFalse(check_early_stop(2, {'val_loss': 0.08, 'train_acc': 0.69}))
        self.assertFalse(check_early_stop(3, {'val_loss': 0.07, 'train_acc': 0.70}))
        self.assertFalse(check_early_stop(4, {'val_loss': 0.07, 'train_acc': 0.70}))
