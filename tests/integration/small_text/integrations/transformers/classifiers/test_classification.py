import unittest
import pytest
import warnings

import numpy as np

from packaging.version import parse, Version
from unittest import mock
from unittest.mock import patch, Mock

from scipy.sparse import issparse, csr_matrix

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.training.early_stopping import EarlyStopping, NoopEarlyStopping, EarlyStoppingOrCondition
from small_text.training.metrics import Metric
from small_text.training.model_selection import ModelSelection, NoopModelSelection

try:
    import torch
    import torch.nn.functional as F

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    from small_text.integrations.pytorch.classifiers.base import AMPArguments
    from small_text.integrations.transformers import (
        TransformerBasedClassification,
        TransformerModelArguments
    )

    from tests.utils.datasets import random_transformer_dataset
except (ImportError, PytorchNotFoundError):
    # prevent "NameError: name 'TransformerBasedClassification' is not defined" in patch.object
    class TransformerBasedClassification(object):
        pass


class _TransformerBasedClassificationTest(object):

    def _get_dataset(self, num_samples=100, num_classes=4):
        return random_transformer_dataset(num_samples, max_length=60, num_classes=num_classes,
                                          multi_label=self.multi_label)

    @patch.object(TransformerBasedClassification, '_train')
    @patch.object(TransformerBasedClassification, '_perform_model_selection')
    def test_fit(self, perform_model_selection_mock, fake_train):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args,
                                             4,
                                             multi_label=self.multi_label,
                                             num_epochs=1)

        train_set = self._get_dataset(num_samples=20)
        clf.fit(train_set)

        # basically tests _get_layer_params for now

        fake_train.assert_called()
        perform_model_selection_mock.assert_not_called()

    def test_fit_with_class_weight(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args,
                                             4,
                                             multi_label=self.multi_label,
                                             class_weight='balanced',
                                             num_epochs=1)

        train_set = self._get_dataset(num_samples=20)
        clf.fit(train_set)
        self.assertIsNotNone(clf.class_weights_)
        self.assertIsNotNone(clf.model)

    def test_fit_with_sample_weight(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args,
                                             4,
                                             multi_label=self.multi_label,
                                             num_epochs=1)

        train_set = self._get_dataset(num_samples=20)
        weights = np.random.randn(len(train_set))
        weights = weights - weights.min() + 1e-8

        clf.fit(train_set, weights=weights)
        self.assertIsNotNone(clf.model)

    def test_fit_with_scheduler_but_without_optimizer(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args,
                                             4,
                                             multi_label=self.multi_label,
                                             class_weight='balanced',
                                             num_epochs=1)

        train_set = self._get_dataset(num_samples=20)

        scheduler = Mock()

        with self.assertRaisesRegex(ValueError, 'You must also pass an optimizer'):
            clf.fit(train_set, scheduler=scheduler)

    def test_fit_and_predict(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args,
                                             4,
                                             multi_label=self.multi_label,
                                             class_weight='balanced',
                                             num_epochs=1)

        train_set = self._get_dataset(num_samples=20)
        test_set = self._get_dataset(num_samples=10)

        clf.fit(train_set)

        with mock.patch.object(clf.model, 'eval', wraps=clf.model.eval) as model_eval_spy, \
                mock.patch.object(clf.model, 'train', wraps=clf.model.train) as model_train_spy:

            y_pred = clf.predict(test_set)

            model_eval_spy.assert_called()
            model_train_spy.assert_called_once_with(False)

        if self.multi_label:
            self.assertTrue(issparse(y_pred))
            self.assertEqual(y_pred.dtype, np.int64)
            self.assertTrue(np.logical_or(y_pred.indices.all() >= 0, y_pred.indices.all() <= 3))
        else:
            self.assertTrue(isinstance(y_pred, np.ndarray))
            self.assertTrue(np.all([isinstance(y, np.int64) for y in y_pred]))
            self.assertTrue(np.logical_or(y_pred.all() >= 0, y_pred.all() <= 3))

    def test_fit_and_predict_proba_dropout_sampling(self, num_classes=4, dropout_sampling=3):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args,
                                             num_classes,
                                             multi_label=self.multi_label,
                                             class_weight='balanced',
                                             num_epochs=1)

        train_set = self._get_dataset(num_samples=20)
        test_set = self._get_dataset(num_samples=10)

        clf.fit(train_set)

        y_pred_proba = clf.predict_proba(test_set)
        self.assertSequenceEqual((len(test_set), num_classes), y_pred_proba.shape)
        if self.multi_label:
            self.assertTrue(isinstance(y_pred_proba, csr_matrix))
            self.assertTrue(np.all([isinstance(p, np.float64) for p in y_pred_proba.data]))
        else:
            self.assertTrue(isinstance(y_pred_proba, np.ndarray))
            self.assertTrue(np.all([isinstance(p, np.float64) for pred in y_pred_proba for p in pred]))

        y_pred_proba = clf.predict_proba(test_set, dropout_sampling=dropout_sampling)
        self.assertSequenceEqual((len(test_set), dropout_sampling, num_classes), y_pred_proba.shape)
        self.assertTrue(isinstance(y_pred_proba, np.ndarray))
        self.assertTrue(np.all([isinstance(p, np.float64) for pred in y_pred_proba
                                for sample in pred for p in sample]))

    def test_fit_validate(self):

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args,
                                             4,
                                             multi_label=self.multi_label,
                                             class_weight='balanced',
                                             num_epochs=1)

        train_set = self._get_dataset(num_samples=20)
        valid_set = self._get_dataset(num_samples=5)

        clf.fit(train_set)

        with mock.patch.object(clf.model, 'eval', wraps=clf.model.eval) as model_eval_spy, \
             mock.patch.object(clf.model, 'train', wraps=clf.model.train) as model_train_spy:

            valid_loss, valid_acc = clf.validate(valid_set)

            model_eval_spy.assert_called()
            model_train_spy.assert_called_once_with(False)

        self.assertTrue(valid_loss >= 0)
        self.assertTrue(0.0 <= valid_acc <= 1.0)

    def test_validate_with_validations_per_epoch(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args,
                                             4,
                                             multi_label=self.multi_label,
                                             num_epochs=1,
                                             mini_batch_size=10,
                                             validations_per_epoch=2)

        train_set = self._get_dataset(num_samples=20)

        with patch.object(clf, 'validate', wraps=clf.validate) as validate_spy, \
                patch('torch.nn.modules.module.Module.eval') as eval_spy, \
                patch('torch.nn.modules.module.Module.train') as train_spy:

            clf.fit(train_set)

            self.assertIsNotNone(clf.model)
            self.assertEqual(2, validate_spy.call_count)
            self.assertEqual(3, train_spy.call_count)
            # once per validate (after the call to validate) plus one call during initialization in from_pretrained()
            self.assertEqual(3, eval_spy.call_count)

    def test_validate_with_validations_per_epoch_too_large(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args,
                                             4,
                                             multi_label=self.multi_label,
                                             num_epochs=2,
                                             mini_batch_size=20,
                                             validations_per_epoch=3)

        train_set = self._get_dataset(num_samples=20)

        with patch.object(clf, 'validate', wraps=clf.validate) as validate_spy, \
                patch('torch.nn.modules.module.Module.eval') as eval_spy, \
                patch('torch.nn.modules.module.Module.train') as train_spy, \
                warnings.catch_warnings(record=True) as w:

            clf.fit(train_set)
            self.assertIsNotNone(clf.model)

            # 2 since we have 2 epochs with 1 batches (and one validate call) each
            self.assertEqual(2, validate_spy.call_count)

            expected_warning = 'validations_per_epoch=3 is greater than the maximum ' \
                               'possible batches of 1'
            found_warning = np.any([
                str(w_.message) == expected_warning and w_.category == RuntimeWarning
                for w_ in w])
            self.assertTrue(found_warning)
            self.assertEqual(4, train_spy.call_count)
            # once per epoch (after the call to validate) plus one call during initialization in from_pretrained()
            self.assertEqual(3, eval_spy.call_count)

    def test_fit_with_early_stopping(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args,
                                                    4,
                                                    multi_label=self.multi_label,
                                                    class_weight='balanced',
                                                    num_epochs=2)

        train_set = self._get_dataset(num_samples=10)
        validation_set = self._get_dataset(num_samples=10)

        early_stopping = EarlyStopping(Metric('val_loss'))

        with mock.patch.object(early_stopping,
                               'check_early_stop',
                               wraps=early_stopping.check_early_stop) as check_early_stop_spy:

            classifier.fit(train_set, validation_set=validation_set, early_stopping=early_stopping)
            self.assertEqual(2, check_early_stop_spy.call_count)
            for i in range(2):
                self.assertEqual(i+1, check_early_stop_spy.call_args_list[i].args[0])
                self.assertTrue('train_acc' in check_early_stop_spy.call_args_list[i].args[1])
                self.assertTrue('train_loss' in check_early_stop_spy.call_args_list[i].args[1])
                self.assertTrue('val_acc' in check_early_stop_spy.call_args_list[i].args[1])
                self.assertTrue('val_loss' in check_early_stop_spy.call_args_list[i].args[1])

    def test_fit_with_early_stopping_with_validations_per_epoch(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args,
                                                    4,
                                                    multi_label=self.multi_label,
                                                    class_weight='balanced',
                                                    num_epochs=2,
                                                    mini_batch_size=5,
                                                    validations_per_epoch=2)

        train_set = self._get_dataset(num_samples=10)
        validation_set = self._get_dataset(num_samples=10)

        early_stopping = EarlyStopping(Metric('val_loss'))

        with mock.patch.object(early_stopping,
                               'check_early_stop',
                               wraps=early_stopping.check_early_stop) as check_early_stop_spy:

            classifier.fit(train_set, validation_set=validation_set, early_stopping=early_stopping)
            # 2 "intermediate" validations with val_acc/vall_los only + one final one per epoch
            self.assertEqual(6, check_early_stop_spy.call_count)
            for i in range(6):
                self.assertEqual(i // 3 + 1, check_early_stop_spy.call_args_list[i].args[0])
                self.assertTrue('val_acc' in check_early_stop_spy.call_args_list[i].args[1])
                self.assertTrue('val_loss' in check_early_stop_spy.call_args_list[i].args[1])
                if (i+1) % 3 == 0:
                    self.assertTrue('train_acc' in check_early_stop_spy.call_args_list[i].args[1])
                    self.assertTrue('train_loss' in check_early_stop_spy.call_args_list[i].args[1])

    def test_fit_with_early_stopping_default(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args,
                                                    4,
                                                    multi_label=self.multi_label,
                                                    class_weight='balanced',
                                                    num_epochs=1)

        train_set = self._get_dataset(num_samples=10)
        validation_set = self._get_dataset(num_samples=10)

        with patch.object(classifier, '_fit_main', wraps=classifier._fit_main) as fit_main_spy:
            classifier.fit(train_set, validation_set=validation_set)

            self.assertEqual(1, fit_main_spy.call_count)
            early_stopping_arg = fit_main_spy.call_args_list[0].args[3]
            self.assertTrue(isinstance(early_stopping_arg, EarlyStoppingOrCondition))
            self.assertEqual(2, len(early_stopping_arg.early_stopping_handlers))

            # this is a quick and dirty test; other values of the early stopping handlers could differ here
            self.assertEqual('val_loss', early_stopping_arg.early_stopping_handlers[0].metric.name)
            self.assertEqual('train_acc', early_stopping_arg.early_stopping_handlers[1].metric.name)

    def test_fit_with_early_stopping_disabled(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args,
                                                    4,
                                                    multi_label=self.multi_label,
                                                    class_weight='balanced',
                                                    num_epochs=1)

        train_set = self._get_dataset(num_samples=10)
        validation_set = self._get_dataset(num_samples=10)

        with patch.object(classifier,
                          '_fit_main',
                          wraps=classifier._fit_main) as fit_main_spy:
            classifier.fit(train_set, validation_set=validation_set, early_stopping='none')

            self.assertEqual(1, fit_main_spy.call_count)
            self.assertTrue(isinstance(fit_main_spy.call_args_list[0].args[3], NoopEarlyStopping))

    def test_fit_with_model_selection_default(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args,
                                                    4,
                                                    multi_label=self.multi_label,
                                                    class_weight='balanced',
                                                    num_epochs=1)

        train_set = self._get_dataset(num_samples=10)
        validation_set = self._get_dataset(num_samples=10)

        with patch.object(classifier, '_fit_main', wraps=classifier._fit_main) as fit_main_spy:
            classifier.fit(train_set, validation_set=validation_set)

            self.assertEqual(1, fit_main_spy.call_count)
            model_selection_arg = fit_main_spy.call_args_list[0].args[4]
            self.assertTrue(isinstance(model_selection_arg, NoopModelSelection))

    def test_fit_with_model_selection_disabled(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args,
                                                    4,
                                                    multi_label=self.multi_label,
                                                    class_weight='balanced',
                                                    num_epochs=1)

        train_set = self._get_dataset(num_samples=10)
        validation_set = self._get_dataset(num_samples=10)

        with patch.object(classifier,
                          '_fit_main',
                          wraps=classifier._fit_main) as fit_main_spy:
            classifier.fit(train_set, validation_set=validation_set, model_selection=None)

            self.assertEqual(1, fit_main_spy.call_count)
            self.assertTrue(isinstance(fit_main_spy.call_args_list[0].args[4], NoopModelSelection))

    def test_fit_with_model_selection_kwarg(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args,
                                                    4,
                                                    multi_label=self.multi_label,
                                                    class_weight='balanced',
                                                    num_epochs=1)

        train_set = self._get_dataset(num_samples=10)
        validation_set = self._get_dataset(num_samples=10)

        with patch.object(classifier,
                          '_fit_main',
                          wraps=classifier._fit_main) as fit_main_spy:
            classifier.fit(train_set, validation_set=validation_set, model_selection=ModelSelection())

            self.assertEqual(1, fit_main_spy.call_count)
            self.assertTrue(isinstance(fit_main_spy.call_args_list[0].args[4], ModelSelection))

    def test_train_loss(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args,
                                                    4,
                                                    multi_label=self.multi_label,
                                                    num_epochs=1)

        ds = self._get_dataset(num_samples=10)

        classifier.fit(ds)

        weights = None

        early_stopping = classifier._get_default_early_stopping(None, classifier.validations_per_epoch)
        model_selection = classifier._get_default_model_selection(None)
        optimizer = torch.optim.SGD(classifier.model.parameters(), lr=0.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.amp.GradScaler(enabled=False)
        amp_args = AMPArguments(use_amp=False, device_type='cpu', dtype=torch.bfloat16)

        expected_loss = self._get_expected_loss(classifier, ds)

        encountered_loss, _, valid_loss, _, _ = classifier._train_loop_process_batches(1,
                                                                                       ds,
                                                                                       ds,
                                                                                       weights,
                                                                                       early_stopping,
                                                                                       model_selection,
                                                                                       optimizer,
                                                                                       scheduler,
                                                                                       scaler,
                                                                                       amp_args)
        self.assertAlmostEqual(expected_loss, encountered_loss, places=5)
        self.assertAlmostEqual(expected_loss, valid_loss, places=5)

    def test_validation_loss(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args,
                                                    4,
                                                    multi_label=self.multi_label,
                                                    num_epochs=1)

        ds = self._get_dataset(num_samples=10)

        classifier.fit(ds)

        expected_loss = self._get_expected_loss(classifier, ds)
        encountered_loss, _ = classifier.validate(ds)
        self.assertAlmostEqual(expected_loss, encountered_loss, places=5)

    def _get_expected_loss(self, clf, ds, weights=None):
        collate_fn = clf._create_collate_fn(use_sample_weights=weights is not None)
        data = ds.data
        if weights is not None:
            data = [d + (weights[i],) for i, d in enumerate(data)]
        x, _, cls, weights = collate_fn(data)

        clf.model.eval()
        with torch.no_grad():
            output = clf.model(x.to(clf.device)).logits
            target = F.one_hot(cls, 2).float() if clf.num_classes == 2 else cls
            loss = clf.criterion(output, target.to(clf.device)) * weights.to(clf.device)

            return loss.mean().item()


@pytest.mark.pytorch
class TransformerBasedClassificationSingleLabelTest(unittest.TestCase,
                                                    _TransformerBasedClassificationTest):

    def setUp(self):
        self.multi_label = False


@pytest.mark.pytorch
class TransformerBasedClassificationMultiLabelTest(unittest.TestCase,
                                                   _TransformerBasedClassificationTest):

    def setUp(self):
        self.multi_label = True

    def test_predict_respects_multi_label_threshold(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args, 3, multi_label=True)

        test_set = self._get_dataset(num_samples=10, num_classes=3)
        probabilities = csr_matrix(
            np.tile([[0.6, 0.7, 0.8]], (len(test_set), 1))
        )

        with mock.patch.object(clf, 'predict_proba', return_value=probabilities):
            predictions = clf.predict(test_set, multi_label_threshold=0.75)

        np.testing.assert_array_equal(
            predictions.toarray(),
            np.tile([[0, 0, 1]], (len(test_set), 1))
        )




@pytest.mark.pytorch
class CompileTest(unittest.TestCase):

    def test_initialize_with_pytorch_geq_v2_and_compile_enabled(self):

        if parse(torch.__version__) >= Version('2.0.0'):
            model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base', compile_model=True)
            classifier = TransformerBasedClassification(model_args,
                                                        4,
                                                        num_epochs=1)

            with patch('torch.__version__', new='2.0.0'), \
                    patch('torch.compile', wraps=torch.compile) as compile_spy:
                classifier.initialize()
                compile_spy.assert_called()

    def test_initialize_with_pytorch_geq_v2_and_compile_disabled(self):

        if parse(torch.__version__) >= Version('2.0.0'):
            model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
            classifier = TransformerBasedClassification(model_args,
                                                        4,
                                                        class_weight='balanced',
                                                        num_epochs=1)

            with patch('torch.__version__', new='2.0.0'), \
                    patch('torch.compile', wraps=torch.compile) as compile_spy:
                classifier.initialize()
                compile_spy.assert_not_called()

    def test_initialize_with_pytorch_lesser_v2_and_compile_enabled(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base', compile_model=True)
        classifier = TransformerBasedClassification(model_args,
                                                    4,
                                                    num_epochs=1)

        with patch.object(torch, 'compile', return_value=None):
            with patch('torch.__version__', new='1.9.0'), \
                    patch('torch.compile', wraps=torch.compile) as compile_spy:
                classifier.initialize()
                compile_spy.assert_not_called()

    def test_initialize_with_pytorch_lesser_v2_and_compile_disabled(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args,
                                                    4,
                                                    num_epochs=1)

        with patch.object(torch, 'compile', return_value=None):
            with patch('torch.__version__', new='1.9.0'), \
                    patch('torch.compile', wraps=torch.compile) as compile_spy:
                classifier.initialize()
                compile_spy.assert_not_called()


@pytest.mark.pytorch
class TransformerBasedClassificationAMPArgumentsTest(unittest.TestCase):

    def test_with_no_amp_args_configured(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args, 3)

        amp_args = clf.amp_args
        self.assertIsNotNone(amp_args)
        self.assertFalse(amp_args.use_amp)
        self.assertEqual('cpu', clf.amp_args.device_type)
        self.assertEqual(torch.bfloat16, clf.amp_args.dtype)

        clf.initialize()
        amp_args = clf.amp_args
        self.assertIsNotNone(amp_args)
        self.assertFalse(amp_args.use_amp)
        self.assertEqual('cpu', clf.amp_args.device_type)
        self.assertEqual(torch.bfloat16, clf.amp_args.dtype)

        clf.model = clf.model.to('cuda')
        amp_args = clf.amp_args
        self.assertIsNotNone(amp_args)
        self.assertFalse(amp_args.use_amp)
        self.assertEqual('cuda', clf.amp_args.device_type)
        self.assertEqual(torch.bfloat16, clf.amp_args.dtype)

    def test_with_amp_args_configured(self):
        amp_args = AMPArguments(use_amp=True, device_type='cuda', dtype=torch.float16)
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args, 3, amp_args=amp_args, device='cpu')

        amp_args = clf.amp_args
        self.assertIsNotNone(amp_args)
        self.assertFalse(amp_args.use_amp)
        self.assertEqual('cuda', clf.amp_args.device_type)
        self.assertEqual(torch.float16, clf.amp_args.dtype)

        clf.initialize()
        amp_args = clf.amp_args
        self.assertIsNotNone(amp_args)
        self.assertFalse(amp_args.use_amp)
        self.assertEqual('cuda', clf.amp_args.device_type)
        self.assertEqual(torch.float16, clf.amp_args.dtype)

        clf.model = clf.model.to('cuda')
        amp_args = clf.amp_args
        self.assertIsNotNone(amp_args)
        self.assertTrue(amp_args.use_amp)
        self.assertEqual('cuda', clf.amp_args.device_type)
        self.assertEqual(torch.float16, clf.amp_args.dtype)
