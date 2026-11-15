import unittest

from unittest import mock
from unittest.mock import Mock, patch

import numpy as np
import pytest
from packaging.version import Version, parse
from scipy.sparse import csr_matrix, issparse

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.training.early_stopping import (
    EarlyStopping,
    EarlyStoppingOrCondition,
    NoopEarlyStopping,
)
from small_text.training.metrics import Metric
from small_text.training.model_selection import ModelSelection, NoopModelSelection

try:
    import torch

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    from small_text.integrations.pytorch.classifiers.base import AMPArguments
    from small_text.integrations.pytorch.classifiers.lstm import LSTMClassifier
    from tests.utils.datasets import random_text_classification_dataset
except PytorchNotFoundError:
    pass


class _LSTMClassifierTest(object):

    def _get_dataset(self, num_samples=20, num_classes=4):
        return random_text_classification_dataset(num_samples,
                                                  max_length=12,
                                                  num_classes=num_classes,
                                                  multi_label=self.multi_label)

    def _get_classifier(self, num_classes=4, **kwargs):
        classifier_kwargs = {
            'max_seq_len': 12,
            'hidden_size': 5,
            'num_epochs': 1,
            'show_progress_bar': False,
        }
        classifier_kwargs.update(kwargs)
        return LSTMClassifier(num_classes,
                              multi_label=self.multi_label,
                              embedding_matrix=torch.FloatTensor(np.random.rand(10, 8)),
                              **classifier_kwargs)

    def test_fit_with_scheduler_but_without_optimizer(self):
        classifier = self._get_classifier()

        with self.assertRaisesRegex(ValueError, 'You must also pass an optimizer'):
            classifier.fit(self._get_dataset(), scheduler=Mock())

    def test_fit_and_predict(self):
        dataset = self._get_dataset()
        classifier = self._get_classifier()
        classifier.fit(dataset)

        with mock.patch.object(classifier.model, 'eval', wraps=classifier.model.eval) as model_eval_spy, \
                mock.patch.object(classifier.model, 'train', wraps=classifier.model.train) as model_train_spy:
            prediction = classifier.predict(dataset)

        self.assertEqual(len(dataset), prediction.shape[0])
        model_eval_spy.assert_called()
        model_train_spy.assert_called_once_with(False)
        if self.multi_label:
            self.assertTrue(issparse(prediction))
            self.assertEqual(prediction.dtype, np.int64)
        else:
            self.assertIsInstance(prediction, np.ndarray)
            self.assertTrue(np.all((prediction >= 0) & (prediction < classifier.num_classes)))

    def test_fit_and_predict_with_sample_weights(self):
        dataset = self._get_dataset()
        weights = np.random.rand(len(dataset)) + 1e-8
        classifier = self._get_classifier()
        classifier.fit(dataset, weights=weights)

        prediction = classifier.predict(dataset)
        self.assertEqual(len(dataset), prediction.shape[0])
        if self.multi_label:
            self.assertTrue(issparse(prediction))
        else:
            self.assertTrue(np.all((prediction >= 0) & (prediction < classifier.num_classes)))

    def test_fit_and_predict_proba_dropout_sampling(self):
        train_set = self._get_dataset(num_samples=20)
        test_set = self._get_dataset(num_samples=8)
        classifier = self._get_classifier()
        classifier.fit(train_set)

        probabilities = classifier.predict_proba(test_set)
        self.assertEqual((len(test_set), classifier.num_classes), probabilities.shape)
        if self.multi_label:
            self.assertIsInstance(probabilities, csr_matrix)
        else:
            self.assertIsInstance(probabilities, np.ndarray)

        dropout_samples = 3
        probabilities = classifier.predict_proba(test_set, dropout_sampling=dropout_samples)
        self.assertEqual((len(test_set), dropout_samples, classifier.num_classes), probabilities.shape)
        self.assertIsInstance(probabilities, np.ndarray)

    def test_fit_and_validate(self):
        dataset = self._get_dataset()
        classifier = self._get_classifier()
        classifier.fit(dataset)

        with mock.patch.object(classifier.model, 'eval', wraps=classifier.model.eval) as model_eval_spy, \
                mock.patch.object(classifier.model, 'train', wraps=classifier.model.train) as model_train_spy:
            loss, accuracy = classifier.validate(dataset)

        self.assertGreaterEqual(loss, 0)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        model_eval_spy.assert_called()
        model_train_spy.assert_called_once_with(False)

    def test_fit_with_early_stopping(self):
        dataset = self._get_dataset()
        classifier = self._get_classifier(num_epochs=2)
        early_stopping = EarlyStopping(Metric('val_loss'))

        with mock.patch.object(early_stopping,
                               'check_early_stop',
                               wraps=early_stopping.check_early_stop) as check_early_stop_spy:
            classifier.fit(dataset[:10], validation_set=dataset[10:], early_stopping=early_stopping)

        self.assertEqual(2, check_early_stop_spy.call_count)
        for epoch, call in enumerate(check_early_stop_spy.call_args_list, start=1):
            self.assertEqual(epoch, call.args[0])
            self.assertIn('train_acc', call.args[1])

    def test_fit_uses_default_early_stopping(self):
        dataset = self._get_dataset()
        classifier = self._get_classifier()

        with patch.object(classifier, '_fit_main', wraps=classifier._fit_main) as fit_main_spy:
            classifier.fit(dataset[:10], validation_set=dataset[10:])

        early_stopping = fit_main_spy.call_args.args[3]
        self.assertIsInstance(early_stopping, EarlyStoppingOrCondition)
        self.assertEqual(['val_loss', 'train_acc'],
                         [handler.metric.name for handler in early_stopping.early_stopping_handlers])

    def test_fit_can_disable_early_stopping(self):
        dataset = self._get_dataset()
        classifier = self._get_classifier()

        with patch.object(classifier, '_fit_main', wraps=classifier._fit_main) as fit_main_spy:
            classifier.fit(dataset[:10], validation_set=dataset[10:], early_stopping='none')

        self.assertIsInstance(fit_main_spy.call_args.args[3], NoopEarlyStopping)

    def test_fit_uses_noop_model_selection_by_default(self):
        dataset = self._get_dataset()
        classifier = self._get_classifier()

        with patch.object(classifier, '_fit_main', wraps=classifier._fit_main) as fit_main_spy:
            classifier.fit(dataset[:10], validation_set=dataset[10:])

        self.assertIsInstance(fit_main_spy.call_args.args[4], NoopModelSelection)

    def test_fit_uses_noop_model_selection_when_none_is_passed(self):
        dataset = self._get_dataset()
        classifier = self._get_classifier()

        with patch.object(classifier, '_fit_main', wraps=classifier._fit_main) as fit_main_spy:
            classifier.fit(dataset[:10], validation_set=dataset[10:], model_selection=None)

        self.assertIsInstance(fit_main_spy.call_args.args[4], NoopModelSelection)

    def test_fit_accepts_model_selection(self):
        dataset = self._get_dataset()
        classifier = self._get_classifier()
        model_selection = ModelSelection()

        with patch.object(classifier, '_fit_main', wraps=classifier._fit_main) as fit_main_spy:
            classifier.fit(dataset[:10], validation_set=dataset[10:], model_selection=model_selection)

        self.assertIs(fit_main_spy.call_args.args[4], model_selection)

    def test_fit_with_optimizer_and_scheduler(self):
        dataset = self._get_dataset()
        classifier = self._get_classifier()
        classifier.fit(dataset)

        parameters = [parameter for parameter in classifier.model.parameters() if parameter.requires_grad]
        optimizer = AdamW(parameters, lr=5e-5)
        scheduler = LambdaLR(optimizer, lambda _: 1)

        with patch.object(classifier, '_train', wraps=classifier._train) as train_spy:
            classifier.fit(dataset, optimizer=optimizer, scheduler=scheduler)

        self.assertEqual(1, train_spy.call_count)
        self.assertIs(optimizer, train_spy.call_args.args[5])
        self.assertIs(scheduler, train_spy.call_args.args[6])

    def test_train_loss_is_average_per_example(self):
        dataset = self._get_dataset(num_samples=12)
        classifier = self._get_classifier(train_batch_size=5)
        classifier.fit(dataset)

        optimizer = torch.optim.SGD(classifier.model.parameters(), lr=0.0)
        scheduler = LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.amp.GradScaler(enabled=False)
        amp_args = AMPArguments(use_amp=False, device_type='cpu', dtype=torch.bfloat16)

        expected_loss = self._get_expected_loss(classifier, dataset)
        encountered_loss, _ = classifier._train_func(dataset, None, optimizer, scheduler, scaler, amp_args)
        self.assertAlmostEqual(expected_loss, encountered_loss, places=5)

    def test_validation_loss_is_average_per_example(self):
        dataset = self._get_dataset(num_samples=12)
        classifier = self._get_classifier(predict_batch_size=5)
        classifier.fit(dataset)

        expected_loss = self._get_expected_loss(classifier, dataset)
        encountered_loss, _ = classifier.validate(dataset)
        self.assertAlmostEqual(expected_loss, encountered_loss, places=5)

    @staticmethod
    def _get_expected_loss(classifier, dataset):
        collate_fn = classifier._create_collate_fn()
        text, cls, weights = collate_fn(dataset.data)

        classifier.model.eval()
        with torch.no_grad():
            output = classifier.model(text.to(classifier.device))
            loss = classifier._compute_loss(cls.to(classifier.device), output)
            return (loss * weights.to(classifier.device)).mean().item()


@pytest.mark.pytorch
class LSTMClassifierSingleLabelTest(unittest.TestCase, _LSTMClassifierTest):

    def setUp(self):
        self.multi_label = False


@pytest.mark.pytorch
class LSTMClassifierMultiLabelTest(unittest.TestCase, _LSTMClassifierTest):

    def setUp(self):
        self.multi_label = True


@pytest.mark.pytorch
class LSTMCompileTest(unittest.TestCase):

    def _get_classifier(self, compile_model):
        return LSTMClassifier(4,
                              embedding_matrix=torch.FloatTensor(np.random.rand(10, 8)),
                              max_seq_len=12,
                              hidden_size=5,
                              compile_model=compile_model)

    def test_initialize_with_compile_enabled_on_pytorch_v2(self):
        if parse(torch.__version__) >= Version('2.0.0'):
            classifier = self._get_classifier(compile_model=True)
            with patch('torch.__version__', new='2.0.0'), \
                    patch('torch.compile', wraps=torch.compile) as compile_spy:
                classifier.initialize()
            compile_spy.assert_called_once()

    def test_initialize_with_compile_disabled_on_pytorch_v2(self):
        if parse(torch.__version__) >= Version('2.0.0'):
            classifier = self._get_classifier(compile_model=False)
            with patch('torch.__version__', new='2.0.0'), \
                    patch('torch.compile', wraps=torch.compile) as compile_spy:
                classifier.initialize()
            compile_spy.assert_not_called()

    def test_initialize_with_compile_enabled_on_pytorch_v1(self):
        classifier = self._get_classifier(compile_model=True)
        with patch.object(torch, 'compile', return_value=None):
            with patch('torch.__version__', new='1.9.0'), \
                    patch('torch.compile', wraps=torch.compile) as compile_spy:
                classifier.initialize()
        compile_spy.assert_not_called()

    def test_initialize_with_compile_disabled_on_pytorch_v1(self):
        classifier = self._get_classifier(compile_model=False)
        with patch.object(torch, 'compile', return_value=None):
            with patch('torch.__version__', new='1.9.0'), \
                    patch('torch.compile', wraps=torch.compile) as compile_spy:
                classifier.initialize()
        compile_spy.assert_not_called()


@pytest.mark.pytorch
class LSTMClassifierAMPArgumentsTest(unittest.TestCase):

    def test_with_no_amp_args_configured(self):
        classifier = LSTMClassifier(3,
                                    device='cpu',
                                    embedding_matrix=torch.FloatTensor(np.random.rand(10, 8)),
                                    max_seq_len=12,
                                    hidden_size=5)

        self.assertFalse(classifier.amp_args.use_amp)
        self.assertEqual('cpu', classifier.amp_args.device_type)
        self.assertEqual(torch.bfloat16, classifier.amp_args.dtype)

        classifier.initialize()
        self.assertFalse(classifier.amp_args.use_amp)
        self.assertEqual('cpu', classifier.amp_args.device_type)
        self.assertEqual(torch.bfloat16, classifier.amp_args.dtype)

    def test_amp_args_are_disabled_for_a_cpu_model(self):
        amp_args = AMPArguments(use_amp=True, device_type='cuda', dtype=torch.float16)
        classifier = LSTMClassifier(3,
                                    device='cpu',
                                    embedding_matrix=torch.FloatTensor(np.random.rand(10, 8)),
                                    max_seq_len=12,
                                    hidden_size=5,
                                    amp_args=amp_args)

        self.assertFalse(classifier.amp_args.use_amp)
        self.assertEqual('cuda', classifier.amp_args.device_type)
        self.assertEqual(torch.float16, classifier.amp_args.dtype)

        classifier.initialize()
        self.assertFalse(classifier.amp_args.use_amp)
        self.assertEqual('cuda', classifier.amp_args.device_type)
        self.assertEqual(torch.float16, classifier.amp_args.dtype)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA to enable AMP')
    def test_amp_arguments_are_used_for_cuda_model(self):
        amp_args = AMPArguments(use_amp=True, device_type='cuda', dtype=torch.float16)
        classifier = LSTMClassifier(3,
                                    device='cuda',
                                    embedding_matrix=torch.FloatTensor(np.random.rand(10, 8)),
                                    max_seq_len=12,
                                    hidden_size=5,
                                    amp_args=amp_args)
        classifier.initialize()

        self.assertTrue(classifier.amp_args.use_amp)
        self.assertEqual('cuda', classifier.amp_args.device_type)
        self.assertEqual(torch.float16, classifier.amp_args.dtype)
