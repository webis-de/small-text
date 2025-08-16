import unittest
import pytest
import numpy as np

from unittest.mock import create_autospec, patch
from scipy.sparse import issparse

from small_text.data.datasets import TextDataset
from small_text.exceptions import UnsupportedOperationException
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from sklearn.utils.validation import check_is_fitted
from tests.utils.datasets import twenty_news_text

try:
    import torch

    from small_text.integrations.pytorch.classifiers.base import AMPArguments
    from small_text.integrations.transformers.classifiers.factories import (
        SetFitClassification,
        SetFitClassificationFactory
    )
    from small_text.integrations.transformers.classifiers.setfit import (
        SetFitModelArguments
    )
except (ImportError, PytorchNotFoundError):
    pass


class _ClassificationTest(object):

    def test_fit_and_predict(self):
        classification_kwargs = {
            'use_differentiable_head': self.use_differentiable_head,
            'multi_label': self.multi_label
        }
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2',
                                                 output_dir='/tmp')
        clf_factory = SetFitClassificationFactory(
            setfit_model_args,
            self.num_classes,
            classification_kwargs=classification_kwargs)

        train_set = twenty_news_text(30, num_classes=self.num_classes, multi_label=self.multi_label)

        clf = clf_factory.new()

        with patch.object(clf, 'initialize', wraps=clf.initialize) as initialize_spy:
            clf.fit(train_set)

        initialize_spy.assert_called_once()

        test_set = twenty_news_text(30, num_classes=self.num_classes, multi_label=self.multi_label)
        y_pred = clf.predict(test_set)

        if self.multi_label:
            self.assertEqual((30, self.num_classes), y_pred.shape)
            self.assertTrue(issparse(y_pred))
            self.assertEqual(y_pred.dtype, np.int64)
            self.assertTrue(np.logical_or(y_pred.indices.all() >= 0, y_pred.indices.all() <= 3))
        else:
            self.assertEqual((30,), y_pred.shape)
            self.assertTrue(isinstance(y_pred, np.ndarray))
            self.assertTrue(np.all([isinstance(y, np.int64) for y in y_pred]))
            self.assertTrue(np.logical_or(y_pred.all() >= 0, y_pred.all() <= 3))

        y_pred_proba = clf.predict_proba(test_set)
        self.assertEqual((30, self.num_classes), y_pred_proba.shape)
        self.assertTrue(isinstance(y_pred_proba, np.ndarray))
        self.assertTrue(np.all([isinstance(y, np.float64) for row in y_pred_proba for y in row]))
        self.assertTrue(np.logical_or(y_pred_proba.all() >= 0.0, y_pred_proba.all() <= 1.0))

    def test_fit_and_predict_proba_dropout(self, dropout_sampling=3):
        classification_kwargs = {
            'use_differentiable_head': self.use_differentiable_head,
            'multi_label': self.multi_label
        }
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2',
                                                 output_dir='/tmp')
        clf_factory = SetFitClassificationFactory(
            setfit_model_args,
            self.num_classes,
            classification_kwargs=classification_kwargs)

        train_set = twenty_news_text(30, num_classes=self.num_classes, multi_label=self.multi_label)
        test_set = twenty_news_text(30, num_classes=self.num_classes, multi_label=self.multi_label)

        clf = clf_factory.new()

        clf.fit(train_set)

        # check Module.train()/.eval() works
        clf.model.model_body.train()
        self.assertTrue(clf.model.model_body.training)
        clf.model.model_body.eval()
        self.assertFalse(clf.model.model_body.training)

        y_pred_proba = clf.predict_proba(test_set)
        self.assertSequenceEqual((len(test_set), self.num_classes), y_pred_proba.shape)
        self.assertTrue(isinstance(y_pred_proba, np.ndarray))
        self.assertTrue(np.all([isinstance(p, np.float64) for pred in y_pred_proba for p in pred]))

        y_pred_proba = clf.predict_proba(test_set, dropout_sampling=dropout_sampling)
        self.assertSequenceEqual((len(test_set), dropout_sampling, self.num_classes), y_pred_proba.shape)
        self.assertTrue(isinstance(y_pred_proba, np.ndarray))
        self.assertTrue(np.all([isinstance(p, np.float64) for pred in y_pred_proba
                                for sample in pred for p in sample]))

        # check Module.train()/.eval() **still** (!) works
        clf.model.model_body.train()
        self.assertTrue(clf.model.model_body.training)
        clf.model.model_body.eval()
        self.assertFalse(clf.model.model_body.training)

    def test_fit_and_validate(self):
        device = 'cuda:0'
        classification_kwargs = {
            'use_differentiable_head': self.use_differentiable_head,
            'multi_label': self.multi_label,
            'device': device
        }
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2',
                                                 output_dir='/tmp')
        clf_factory = SetFitClassificationFactory(
            setfit_model_args,
            self.num_classes,
            classification_kwargs=classification_kwargs)

        train_set = twenty_news_text(30, num_classes=self.num_classes, multi_label=self.multi_label)

        clf = clf_factory.new()

        # required to verify that initializes() and to() are called
        original_initialize = clf.initialize

        def mocked_initialize():
            model = original_initialize()
            model.model_body.to = create_autospec(model.model_body.to, spec_set=True, wraps=model.model_body.to)
            return model

        with patch.object(clf, 'initialize', wraps=mocked_initialize) as initialize_spy:
            clf.fit(train_set)

            initialize_spy.assert_called_once()

            self.assertTrue(clf.model.model_body.to.call_count >= 2)
            # our call is the first
            self.assertEqual(1, len(clf.model.model_body.to.call_args_list[0].args))
            self.assertEqual(device, clf.model.model_body.to.call_args_list[0].args[0])

        valid_set = twenty_news_text(10, num_classes=self.num_classes, multi_label=self.multi_label)

        if self.use_differentiable_head:
            with self.assertRaises(NotImplementedError):
                clf.validate(valid_set)
        else:
            with self.assertRaises(UnsupportedOperationException):
                clf.validate(valid_set)

    def test_fit_with_non_default_settings(self):
        # in particularly we test max_length and mini_batch_size here
        mini_batch_size = 8
        max_length = 32
        device = 'cuda:0'
        classification_kwargs = {
            'use_differentiable_head': self.use_differentiable_head,
            'multi_label': self.multi_label,
            'device': device,
            'mini_batch_size': mini_batch_size,
            'max_length': max_length
        }
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2')
        clf_factory = SetFitClassificationFactory(
            setfit_model_args,
            self.num_classes,
            classification_kwargs=classification_kwargs)

        train_set = twenty_news_text(30, num_classes=self.num_classes, multi_label=self.multi_label)

        clf = clf_factory.new()

        with patch('small_text.integrations.transformers.classifiers.setfit.Trainer',
                   autospec=True, create=True) as trainer_mock:
            clf.fit(train_set)

            self.assertEqual(1, trainer_mock.call_count)

            train_mock = trainer_mock.return_value.train
            self.assertEqual(1, train_mock.call_count)
            self.assertIsNotNone(train_mock.call_args_list[0].kwargs['args'])

    def test_fit_prevent_fixed_seed(self):
        ds = twenty_news_text(10, num_classes=self.num_classes, multi_label=self.multi_label)
        num_classes = 5

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                                 output_dir='/tmp')

        with patch('setfit.trainer.set_seed') as set_seed_mock:
            clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label)

            clf.fit(ds)
            self.assertEqual(1, set_seed_mock.call_count)
            first_seed = set_seed_mock.call_args_list[0][0]

            clf.fit(ds)
            self.assertEqual(2, set_seed_mock.call_count)
            second_seed = set_seed_mock.call_args_list[1][0]

            self.assertNotEqual(first_seed, second_seed)


@pytest.mark.pytorch
@pytest.mark.optional
class SetFitClassificationRegressionHeadSingleLabelTest(unittest.TestCase, _ClassificationTest):

    def setUp(self):
        self.num_classes = 3
        self.multi_label = False
        self.use_differentiable_head = False


@pytest.mark.pytorch
@pytest.mark.optional
class SetFitClassificationDifferentiableHeadSingleLabelTest(unittest.TestCase, _ClassificationTest):

    def setUp(self):
        self.num_classes = 3
        self.multi_label = False
        self.use_differentiable_head = True


@pytest.mark.pytorch
@pytest.mark.optional
class SetFitClassificationRegressionHeadMultiLabelTest(unittest.TestCase, _ClassificationTest):

    def setUp(self):
        self.num_classes = 3
        self.multi_label = True
        self.use_differentiable_head = False


@pytest.mark.pytorch
@pytest.mark.optional
class SetFitClassificationDifferentiableHeadMultiLabelTest(unittest.TestCase, _ClassificationTest):

    def setUp(self):
        self.num_classes = 3
        self.multi_label = False
        self.use_differentiable_head = True


@pytest.mark.pytorch
@pytest.mark.optional
class SetFitClassificationAMPArgumentsTest(unittest.TestCase):

    def test_with_no_amp_args_configured(self):
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2')
        clf = SetFitClassification(setfit_model_args, 3)

        amp_args = clf.amp_args
        self.assertIsNotNone(amp_args)
        self.assertFalse(amp_args.use_amp)
        self.assertEqual('cpu', clf.amp_args.device_type)
        self.assertEqual(torch.bfloat16, clf.amp_args.dtype)

        clf.model = clf.initialize()
        # TODO: model is by default on the GPU; different default behavior than the other classifiers
        clf.model = clf.model.to('cpu')
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
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2')
        clf = SetFitClassification(setfit_model_args, 3, amp_args=amp_args)

        amp_args = clf.amp_args
        self.assertIsNotNone(amp_args)
        self.assertFalse(amp_args.use_amp)
        self.assertEqual('cuda', clf.amp_args.device_type)
        self.assertEqual(torch.float16, clf.amp_args.dtype)

        clf.model = clf.initialize()
        # TODO: model is by default on the GPU; different default behavior than the other classifiers
        clf.model = clf.model.to('cpu')
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
