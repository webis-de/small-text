import os
import unittest

import pytest
import small_text

from importlib import reload
from packaging.version import parse, Version
from unittest.mock import patch

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.utils.logging import VERBOSITY_QUIET, VERBOSITY_MORE_VERBOSE
from small_text.utils.system import OFFLINE_MODE_VARIABLE

try:
    import torch

    from small_text.integrations.transformers.classifiers.base import (
        ModelLoadingStrategy
    )
    from small_text.integrations.transformers.classifiers.setfit import (
        SetFitClassification,
        SetFitModelArguments
    )
    from tests.utils.datasets import random_text_dataset
except (ModuleNotFoundError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
@pytest.mark.optional
class TestSetFitModelArguments(unittest.TestCase):

    def test_setfit_model_arguments_init(self):
        sentence_transformer_model = 'sentence-transformers/all-MiniLM-L6-v2'
        args = SetFitModelArguments(sentence_transformer_model)
        self.assertEqual(sentence_transformer_model, args.sentence_transformer_model)
        self.assertIsNotNone(args.model_loading_strategy)
        self.assertEqual(ModelLoadingStrategy.DEFAULT, args.model_loading_strategy)
        self.assertFalse(args.compile_model)

    def test_setfit_model_arguments_init_with_model_loading_strategy(self):
        sentence_transformer_model = 'sentence-transformers/all-MiniLM-L6-v2'
        model_loading_strategy = ModelLoadingStrategy.ALWAYS_LOCAL
        args = SetFitModelArguments(sentence_transformer_model,
                                    model_loading_strategy=model_loading_strategy)
        self.assertEqual(sentence_transformer_model, args.sentence_transformer_model)
        self.assertIsNotNone(args.model_loading_strategy)
        self.assertEqual(model_loading_strategy, args.model_loading_strategy)
        self.assertFalse(args.compile_model)

    def test_transformer_model_arguments_init_with_env_override(self):
        with patch.dict(os.environ, {OFFLINE_MODE_VARIABLE: '1'}):

            # reload TransformerModelArguments so that updated environment variables are read
            reload(small_text.integrations.transformers.classifiers.setfit)
            from small_text.integrations.transformers.classifiers.setfit import SetFitModelArguments

            sentence_transformer_model = 'sentence-transformers/all-MiniLM-L6-v2'
            args = SetFitModelArguments(sentence_transformer_model)

            self.assertEqual(sentence_transformer_model, args.sentence_transformer_model)
            self.assertIsNotNone(args.model_loading_strategy)
            self.assertEqual(ModelLoadingStrategy.ALWAYS_LOCAL, args.model_loading_strategy)
            self.assertFalse(args.compile_model)

    def test_setfit_model_arguments_init_with_compile(self):
        sentence_transformer_model = 'sentence-transformers/all-MiniLM-L6-v2'
        args = SetFitModelArguments(sentence_transformer_model,
                                    compile_model=True)
        self.assertEqual(sentence_transformer_model, args.sentence_transformer_model)
        self.assertIsNotNone(args.model_loading_strategy)
        self.assertEqual(ModelLoadingStrategy.DEFAULT, args.model_loading_strategy)
        self.assertTrue(args.compile_model)


@pytest.mark.pytorch
@pytest.mark.optional
class TestSetFitClassificationKeywordArguments(unittest.TestCase):

    def test_init_with_misplaced_use_differentiable_head_kwarg(self):
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        num_classes = 5

        model_kwargs = {'use_differentiable_head': False}

        with self.assertRaisesRegex(ValueError, 'Invalid keyword argument in model_kwargs'):
            SetFitClassification(setfit_model_args, num_classes, model_kwargs=model_kwargs)

    def test_init_with_misplaced_force_download_kwarg(self):
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        num_classes = 5

        model_kwargs = {'force_download': True}

        with self.assertRaisesRegex(ValueError, 'Invalid keyword argument in model_kwargs'):
            SetFitClassification(setfit_model_args, num_classes, model_kwargs=model_kwargs)

    def test_init_with_misplaced_local_files_only_kwarg(self):
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        num_classes = 5

        model_kwargs = {'local_files_only': True}

        with self.assertRaisesRegex(ValueError, 'Invalid keyword argument in model_kwargs'):
            SetFitClassification(setfit_model_args, num_classes, model_kwargs=model_kwargs)

    def test_init_with_misplaced_batch_size_kwargs(self):
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        num_classes = 5

        trainer_kwargs = {'batch_size': 20}

        with self.assertRaisesRegex(ValueError, 'Invalid keyword argument in trainer_kwargs'):
            SetFitClassification(setfit_model_args, num_classes, trainer_kwargs=trainer_kwargs)

    def test_init_with_misplaced_show_progress_bar_kwargs(self):
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        num_classes = 5

        trainer_kwargs = {'show_progress_bar': True}

        with self.assertRaisesRegex(ValueError, 'Invalid keyword argument in trainer_kwargs'):
            SetFitClassification(setfit_model_args, num_classes, trainer_kwargs=trainer_kwargs)


class _SetFitClassification(object):

    def test_init(self):
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        num_classes = 5

        clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label,
                                   use_differentiable_head=self.use_differentiable_head)

        self.assertEqual(setfit_model_args, clf.setfit_model_args)
        self.assertEqual(num_classes, clf.num_classes)
        self.assertEqual({}, clf.model_kwargs)
        self.assertEqual({}, clf.trainer_kwargs)

        self.assertEqual(self.use_differentiable_head, clf.use_differentiable_head)
        self.assertEqual(32, clf.mini_batch_size)
        self.assertIsNone(clf.device)
        self.assertIsNone(clf.model)
        self.assertEqual(VERBOSITY_MORE_VERBOSE, clf.verbosity)

    def test_init_device(self):
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        num_classes = 5
        device = 'cuda:0'

        clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label,
                                   use_differentiable_head=self.use_differentiable_head,
                                   device=device)

        self.assertEqual(setfit_model_args, clf.setfit_model_args)
        self.assertEqual(num_classes, clf.num_classes)
        self.assertEqual({}, clf.model_kwargs)
        self.assertEqual({}, clf.trainer_kwargs)

        self.assertEqual(self.use_differentiable_head, clf.use_differentiable_head)
        self.assertEqual(32, clf.mini_batch_size)
        self.assertEqual(clf.device, device)
        self.assertIsNone(clf.model)
        self.assertEqual(VERBOSITY_MORE_VERBOSE, clf.verbosity)

    def test_init_model_kwargs(self):
        from sklearn.multiclass import OneVsRestClassifier

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        num_classes = 5

        if not self.use_differentiable_head:
            model_kwargs = {'multi_target_strategy': 'one-vs-rest'}
        else:
            model_kwargs = {}

        clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label,
                                   model_kwargs=model_kwargs,
                                   use_differentiable_head=self.use_differentiable_head)

        self.assertEqual(setfit_model_args, clf.setfit_model_args)
        self.assertEqual(num_classes, clf.num_classes)
        self.assertEqual(model_kwargs, clf.model_kwargs)
        self.assertIsNone(clf.device)
        self.assertIsNone(clf.model)
        self.assertEqual(VERBOSITY_MORE_VERBOSE, clf.verbosity)

    def test_init_verbosity(self):
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        num_classes = 5
        device = 'cuda:0'

        clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label,
                                   use_differentiable_head=self.use_differentiable_head,
                                   verbosity=VERBOSITY_QUIET)

        self.assertEqual(setfit_model_args, clf.setfit_model_args)
        self.assertEqual(num_classes, clf.num_classes)
        self.assertEqual({}, clf.model_kwargs)
        self.assertEqual({}, clf.trainer_kwargs)

        self.assertEqual(self.use_differentiable_head, clf.use_differentiable_head)
        self.assertEqual(32, clf.mini_batch_size)
        self.assertIsNone(clf.device)
        self.assertEqual(VERBOSITY_QUIET, clf.verbosity)

    def test_init_model_loading_strategy_default(self):
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        num_classes = 5

        if not self.use_differentiable_head:
            model_kwargs = {'multi_target_strategy': 'one-vs-rest'}
        else:
            model_kwargs = {}

        with patch('setfit.modeling.SetFitModel.from_pretrained') as from_pretrained_mock:
            clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label,
                                       model_kwargs=model_kwargs,
                                       use_differentiable_head=self.use_differentiable_head)
            model = clf.initialize()

            self.assertIsNotNone(model)
            self.assertEqual(1, from_pretrained_mock.call_count)
            self.assertFalse(from_pretrained_mock.call_args.kwargs['force_download'])
            self.assertFalse(from_pretrained_mock.call_args.kwargs['local_files_only'])

    def test_initialize_model_loading_strategy_always_local(self):
        model_loading_strategy = ModelLoadingStrategy.ALWAYS_LOCAL
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                                 model_loading_strategy=model_loading_strategy)
        num_classes = 5

        if not self.use_differentiable_head:
            model_kwargs = {'multi_target_strategy': 'one-vs-rest'}
        else:
            model_kwargs = {}

        with patch('setfit.modeling.SetFitModel.from_pretrained') as from_pretrained_mock:
            clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label,
                                       model_kwargs=model_kwargs,
                                       use_differentiable_head=self.use_differentiable_head)
            model = clf.initialize()

            self.assertIsNotNone(model)
            self.assertEqual(1, from_pretrained_mock.call_count)
            self.assertFalse(from_pretrained_mock.call_args.kwargs['force_download'])
            self.assertTrue(from_pretrained_mock.call_args.kwargs['local_files_only'])

    def test_initialize_model_loading_strategy_always_download(self):
        model_loading_strategy = ModelLoadingStrategy.ALWAYS_DOWNLOAD
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                                 model_loading_strategy=model_loading_strategy)
        num_classes = 5

        if not self.use_differentiable_head:
            model_kwargs = {'multi_target_strategy': 'one-vs-rest'}
        else:
            model_kwargs = {}

        with patch('setfit.modeling.SetFitModel.from_pretrained') as from_pretrained_mock:
            clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label,
                                       model_kwargs=model_kwargs,
                                       use_differentiable_head=self.use_differentiable_head)

            model = clf.initialize()

            self.assertIsNotNone(model)
            self.assertEqual(1, from_pretrained_mock.call_count)
            self.assertTrue(from_pretrained_mock.call_args.kwargs['force_download'])
            self.assertFalse(from_pretrained_mock.call_args.kwargs['local_files_only'])

    def test_fit_without_train_kwargs(self):
        ds = random_text_dataset(10, multi_label=self.multi_label)
        num_classes = 5

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')

        with patch('small_text.integrations.transformers.classifiers.setfit.SetFitTrainer',
                   autospec=True) as trainer_mock:
            clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label)
            self.assertIsNone(clf.model)
            clf.fit(ds)

            self.assertIsNotNone(clf.model)
            trainer_mock.return_value.train.assert_called_with(max_length=512, show_progress_bar=True)

    def test_fit_with_train_kwargs(self):
        ds = random_text_dataset(10, multi_label=self.multi_label)
        num_classes = 5

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        setfit_train_kwargs = {'l2_weight': 0.2}

        with patch('small_text.integrations.transformers.classifiers.setfit.SetFitTrainer') as trainer_mock:
            clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label)
            self.assertIsNone(clf.model)
            clf.fit(ds, setfit_train_kwargs=setfit_train_kwargs)

            trainer_mock.return_value.train.assert_called()

            self.assertIsNotNone(clf.model)
            call_args = trainer_mock.return_value.train.call_args
            self.assertTrue('l2_weight' in call_args.kwargs)
            self.assertEqual(0.2, call_args.kwargs['l2_weight'])


@pytest.mark.pytorch
@pytest.mark.optional
class TestSetFitClassificationRegressionSingleLabel(unittest.TestCase, _SetFitClassification):

    def setUp(self):
        self.multi_label = False
        self.use_differentiable_head = False

    def test_fit(self):
        import datasets

        ds = random_text_dataset(10, multi_label=self.multi_label)
        num_classes = 5

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label)

        with patch.object(clf, '_fit') as fit_main_mock:
            clf.fit(ds)

            fit_main_mock.assert_called()

            call_args = fit_main_mock.call_args[0]
            self.assertTrue(isinstance(call_args[0], datasets.Dataset))
            self.assertTrue(call_args[1] is None)

            self.assertEqual(len(ds), len(call_args[0]))


@pytest.mark.pytorch
@pytest.mark.optional
class TestSetFitClassificationRegressionMultiLabel(unittest.TestCase, _SetFitClassification):

    def setUp(self):
        self.multi_label = True
        self.use_differentiable_head = False

    def test_fit(self):
        import datasets

        ds = random_text_dataset(10, multi_label=self.multi_label)
        num_classes = 5

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label)

        with patch.object(clf, '_fit') as fit_main_mock:
            clf.fit(ds)

            fit_main_mock.assert_called()

            call_args = fit_main_mock.call_args[0]
            self.assertTrue(isinstance(call_args[0], datasets.Dataset))
            self.assertTrue(call_args[1] is None)

            self.assertEqual(len(ds), len(call_args[0]))


@pytest.mark.pytorch
@pytest.mark.optional
class TestSetFitClassificationDifferentiableSingleLabel(unittest.TestCase,  _SetFitClassification):

    def setUp(self):
        self.multi_label = False
        self.use_differentiable_head = True

    def test_fit_with_differentiable_head(self):
        ds = random_text_dataset(10, multi_label=self.multi_label)
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        num_classes = 5

        clf = SetFitClassification(setfit_model_args,
                                   num_classes,
                                   multi_label=self.multi_label,
                                   use_differentiable_head=self.use_differentiable_head)
        with self.assertRaises(NotImplementedError):
            clf.fit(ds)


@pytest.mark.pytorch
@pytest.mark.optional
class TestSetFitClassificationDifferentiableMultiLabel(unittest.TestCase,  _SetFitClassification):

    def setUp(self):
        self.multi_label = True
        self.use_differentiable_head = True

    def test_fit_with_differentiable_head(self):
        ds = random_text_dataset(10, multi_label=self.multi_label)
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        num_classes = 5

        clf = SetFitClassification(setfit_model_args,
                                   num_classes,
                                   multi_label=self.multi_label,
                                   use_differentiable_head=self.use_differentiable_head)
        with self.assertRaises(NotImplementedError):
            clf.fit(ds)


@pytest.mark.pytorch
@pytest.mark.optional
class CompileTest(unittest.TestCase):

    def test_initialize_with_pytorch_geq_v2_and_compile_enabled(self):

        if parse(torch.__version__) >= Version('2.0.0'):
            setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2', compile_model=True)
            num_classes = 5

            clf = SetFitClassification(setfit_model_args, num_classes)

            with patch('torch.__version__', new='2.0.0'), \
                    patch('torch.compile', wraps=torch.compile) as compile_spy:
                clf.initialize()
                compile_spy.assert_called()

    def test_initialize_with_pytorch_geq_v2_and_compile_disabled(self):

        if parse(torch.__version__) >= Version('2.0.0'):
            setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2', compile_model=False)
            num_classes = 5

            clf = SetFitClassification(setfit_model_args, num_classes)

            with patch('torch.__version__', new='2.0.0'), \
                    patch('torch.compile', wraps=torch.compile) as compile_spy:
                clf.initialize()
                compile_spy.assert_not_called()

    def test_initialize_with_pytorch_lesser_v2_and_compile_enabled(self):

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2', compile_model=True)
        num_classes = 5

        clf = SetFitClassification(setfit_model_args, num_classes)

        with patch.object(torch, 'compile', return_value=None):
            with patch('torch.__version__', new='1.9.0'), \
                    patch('torch.compile', wraps=torch.compile) as compile_spy:
                clf.initialize()
                compile_spy.assert_not_called()

    def test_initialize_with_pytorch_lesser_v2_and_compile_disabled(self):

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2', compile_model=False)
        num_classes = 5

        clf = SetFitClassification(setfit_model_args, num_classes)

        with patch.object(torch, 'compile', return_value=None):
            with patch('torch.__version__', new='2.0.0'), \
                    patch('torch.compile', wraps=torch.compile) as compile_spy:
                clf.initialize()
                compile_spy.assert_not_called()
