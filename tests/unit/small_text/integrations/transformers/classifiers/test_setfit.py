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

    from small_text.integrations.pytorch.classifiers.base import AMPArguments
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

    def test_setfit_model_arguments_init_default(self):
        sentence_transformer_model = 'sentence-transformers/all-MiniLM-L6-v2'
        args = SetFitModelArguments(sentence_transformer_model)
        self.assertEqual(sentence_transformer_model, args.sentence_transformer_model)
        self.assertIsNone(args.max_length)
        self.assertEqual((16, 2), args.mini_batch_size)
        self.assertEqual((1, 16), args.num_epochs)
        self.assertEqual(-1, args.max_steps)
        self.assertEqual('oversampling', args.sampling_strategy)
        self.assertIsNone(args.num_iterations)
        self.assertEqual((2e-5, 1e-5), args.body_learning_rate)
        self.assertEqual(1e-2, args.head_learning_rate)
        self.assertFalse(args.end_to_end)
        self.assertIsNone(args.seed)
        self.assertEqual('checkpoints', args.output_dir)
        self.assertIsNotNone(args.model_kwargs)
        self.assertEqual(0, len(args.model_kwargs))
        self.assertIsNotNone(args.trainer_kwargs)
        self.assertEqual(0, len(args.trainer_kwargs))
        self.assertIsNotNone(args.model_loading_strategy)
        self.assertEqual(ModelLoadingStrategy.DEFAULT, args.model_loading_strategy)
        self.assertFalse(args.compile_model)

    def test_setfit_model_arguments_init_custom(self):
        sentence_transformer_model = 'sentence-transformers/all-MiniLM-L6-v2'
        args = SetFitModelArguments(sentence_transformer_model,
                                    max_length=512,
                                    mini_batch_size=(32, 4),
                                    num_epochs=(2, 32),
                                    max_steps=10,
                                    sampling_strategy='undersampling',
                                    num_iterations=100,
                                    body_learning_rate=(1e-5, 2e-5),
                                    head_learning_rate=1e-3,
                                    end_to_end=True,
                                    seed=42,
                                    output_dir='/tmp/small-text',
                                    model_kwargs={'cache_dir': '/tmp/cache'},
                                    trainer_kwargs={'margin': 0.26},
                                    model_loading_strategy=ModelLoadingStrategy.ALWAYS_DOWNLOAD,
                                    compile_model=True)

        self.assertEqual(sentence_transformer_model, args.sentence_transformer_model)
        self.assertEqual(512, args.max_length)
        self.assertEqual((32, 4), args.mini_batch_size)
        self.assertEqual((2, 32), args.num_epochs)
        self.assertEqual(10, args.max_steps)
        self.assertEqual('undersampling', args.sampling_strategy)
        self.assertEqual(100, args.num_iterations)
        self.assertEqual((1e-5, 2e-5), args.body_learning_rate)
        self.assertEqual(1e-3, args.head_learning_rate)
        self.assertTrue(args.end_to_end)
        self.assertEqual(42, args.seed)
        self.assertEqual('/tmp/small-text', args.output_dir)
        self.assertEqual({'cache_dir': '/tmp/cache'}, args.model_kwargs)
        self.assertEqual({'margin': 0.26}, args.trainer_kwargs)
        self.assertEqual(ModelLoadingStrategy.ALWAYS_DOWNLOAD, args.model_loading_strategy)
        self.assertTrue(args.compile_model)

    def test_setfit_model_arguments_init_with_model_kwargs(self):
        sentence_transformer_model = 'sentence-transformers/all-MiniLM-L6-v2'
        model_kwargs = {'cache_dir': '/tmp/cache'}
        args = SetFitModelArguments(sentence_transformer_model, model_kwargs=model_kwargs)
        self.assertEqual(sentence_transformer_model, args.sentence_transformer_model)
        self.assertIsNotNone(args.model_kwargs)
        self.assertEqual(1, len(args.model_kwargs))
        self.assertIsNotNone(args.trainer_kwargs)
        self.assertEqual(0, len(args.trainer_kwargs))
        self.assertIsNotNone(args.model_loading_strategy)
        self.assertEqual(ModelLoadingStrategy.DEFAULT, args.model_loading_strategy)
        self.assertFalse(args.compile_model)

    # model_loading_strategy is tested separately since it interacts with environment variables
    def test_setfit_model_arguments_init_with_model_loading_strategy(self):
        sentence_transformer_model = 'sentence-transformers/all-MiniLM-L6-v2'
        model_loading_strategy = ModelLoadingStrategy.ALWAYS_LOCAL
        args = SetFitModelArguments(sentence_transformer_model,
                                    model_loading_strategy=model_loading_strategy)

        self.assertEqual(model_loading_strategy, args.model_loading_strategy)

    # model_loading_strategy is tested separately since it interacts with environment variables
    def test_setfit_model_arguments_init_with_env_override(self):
        with patch.dict(os.environ, {OFFLINE_MODE_VARIABLE: '1'}):

            # reload TransformerModelArguments so that updated environment variables are read
            reload(small_text.integrations.transformers.classifiers.setfit)
            from small_text.integrations.transformers.classifiers.setfit import SetFitModelArguments

            sentence_transformer_model = 'sentence-transformers/all-MiniLM-L6-v2'
            args = SetFitModelArguments(sentence_transformer_model)

            self.assertIsNotNone(args.model_loading_strategy)
            self.assertEqual(ModelLoadingStrategy.ALWAYS_LOCAL, args.model_loading_strategy)

@pytest.mark.pytorch
@pytest.mark.optional
class TestSetFitModelArgumentsWithManagedKeywords(unittest.TestCase):

    def test_init_with_misplaced_use_differentiable_head_kwarg(self):
        model_kwargs = {'use_differentiable_head': False}

        with self.assertRaisesRegex(ValueError, 'Invalid keyword argument in model_kwargs'):
            SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                 model_kwargs=model_kwargs)

    def test_init_with_misplaced_force_download_kwarg(self):
        model_kwargs = {'force_download': True}

        with self.assertRaisesRegex(ValueError, 'Invalid keyword argument in model_kwargs'):
            SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                 model_kwargs=model_kwargs)

    def test_init_with_misplaced_local_files_only_kwarg(self):
        model_kwargs = {'local_files_only': True}

        with self.assertRaisesRegex(ValueError, 'Invalid keyword argument in model_kwargs'):
            SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                 model_kwargs=model_kwargs)

    def test_init_with_misplaced_batch_size_kwargs(self):
        trainer_kwargs = {'batch_size': 20}

        with self.assertRaisesRegex(ValueError, 'Invalid keyword argument in trainer_kwargs'):
            SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                 trainer_kwargs=trainer_kwargs)

    def test_init_with_misplaced_show_progress_bar_kwargs(self):
        trainer_kwargs = {'show_progress_bar': True}

        with self.assertRaisesRegex(ValueError, 'Invalid keyword argument in trainer_kwargs'):
            SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                 trainer_kwargs=trainer_kwargs)
    def test_init_with_misplaced_seed_kwargs(self):
        trainer_kwargs = {'seed': 4242}

        with self.assertRaisesRegex(ValueError, 'Invalid keyword argument in trainer_kwargs'):
            SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                 trainer_kwargs=trainer_kwargs)


@pytest.mark.pytorch
@pytest.mark.optional
class TestTrainingArgumentsConversion(unittest.TestCase):

    def test_defaults(self):

        ds = random_text_dataset(10)
        num_classes = 5

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        clf = SetFitClassification(setfit_model_args, num_classes)

        with (patch.object(clf, '_call_setfit_trainer') as call_setfit_trainer_mock,
              patch.object(clf, '_get_training_args', wraps=clf._get_training_args) as get_training_args_mock):
            clf.fit(ds)

            self.assertEqual(2, len(get_training_args_mock.call_args))
            call_args = get_training_args_mock.call_args.args
            self.assertTrue(isinstance(call_args[0], SetFitModelArguments))
            self.assertTrue(isinstance(call_args[1], AMPArguments))

            model_args = call_args[0]

            self.assertIsNone(model_args.max_length)
            self.assertEqual((16, 2), model_args.mini_batch_size)
            self.assertEqual((1, 16), model_args.num_epochs)
            self.assertEqual(-1, model_args.max_steps)
            self.assertEqual('oversampling', model_args.sampling_strategy)
            self.assertIsNone(model_args.num_iterations)
            self.assertEqual((2e-5, 1e-5), model_args.body_learning_rate)
            self.assertEqual(1e-2, model_args.head_learning_rate)
            self.assertFalse(model_args.end_to_end)
            self.assertIsNone(model_args.seed)
            self.assertTrue(model_args.show_progress_bar)

            call_setfit_trainer_mock.assert_called()

    def test_non_defaults(self):

        ds = random_text_dataset(10)
        num_classes = 5

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                                 max_length=1024,
                                                 mini_batch_size=(8, 4),
                                                 num_epochs=(2, 4),
                                                 max_steps=100,
                                                 sampling_strategy='undersampling',
                                                 num_iterations=100,
                                                 body_learning_rate=(5e-5, 2e-5),
                                                 head_learning_rate=1e-3,
                                                 end_to_end=True,
                                                 show_progress_bar=False,
                                                 seed=42)
        clf = SetFitClassification(setfit_model_args, num_classes)

        with (patch.object(clf, '_call_setfit_trainer') as call_setfit_trainer_mock,
              patch.object(clf, '_get_training_args', wraps=clf._get_training_args) as get_training_args_mock):
            clf.fit(ds)

            self.assertEqual(2, len(get_training_args_mock.call_args))
            call_args = get_training_args_mock.call_args.args
            self.assertTrue(isinstance(call_args[0], SetFitModelArguments))
            self.assertTrue(isinstance(call_args[1], AMPArguments))

            model_args = call_args[0]

            self.assertEqual(1024, model_args.max_length)
            self.assertEqual((8, 4), model_args.mini_batch_size)
            self.assertEqual((2, 4), model_args.num_epochs)
            self.assertEqual(100, model_args.max_steps)
            self.assertEqual('undersampling', model_args.sampling_strategy)
            self.assertEqual(100, model_args.num_iterations)
            self.assertEqual((5e-5, 2e-5), model_args.body_learning_rate)
            self.assertEqual(1e-3, model_args.head_learning_rate)
            self.assertTrue(model_args.end_to_end)
            self.assertEqual(42, model_args.seed)
            self.assertFalse(model_args.show_progress_bar)

            call_setfit_trainer_mock.assert_called()


class _SetFitClassification(object):

    def test_init(self):
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        num_classes = 5

        clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label,
                                   use_differentiable_head=self.use_differentiable_head)

        self.assertEqual(setfit_model_args, clf.setfit_model_args)
        self.assertEqual(num_classes, clf.num_classes)

        self.assertEqual(self.use_differentiable_head, clf.use_differentiable_head)
        self.assertEqual(32, clf.mini_batch_size)
        self.assertIsNone(clf.device)
        self.assertIsNone(clf.model)
        # self.assertEqual(VERBOSITY_MORE_VERBOSE, clf.verbosity)

    def test_init_device(self):
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        num_classes = 5
        device = 'cuda:0'

        clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label,
                                   use_differentiable_head=self.use_differentiable_head,
                                   device=device)

        self.assertEqual(setfit_model_args, clf.setfit_model_args)
        self.assertEqual(num_classes, clf.num_classes)

        self.assertEqual(self.use_differentiable_head, clf.use_differentiable_head)
        self.assertEqual(32, clf.mini_batch_size)
        self.assertEqual(clf.device, device)
        self.assertIsNone(clf.model)
        # self.assertEqual(VERBOSITY_MORE_VERBOSE, clf.verbosity)

    def test_init_model_kwargs(self):
        if not self.use_differentiable_head:
            model_kwargs = {'multi_target_strategy': 'one-vs-rest'}
        else:
            model_kwargs = {}

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                                 model_kwargs=model_kwargs)
        num_classes = 5

        clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label,
                                   use_differentiable_head=self.use_differentiable_head)

        self.assertEqual(setfit_model_args, clf.setfit_model_args)
        self.assertEqual(model_kwargs, clf.setfit_model_args.model_kwargs)
        self.assertEqual(num_classes, clf.num_classes)
        self.assertIsNone(clf.device)
        self.assertIsNone(clf.model)
        # self.assertEqual(VERBOSITY_MORE_VERBOSE, clf.verbosity)

    def test_init_model_loading_strategy_default(self):
        if not self.use_differentiable_head:
            model_kwargs = {'multi_target_strategy': 'one-vs-rest'}
        else:
            model_kwargs = {}

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                                 model_kwargs=model_kwargs)
        num_classes = 5

        with patch('setfit.modeling.SetFitModel.from_pretrained') as from_pretrained_mock:
            clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label,
                                       use_differentiable_head=self.use_differentiable_head)
            model = clf.initialize()

            self.assertIsNotNone(model)
            self.assertEqual(1, from_pretrained_mock.call_count)
            self.assertFalse(from_pretrained_mock.call_args.kwargs['force_download'])
            self.assertFalse(from_pretrained_mock.call_args.kwargs['local_files_only'])

    def test_initialize_model_loading_strategy_always_local(self):
        if not self.use_differentiable_head:
            model_kwargs = {'multi_target_strategy': 'one-vs-rest'}
        else:
            model_kwargs = {}

        model_loading_strategy = ModelLoadingStrategy.ALWAYS_LOCAL
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                                 model_loading_strategy=model_loading_strategy,
                                                 model_kwargs=model_kwargs)
        num_classes = 5

        with patch('setfit.modeling.SetFitModel.from_pretrained') as from_pretrained_mock:
            clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label,
                                       use_differentiable_head=self.use_differentiable_head)
            model = clf.initialize()

            self.assertIsNotNone(model)
            self.assertEqual(1, from_pretrained_mock.call_count)
            self.assertFalse(from_pretrained_mock.call_args.kwargs['force_download'])
            self.assertTrue(from_pretrained_mock.call_args.kwargs['local_files_only'])

    def test_initialize_model_loading_strategy_always_download(self):
        if not self.use_differentiable_head:
            model_kwargs = {'multi_target_strategy': 'one-vs-rest'}
        else:
            model_kwargs = {}

        model_loading_strategy = ModelLoadingStrategy.ALWAYS_DOWNLOAD
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                                 model_loading_strategy=model_loading_strategy,
                                                 model_kwargs=model_kwargs)
        num_classes = 5



        with patch('setfit.modeling.SetFitModel.from_pretrained') as from_pretrained_mock:
            clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label,
                                       use_differentiable_head=self.use_differentiable_head)

            model = clf.initialize()

            self.assertIsNotNone(model)
            self.assertEqual(1, from_pretrained_mock.call_count)
            self.assertTrue(from_pretrained_mock.call_args.kwargs['force_download'])
            self.assertFalse(from_pretrained_mock.call_args.kwargs['local_files_only'])

    def test_fit_with_train_set_mismatch(self):
        train_set = random_text_dataset(10, multi_label=not self.multi_label)
        num_classes = 5

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        classifier = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label)

        with self.assertRaisesRegex(ValueError, 'The classifier is configured for '):
            classifier.fit(train_set)

    def test_fit_with_validation_set_mismatch(self):
        train_set = random_text_dataset(10, multi_label=self.multi_label)
        validation_set = random_text_dataset(2,
                                             multi_label=not self.multi_label,
                                             assure_all_labels_occur=False)
        num_classes = 5

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        classifier = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label)

        with self.assertRaisesRegex(ValueError, 'The classifier is configured for '):
            classifier.fit(train_set, validation_set=validation_set)

    def test_fit_without_train_kwargs(self):
        ds = random_text_dataset(10, multi_label=self.multi_label)
        num_classes = 5

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')

        with patch('small_text.integrations.transformers.classifiers.setfit.Trainer',
                   autospec=True) as trainer_mock:
            clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label)
            self.assertIsNone(clf.model)
            clf.fit(ds)

            self.assertIsNotNone(clf.model)
            trainer_mock.return_value.train.assert_called_once()
            self.assertEqual(1, len(trainer_mock.return_value.train.call_args.kwargs))
            self.assertTrue('args' in trainer_mock.return_value.train.call_args.kwargs)

    def test_fit_with_amp(self):
        ds = random_text_dataset(10, multi_label=self.multi_label)
        num_classes = 5

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        amp_args = AMPArguments(use_amp=True)

        with patch('small_text.integrations.transformers.classifiers.setfit.Trainer') as trainer_mock:
            clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label, amp_args=amp_args)
            self.assertIsNone(clf.model)
            clf.fit(ds)

            trainer_mock.return_value.train.assert_called()
            self.assertTrue(trainer_mock.call_args_list[0].kwargs['args'].use_amp)

    def test_fit_with_show_progress_bar(self):
        ds = random_text_dataset(10, multi_label=self.multi_label)
        num_classes = 5

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2', show_progress_bar=False)

        with patch('small_text.integrations.transformers.classifiers.setfit.Trainer') as trainer_mock:
            clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label)
            self.assertIsNone(clf.model)
            clf.fit(ds)

            trainer_mock.return_value.train.assert_called()
            self.assertFalse(trainer_mock.call_args_list[0].kwargs['args'].show_progress_bar)


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
        import datasets

        ds = random_text_dataset(10, multi_label=self.multi_label)
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                                 output_dir='/tmp')
        num_classes = 5

        clf = SetFitClassification(setfit_model_args,
                                   num_classes,
                                   multi_label=self.multi_label,
                                   use_differentiable_head=self.use_differentiable_head)

        with patch.object(clf, '_fit') as fit_main_mock:
            clf.fit(ds)

            fit_main_mock.assert_called()

            call_args = fit_main_mock.call_args[0]
            self.assertTrue(isinstance(call_args[0], datasets.Dataset))
            self.assertTrue(call_args[1] is None)

            self.assertEqual(len(ds), len(call_args[0]))


@pytest.mark.pytorch
@pytest.mark.optional
class TestSetFitClassificationDifferentiableMultiLabel(unittest.TestCase,  _SetFitClassification):

    def setUp(self):
        self.multi_label = True
        self.use_differentiable_head = True

    def test_fit_with_differentiable_head(self):
        import datasets

        ds = random_text_dataset(10, multi_label=self.multi_label)
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                                 output_dir='/tmp')
        num_classes = 5

        clf = SetFitClassification(setfit_model_args,
                                   num_classes,
                                   multi_label=self.multi_label,
                                   use_differentiable_head=self.use_differentiable_head)

        with patch.object(clf, '_fit') as fit_main_mock:
            clf.fit(ds)

            fit_main_mock.assert_called()

            call_args = fit_main_mock.call_args[0]
            self.assertTrue(isinstance(call_args[0], datasets.Dataset))
            self.assertTrue(call_args[1] is None)

            self.assertEqual(len(ds), len(call_args[0]))


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
