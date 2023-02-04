import unittest
import pytest

from unittest.mock import patch

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
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

    def test_setfit_model_arguments_init_with_model_loading_strategy(self):
        sentence_transformer_model = 'sentence-transformers/all-MiniLM-L6-v2'
        model_loading_strategy = ModelLoadingStrategy.ALWAYS_LOCAL
        args = SetFitModelArguments(sentence_transformer_model,
                                    model_loading_strategy=model_loading_strategy)
        self.assertEqual(sentence_transformer_model, args.sentence_transformer_model)
        self.assertIsNotNone(args.model_loading_strategy)
        self.assertEqual(model_loading_strategy, args.model_loading_strategy)


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
        if not self.use_differentiable_head:
            self.assertTrue(isinstance(clf.model.model_head, OneVsRestClassifier))

    def test_init_model_loading_strategy_default(self):
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        num_classes = 5

        if not self.use_differentiable_head:
            model_kwargs = {'multi_target_strategy': 'one-vs-rest'}
        else:
            model_kwargs = {}

        with patch('setfit.modeling.SetFitModel.from_pretrained') as from_pretrained_mock:
            SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label,
                                 model_kwargs=model_kwargs,
                                 use_differentiable_head=self.use_differentiable_head)
            self.assertEqual(1, from_pretrained_mock.call_count)
            self.assertFalse(from_pretrained_mock.call_args.kwargs['force_download'])
            self.assertFalse(from_pretrained_mock.call_args.kwargs['local_files_only'])

    def test_init_model_loading_strategy_always_local(self):
        model_loading_strategy = ModelLoadingStrategy.ALWAYS_LOCAL
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                                 model_loading_strategy=model_loading_strategy)
        num_classes = 5

        if not self.use_differentiable_head:
            model_kwargs = {'multi_target_strategy': 'one-vs-rest'}
        else:
            model_kwargs = {}

        with patch('setfit.modeling.SetFitModel.from_pretrained') as from_pretrained_mock:
            SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label,
                                 model_kwargs=model_kwargs,
                                 use_differentiable_head=self.use_differentiable_head)
            self.assertEqual(1, from_pretrained_mock.call_count)
            self.assertFalse(from_pretrained_mock.call_args.kwargs['force_download'])
            self.assertTrue(from_pretrained_mock.call_args.kwargs['local_files_only'])

    def test_init_model_loading_strategy_always_download(self):
        model_loading_strategy = ModelLoadingStrategy.ALWAYS_DOWNLOAD
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2',
                                                 model_loading_strategy=model_loading_strategy)
        num_classes = 5

        if not self.use_differentiable_head:
            model_kwargs = {'multi_target_strategy': 'one-vs-rest'}
        else:
            model_kwargs = {}

        with patch('setfit.modeling.SetFitModel.from_pretrained') as from_pretrained_mock:
            SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label,
                                 model_kwargs=model_kwargs,
                                 use_differentiable_head=self.use_differentiable_head)
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
            clf.fit(ds)

            trainer_mock.return_value.train.assert_called_with(max_length=512)

    def test_fit_with_train_kwargs(self):
        ds = random_text_dataset(10, multi_label=self.multi_label)
        num_classes = 5

        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        setfit_train_kwargs = {'show_progress_bar': False}

        with patch('small_text.integrations.transformers.classifiers.setfit.SetFitTrainer') as trainer_mock:
            clf = SetFitClassification(setfit_model_args, num_classes, multi_label=self.multi_label)
            clf.fit(ds, setfit_train_kwargs=setfit_train_kwargs)

            trainer_mock.return_value.train.assert_called()

            call_args = trainer_mock.return_value.train.call_args
            self.assertTrue('show_progress_bar' in call_args.kwargs)
            self.assertFalse(call_args.kwargs['show_progress_bar'])


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
