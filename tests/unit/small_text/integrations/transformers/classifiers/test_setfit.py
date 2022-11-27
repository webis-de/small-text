import unittest
import pytest

from unittest.mock import patch

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from small_text.integrations.transformers.classifiers.setfit import (
        SetFitClassification,
        SetFitModelArguments
    )
    from tests.utils.datasets import random_text_dataset
except (ModuleNotFoundError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class TestSetFitModelArguments(unittest.TestCase):

    def test_setfit_model_arguments_init(self):
        sentence_transformer_model = 'sentence-transformers/all-MiniLM-L6-v2'
        args = SetFitModelArguments(sentence_transformer_model)
        self.assertEqual(sentence_transformer_model, args.sentence_transformer_model)


class TestSetFitClassificationKeywordArguments(unittest.TestCase):

    def test_init_with_misplaced_use_differentiable_head_kwargs(self):
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        num_classes = 5

        model_kwargs = {'use_differentiable_head': False}

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
        if not self.use_differentiable_head:
            self.assertTrue(isinstance(clf.model.model_head, OneVsRestClassifier))


@pytest.mark.pytorch
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
