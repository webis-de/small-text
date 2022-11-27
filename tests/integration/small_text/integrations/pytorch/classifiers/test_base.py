import unittest

import pytest
import torch

import numpy as np

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from torch.optim import Adadelta, AdamW
    from torch.optim.lr_scheduler import ExponentialLR

    from small_text.integrations.pytorch.classifiers.base import PytorchClassifier
    from small_text.integrations.pytorch.models.kimcnn import KimCNN
    from tests.utils.datasets import random_text_classification_dataset
except PytorchNotFoundError:
    pass


class PytorchClassifierImplementation(PytorchClassifier):

    def __init__(self, num_classes, class_weight=None):
        self.num_classes = num_classes
        self.class_weight = class_weight
        super().__init__()

    def _default_optimizer(self, base_lr):
        params = [param for param in self.model.parameters() if param.requires_grad]
        return params, AdamW(params, lr=base_lr, eps=1e-8)

    def fit(self, train_set, validation_set=None, **kwargs):
        raise NotImplementedError()

    def predict(self, test_set, return_proba=False):
        raise NotImplementedError()

    def predict_proba(self, test_set):
        raise NotImplementedError()

    def _predict_proba(self, dataset_iter, logits_transform):
        pass

    def _predict_proba_dropout_sampling(self, dataset_iter, logits_transform, dropout_samples=2):
        pass


@pytest.mark.pytorch
class PytorchClassifierTest(unittest.TestCase):

    def _get_dataset(self, num_samples=100, num_classes=4):
        return random_text_classification_dataset(num_samples, max_length=60, num_classes=num_classes,
                                                  multi_label=self.multi_label)

    def test_initialize_optimizer_and_scheduler_default(self):
        sub_train = random_text_classification_dataset(10)

        classifier = PytorchClassifierImplementation(2)
        # initialize the model
        classifier.model = KimCNN(10, 60)

        optimizer = None
        scheduler = None
        num_epochs = 2
        base_lr = 2e-5

        optimizer, scheduler = classifier._initialize_optimizer_and_scheduler(optimizer,
                                                                              scheduler,
                                                                              num_epochs,
                                                                              sub_train,
                                                                              base_lr)

        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(scheduler)

        params, default_optimizer = classifier._default_optimizer(base_lr)
        self.assertEqual(optimizer.__class__, default_optimizer.__class__)

    def test_initialize_optimizer_and_scheduler_custom(self):
        sub_train = random_text_classification_dataset(10)

        classifier = PytorchClassifierImplementation(2)
        # initialize the model
        classifier.model = KimCNN(10, 60)

        optimizer_params = [param for param in classifier.model.parameters() if param.requires_grad]
        optimizer_arg = Adadelta(optimizer_params)
        scheduler_arg = ExponentialLR(optimizer_arg, 0.99)

        num_epochs = 2
        base_lr = 2e-5

        optimizer, scheduler = classifier._initialize_optimizer_and_scheduler(optimizer_arg,
                                                                              scheduler_arg,
                                                                              num_epochs,
                                                                              sub_train,
                                                                              base_lr)

        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(scheduler)

        self.assertEqual(optimizer, optimizer_arg)
        self.assertEqual(scheduler, scheduler_arg)

    def test_initialize_class_weights_binary_no_class_weights(self):
        sub_train = random_text_classification_dataset(10)
        classifier = PytorchClassifierImplementation(2)
        class_weights = classifier.initialize_class_weights(sub_train)

        self.assertIsNone(class_weights)

    def test_initialize_class_weights_binary_balanced(self):
        sub_train = random_text_classification_dataset(10)
        sub_train.y = np.array([0] * 5 + [1] * 5)

        classifier = PytorchClassifierImplementation(2, class_weight='balanced')
        class_weights = classifier.initialize_class_weights(sub_train)

        expected_weights = torch.ones(2, dtype=torch.float)

        self.assertTrue(torch.equal(expected_weights.cpu(), class_weights.cpu()))

    def test_initialize_class_weights_binary_imbalanced(self):
        sub_train = random_text_classification_dataset(10)
        sub_train.y = np.array([0] * 2 + [1] * 8)

        classifier = PytorchClassifierImplementation(2, class_weight='balanced')
        class_weights = classifier.initialize_class_weights(sub_train)

        expected_weights = torch.FloatTensor([4.0, 1.0])

        self.assertTrue(torch.equal(expected_weights.cpu(), class_weights.cpu()))

    def test_initialize_class_weights_multi_class_balanced(self):
        sub_train = random_text_classification_dataset(8)
        sub_train.y = np.array([0] * 2 + [1] * 2 + [2] * 2 + [3] * 2)

        classifier = PytorchClassifierImplementation(4, class_weight='balanced')
        class_weights = classifier.initialize_class_weights(sub_train)

        expected_weights = torch.ones(4, dtype=torch.float)

        self.assertTrue(torch.equal(expected_weights.cpu(), class_weights.cpu()))

    def test_initialize_class_weights_multi_class_imbalanced(self):
        sub_train = random_text_classification_dataset(8)
        sub_train.y = np.array([0] * 1 + [1] * 1 + [2] * 2 + [3] * 4)

        classifier = PytorchClassifierImplementation(4, class_weight='balanced')
        class_weights = classifier.initialize_class_weights(sub_train)

        expected_weights = torch.FloatTensor([7.0, 7.0, 3.0, 1.0])

        self.assertTrue(torch.equal(expected_weights.cpu(), class_weights.cpu()))
