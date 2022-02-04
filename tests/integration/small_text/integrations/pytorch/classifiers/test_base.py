import unittest

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    from torch.nn.modules import BCEWithLogitsLoss

    from torch.optim import Adadelta, AdamW
    from torch.optim.lr_scheduler import ExponentialLR

    from small_text.integrations.pytorch.classifiers.base import PytorchClassifier
    from small_text.integrations.pytorch.models.kimcnn import KimCNN
    from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset
    from tests.utils.datasets import random_text_classification_dataset
except PytorchNotFoundError:
    pass


class PytorchClassifierImplementation(PytorchClassifier):

    def _default_optimizer(self, params, base_lr):
        return AdamW(params, lr=base_lr, eps=1e-8)

    def fit(self, train_set, validation_set=None, **kwargs):
        raise NotImplementedError()

    def predict(self, test_set, return_proba=False):
        raise NotImplementedError()

    def predict_proba(self, test_set):
        raise NotImplementedError()


class PytorchClassifierTest(unittest.TestCase):

    def _get_dataset(self, num_samples=100, num_classes=4):
        return random_text_classification_dataset(num_samples, max_length=60, num_classes=num_classes,
                                                  multi_label=self.multi_label)

    def test_initialize_optimizer_and_scheduler_default(self):
        sub_train = random_text_classification_dataset(10)

        classifier = PytorchClassifierImplementation()
        # initialize the model
        classifier.model = KimCNN(10, 60)

        optimizer = None
        scheduler = None
        params = None
        num_epochs = 2
        base_lr = 2e-5

        optimizer, scheduler = classifier._initialize_optimizer_and_scheduler(optimizer,
                                                                              scheduler,
                                                                              params,
                                                                              num_epochs,
                                                                              sub_train,
                                                                              base_lr,
                                                                              classifier.model)

        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(scheduler)

        optimizer_params = [param for param in classifier.model.parameters() if param.requires_grad]
        self.assertEqual(optimizer.__class__,
                         classifier._default_optimizer(optimizer_params, base_lr).__class__)

    def test_initialize_optimizer_and_scheduler_custom(self):
        sub_train = random_text_classification_dataset(10)

        classifier = PytorchClassifierImplementation()
        # initialize the model
        classifier.model = KimCNN(10, 60)

        optimizer_params = [param for param in classifier.model.parameters() if param.requires_grad]
        optimizer_arg = Adadelta(optimizer_params)
        scheduler_arg = ExponentialLR(optimizer_arg, 0.99)

        params = None
        num_epochs = 2
        base_lr = 2e-5

        optimizer, scheduler = classifier._initialize_optimizer_and_scheduler(optimizer_arg,
                                                                              scheduler_arg,
                                                                              params,
                                                                              num_epochs,
                                                                              sub_train,
                                                                              base_lr,
                                                                              classifier.model)

        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(scheduler)

        self.assertEqual(optimizer, optimizer_arg)
        self.assertEqual(scheduler, scheduler_arg)
