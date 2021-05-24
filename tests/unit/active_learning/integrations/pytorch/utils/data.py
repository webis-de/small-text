import unittest

import pytest

from active_learning.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler

    from active_learning.integrations.pytorch.utils.data import dataloader
    from active_learning.integrations.pytorch.classifiers.kimcnn import kimcnn_collate_fn
    from tests.utils.datasets import random_text_classification_dataset
except (ModuleNotFoundError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class DataloaderTest(unittest.TestCase):

    def test_dataloader_train(self):
        ds = random_text_classification_dataset(10)
        iter = dataloader(ds, 3, kimcnn_collate_fn)

        self.assertTrue(isinstance(iter.sampler, BatchSampler))
        self.assertTrue(isinstance(iter.sampler.sampler, RandomSampler))

    def test_dataloader_test(self):
        ds = random_text_classification_dataset(10)
        iter = dataloader(ds, 3, kimcnn_collate_fn, train=False)

        self.assertTrue(isinstance(iter.sampler, BatchSampler))
        self.assertTrue(isinstance(iter.sampler.sampler, SequentialSampler))
