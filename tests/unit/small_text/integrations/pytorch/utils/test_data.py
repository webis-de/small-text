import unittest

import numpy as np
import pytest
import torch

from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_matrix
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler

    from small_text.integrations.pytorch.utils.data import dataloader, get_class_weights
    from small_text.integrations.pytorch.classifiers.kimcnn import kimcnn_collate_fn
    from tests.utils.datasets import random_text_classification_dataset
except (ModuleNotFoundError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class DataloaderTest(unittest.TestCase):

    def test_dataloader_train(self):
        ds = random_text_classification_dataset(10)
        loader = dataloader(ds, 3, kimcnn_collate_fn)

        self.assertTrue(isinstance(loader.sampler, BatchSampler))
        self.assertTrue(isinstance(loader.sampler.sampler, RandomSampler))
        self.assertTrue(isinstance(loader.dataset, np.ndarray))

    def test_dataloader_test(self):
        ds = random_text_classification_dataset(10)
        loader = dataloader(ds, 3, kimcnn_collate_fn, train=False)

        self.assertTrue(isinstance(loader.sampler, BatchSampler))
        self.assertTrue(isinstance(loader.sampler.sampler, SequentialSampler))
        self.assertTrue(isinstance(loader.dataset, np.ndarray))


@pytest.mark.pytorch
class ClassWeightsTest(unittest.TestCase):

    def test_get_class_weights_binary(self):
        y = np.array([0, 1, 1, 1, 1])
        class_weights = get_class_weights(y, 2)
        self.assertTrue(torch.equal(torch.tensor([4., 1.0]), class_weights))

    def test_get_class_weights_multiclass(self):
        y = np.array([0, 1, 1, 1, 1, 2, 3, 3])
        class_weights = get_class_weights(y, 4)
        self.assertTrue(torch.equal(torch.tensor([7.0, 1.0, 7.0, 3.0]), class_weights))

    def test_get_class_weights_multi_label(self):
        num_classes = 4

        y = np.array([[1, 0, 1, 0],
                      [1, 1, 0, 0],
                      [0, 1, 1, 0],
                      [0, 1, 0, 0],
                      [1, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 1, 1, 1],
                      [1, 1, 0, 1]])
        y = csr_matrix(y, shape=(y.shape[0], num_classes))

        class_weights = get_class_weights(y, num_classes)
        assert_array_almost_equal(np.array([1.8, 1., 1.8, 4.2]), class_weights.cpu().numpy())
