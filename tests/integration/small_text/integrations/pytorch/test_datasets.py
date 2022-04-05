import unittest

import pytest

import numpy as np

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset
    from tests.utils.datasets import random_text_classification_dataset
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class PytorchTextClassificationDatasetTest(unittest.TestCase):

    def test_dataset_to(self):
        ds = random_text_classification_dataset(10)
        self.assertTrue(
            np.all([item[PytorchTextClassificationDataset.INDEX_TEXT].device == torch.device('cpu')
                    for item in ds.data])
        )

        ds = ds.to('cuda')
        self.assertTrue(
            np.all([item[PytorchTextClassificationDataset.INDEX_TEXT].device.type == 'cuda'
                    for item in ds.data])
        )

    def test_dataset_to_non_blocking(self):
        ds = random_text_classification_dataset(10)
        self.assertTrue(
            np.all([item[PytorchTextClassificationDataset.INDEX_TEXT].device == torch.device('cpu')
                    for item in ds.data])
        )

        ds = ds.to('cuda', non_blocking=True)
        self.assertTrue(
            np.all([item[PytorchTextClassificationDataset.INDEX_TEXT].device.type == 'cuda'
                    for item in ds.data])
        )

    def test_dataset_to_copy(self):
        ds = random_text_classification_dataset(10)
        self.assertTrue(ds.target_labels.shape[0] > 0)
        self.assertTrue(
            np.all([item[PytorchTextClassificationDataset.INDEX_TEXT].device == torch.device('cpu')
                    for item in ds.data])
        )

        ds_new = ds.to('cuda', copy=True)
        self.assertTrue(
            np.all([item[PytorchTextClassificationDataset.INDEX_TEXT].device.type == 'cuda'
                    for item in ds_new.data])
        )

        # test for object equality
        num_target_labels = ds.target_labels.shape[0]
        ds._target_labels = np.delete(ds.target_labels, 0)
        self.assertTrue(ds_new.target_labels.shape[0] == num_target_labels)

        # assign test attribute to check for object equality
        ds.vocab.TEST = 'test'
        self.assertFalse(hasattr(ds_new, 'TEST'))

    def test_dataset_to_copy_with_target_labels_explicit(self):
        ds = random_text_classification_dataset(10, target_labels='explicit')
        self.assertTrue(ds.target_labels.shape[0] > 0)
        self.assertTrue(
            np.all([item[PytorchTextClassificationDataset.INDEX_TEXT].device == torch.device('cpu')
                    for item in ds.data])
        )

        ds_new = ds.to('cuda', copy=True)
        self.assertTrue(
            np.all([item[PytorchTextClassificationDataset.INDEX_TEXT].device.type == 'cuda'
                    for item in ds_new.data])
        )

        # test for object equality
        num_target_labels = ds.target_labels.shape[0]
        ds._target_labels = np.delete(ds.target_labels, 0)
        self.assertTrue(ds_new.target_labels.shape[0] == num_target_labels)

        # assign test attribute to check for object equality
        ds.vocab.TEST = 'test'
        self.assertFalse(hasattr(ds_new, 'TEST'))
