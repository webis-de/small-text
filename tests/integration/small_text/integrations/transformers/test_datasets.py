import unittest

import pytest

import numpy as np

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    from small_text.integrations.transformers.datasets import TransformersDataset
    from tests.utils.datasets import random_transformer_dataset
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class TransformersDatasetTest(unittest.TestCase):

    def test_init_and_len(self):
        data = random_transformer_dataset(10)
        dataset = TransformersDataset(data)
        self.assertEqual(10, len(dataset))

    def test_indexing(self):
        data = random_transformer_dataset(10)
        dataset = TransformersDataset(data)
        subset = dataset[[0, 1, 3, 5]]
        self.assertEqual(4, len(subset))

    def test_dataset_to(self):
        ds = random_transformer_dataset(10)
        ds = ds.to('cpu')
        self.assertTrue(
            np.all([item[TransformersDataset.INDEX_TEXT].device == torch.device('cpu')
                    for item in ds.data])
        )

        ds = ds.to('cuda')
        self.assertTrue(
            np.all([item[TransformersDataset.INDEX_TEXT].device.type == 'cuda'
                    for item in ds.data])
        )

    def test_dataset_to_non_blocking(self):
        ds = random_transformer_dataset(10)
        ds = ds.to('cpu')
        self.assertTrue(
            np.all([item[TransformersDataset.INDEX_TEXT].device == torch.device('cpu')
                    for item in ds.data])
        )

        ds = ds.to('cuda', non_blocking=True)
        self.assertTrue(
            np.all([item[TransformersDataset.INDEX_TEXT].device.type == 'cuda'
                    for item in ds.data])
        )

    def test_dataset_to_copy(self):
        ds = random_transformer_dataset(10)
        ds = ds.to('cpu')
        self.assertTrue(
            np.all([item[TransformersDataset.INDEX_TEXT].device == torch.device('cpu')
                    for item in ds.data])
        )

        ds_new = ds.to('cuda', copy=True)
        self.assertTrue(
            np.all([item[TransformersDataset.INDEX_TEXT].device.type == 'cuda'
                    for item in ds_new.data])
        )

    def test_dataset_to_copy_with_target_labels_explicit(self):
        ds = random_transformer_dataset(10, target_labels='explicit')
        ds = ds.to('cpu')
        self.assertTrue(
            np.all([item[TransformersDataset.INDEX_TEXT].device == torch.device('cpu')
                    for item in ds.data])
        )

        ds_new = ds.to('cuda', copy=True)
        self.assertTrue(
            np.all([item[TransformersDataset.INDEX_TEXT].device.type == 'cuda'
                    for item in ds_new.data])
        )

        # test for object equality
        num_target_labels = ds.target_labels.shape[0]
        ds._target_labels = np.delete(ds.target_labels, 0)
        self.assertTrue(ds_new.target_labels.shape[0] == num_target_labels)
