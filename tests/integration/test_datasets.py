import unittest

from active_learning.integrations.transformers.datasets import TransformersDataset
from tests.utils.datasets import random_transformer_dataset


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
