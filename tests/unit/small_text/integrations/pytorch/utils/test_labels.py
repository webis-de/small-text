import unittest
import pytest

import numpy as np

from scipy.sparse import csr_matrix
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from small_text.base import LABEL_UNLABELED
    from small_text.integrations.pytorch.utils.labels import get_flattened_unique_labels
    from tests.utils.datasets import (
        random_text_classification_dataset,
        random_transformer_dataset
    )
    from tests.utils.testing import assert_array_equal
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class LabelsUtilsForPytorchTextClassificationDatasetTest(unittest.TestCase):

    def test_get_flattened_unique_labels(self):
        dataset = random_text_classification_dataset()
        labels = get_flattened_unique_labels(dataset)
        assert_array_equal(np.array([0, 1]), labels)

    def test_get_flattened_unique_labels_no_labels(self):
        dataset = random_text_classification_dataset()
        dataset.y = np.array([LABEL_UNLABELED] * len(dataset))
        labels = get_flattened_unique_labels(dataset)
        self.assertEqual((0,), labels.shape)

    def test_get_flattened_unique_labels_multi_label(self):
        num_classes = 3
        dataset = random_text_classification_dataset(multi_label=True, num_classes=num_classes)
        labels = get_flattened_unique_labels(dataset)
        assert_array_equal(np.array([0, 1, 2]), labels)

    def test_get_flattened_unique_labels_multi_label_no_labels(self):
        num_classes = 3
        dataset = random_text_classification_dataset(multi_label=True, num_classes=num_classes)
        dataset.y = csr_matrix((10, num_classes))
        labels = get_flattened_unique_labels(dataset)
        self.assertEqual((0,), labels.shape)


@pytest.mark.pytorch
class LabelsUtilsForTransformersDatasetTest(unittest.TestCase):

    def test_get_flattened_unique_labels(self):
        dataset = random_transformer_dataset(10)
        labels = get_flattened_unique_labels(dataset)
        assert_array_equal(np.array([0, 1]), labels)

    def test_get_flattened_unique_labels_no_labels(self):
        dataset = random_transformer_dataset(10)
        dataset.y = np.array([LABEL_UNLABELED] * len(dataset))
        labels = get_flattened_unique_labels(dataset)
        self.assertEqual((0,), labels.shape)

    def test_get_flattened_unique_labels_multi_label(self):
        num_classes = 3
        dataset = random_transformer_dataset(10, multi_label=True, num_classes=num_classes)
        labels = get_flattened_unique_labels(dataset)
        assert_array_equal(np.array([0, 1, 2]), labels)

    def test_get_flattened_unique_labels_multi_label_no_labels(self):
        num_classes = 3
        dataset = random_transformer_dataset(10, multi_label=True, num_classes=num_classes)
        dataset.y = csr_matrix((10, num_classes))
        labels = get_flattened_unique_labels(dataset)
        self.assertEqual((0,), labels.shape)
