import unittest
import numpy as np

from numpy.testing import assert_array_equal
from sklearn.feature_extraction.text import TfidfVectorizer
from unittest.mock import patch

from small_text.data import SklearnDataset, TextDataset
from tests.utils.datasets import random_labeling, random_labels
from tests.utils.testing import assert_csr_matrix_equal


class SklearnDatasetConstructionSingleLabelTest(unittest.TestCase):

    def test_from_arrays_with_lists(self):
        texts = [f'train data {i}' for i in range(10)]
        labels = np.array([random_labeling(3) for _ in range(10)])

        vectorizer = TfidfVectorizer()

        with patch.object(vectorizer,
                          'fit_transform',
                          wraps=vectorizer.fit_transform) as fit_transform_spy:
            dataset = SklearnDataset.from_arrays(texts, labels, vectorizer)

            self.assertEqual(10, len(dataset))
            assert_array_equal(labels, dataset.y)
            is_fitted = [v for v in vars(vectorizer)
                         if v.endswith('_') and not v.startswith('__')]
            self.assertTrue(is_fitted)

            fit_transform_spy.assert_called_with(texts)

    def test_from_arrays_with_ndarray(self):
        texts = np.array([f'train data {i}' for i in range(10)])
        labels = np.array([random_labeling(3) for _ in range(10)])

        vectorizer = TfidfVectorizer()

        with patch.object(vectorizer,
                          'fit_transform',
                          wraps=vectorizer.fit_transform) as fit_transform_spy:
            dataset = SklearnDataset.from_arrays(texts, labels, vectorizer)

            self.assertEqual(10, len(dataset))
            assert_array_equal(labels, dataset.y)
            is_fitted = [v for v in vars(vectorizer)
                         if v.endswith('_') and not v.startswith('__')]
            self.assertTrue(is_fitted)

            fit_transform_spy.assert_called_with(texts)

    def test_from_arrays_test_data(self):
        texts = np.array([f'data {i}' for i in range(10)])
        labels = np.array([random_labeling(3) for _ in range(10)])

        vectorizer = TfidfVectorizer()
        # usually we would not call fit here but a fitted vocab is required
        # before calling transform()
        vectorizer.fit(texts)

        with patch.object(vectorizer,
                          'transform',
                          wraps=vectorizer.transform) as transform_spy:
            dataset = SklearnDataset.from_arrays(texts, labels, vectorizer, train=False)

            self.assertEqual(10, len(dataset))
            assert_array_equal(labels, dataset.y)
            is_fitted = [v for v in vars(vectorizer)
                         if v.endswith('_') and not v.startswith('__')]
            self.assertTrue(is_fitted)

            transform_spy.assert_called_with(texts)


class SklearnDatasetConstructionMultiLabelTest(unittest.TestCase):

    def test_from_arrays_with_lists(self):
        texts = [f'train data {i}' for i in range(10)]
        labels = random_labels(10, 3, multi_label=True)

        vectorizer = TfidfVectorizer()

        with patch.object(vectorizer,
                          'fit_transform',
                          wraps=vectorizer.fit_transform) as fit_transform_spy:
            dataset = SklearnDataset.from_arrays(texts, labels, vectorizer)

            self.assertEqual(10, len(dataset))
            assert_csr_matrix_equal(labels, dataset.y)
            is_fitted = [v for v in vars(vectorizer)
                         if v.endswith('_') and not v.startswith('__')]
            self.assertTrue(is_fitted)

            fit_transform_spy.assert_called_with(texts)

    def test_from_arrays_with_ndarray(self):
        texts = np.array([f'train data {i}' for i in range(10)])
        labels = random_labels(10, 3, multi_label=True)

        vectorizer = TfidfVectorizer()

        with patch.object(vectorizer,
                          'fit_transform',
                          wraps=vectorizer.fit_transform) as fit_transform_spy:
            dataset = SklearnDataset.from_arrays(texts, labels, vectorizer)

            self.assertEqual(10, len(dataset))
            assert_csr_matrix_equal(labels, dataset.y)
            is_fitted = [v for v in vars(vectorizer)
                         if v.endswith('_') and not v.startswith('__')]
            self.assertTrue(is_fitted)

            fit_transform_spy.assert_called_with(texts)

    def test_from_arrays_test_data(self):
        texts = np.array([f'data {i}' for i in range(10)])
        labels = random_labels(10, 3, multi_label=True)

        vectorizer = TfidfVectorizer()
        # usually we would not call fit here but a fitted vocab is required
        # before calling transform()
        vectorizer.fit(texts)

        with patch.object(vectorizer,
                          'transform',
                          wraps=vectorizer.transform) as transform_spy:
            dataset = SklearnDataset.from_arrays(texts, labels, vectorizer, train=False)

            self.assertEqual(10, len(dataset))
            assert_csr_matrix_equal(labels, dataset.y)
            is_fitted = [v for v in vars(vectorizer)
                         if v.endswith('_') and not v.startswith('__')]
            self.assertTrue(is_fitted)

            transform_spy.assert_called_with(texts)


class TextDatasetConstructionSingleLabelTest(unittest.TestCase):

    def test_from_arrays_with_lists(self):
        texts = [f'train data {i}' for i in range(10)]
        labels = np.array([random_labeling(3) for _ in range(10)])

        dataset = TextDataset.from_arrays(texts, labels)

        self.assertEqual(10, len(dataset))
        self.assertTrue(isinstance(dataset.x, list))
        assert_array_equal(texts, dataset.x)
        assert_array_equal(labels, dataset.y)

    def test_from_arrays_with_ndarray(self):
        texts = np.array([f'train data {i}' for i in range(10)])
        labels = np.array([random_labeling(3) for _ in range(10)])

        dataset = TextDataset.from_arrays(texts, labels)
        self.assertTrue(isinstance(dataset.x, list))
        assert_array_equal(texts, dataset.x)
        assert_array_equal(labels, dataset.y)


class TextDatasetConstructionMultiLabelTest(unittest.TestCase):

    def test_from_arrays_with_lists(self):
        texts = [f'train data {i}' for i in range(10)]
        labels = random_labels(10, 3, multi_label=True)

        dataset = TextDataset.from_arrays(texts, labels)

        self.assertEqual(10, len(dataset))
        self.assertTrue(isinstance(dataset.x, list))
        assert_array_equal(texts, dataset.x)
        assert_csr_matrix_equal(labels, dataset.y)

    def test_from_arrays_with_ndarray(self):
        texts = np.array([f'train data {i}' for i in range(10)])
        labels = random_labels(10, 3, multi_label=True)

        dataset = TextDataset.from_arrays(texts, labels)
        self.assertTrue(isinstance(dataset.x, list))
        assert_array_equal(texts, dataset.x)
        assert_csr_matrix_equal(labels, dataset.y)
