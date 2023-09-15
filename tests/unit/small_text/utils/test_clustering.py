import unittest
import pytest
import numpy as np

from unittest.mock import patch

from small_text.utils.clustering import init_kmeans_plusplus_safe


class ClusteringUtilsTest(unittest.TestCase):

    @patch('small_text.utils.clustering.warnings.warn')
    @patch('small_text.utils.clustering.choice')
    @patch('small_text.utils.clustering.kmeans_plusplus')
    def test_init_kmeans_plusplus_safe(self, kmeans_plusplus_mock, choice_mock, warn_mock):
        x = np.random.rand(100, 10)

        result_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 42, 99])
        kmeans_plusplus_mock.return_value = [x[result_indices], result_indices]

        centers, indices = init_kmeans_plusplus_safe(x, 10)
        kmeans_plusplus_mock.assert_called()
        choice_mock.assert_not_called()
        warn_mock.assert_not_called()
        self.assertEqual((10, 10), centers.shape)
        self.assertEqual(10, indices.shape[0])

    def test_init_kmeans_plusplus_safe_weights(self):
        x = np.random.rand(100, 10)
        weights = np.random.rand(100,)

        with patch('sklearn.__version__', new='1.2.0'):
            with self.assertRaisesRegex(ValueError, r'scikit-learn>=1.3.0 is required'):
                init_kmeans_plusplus_safe(x, 10, weights=weights)

    @patch('small_text.utils.clustering.choice', wraps=np.random.choice)
    @patch('small_text.utils.clustering.kmeans_plusplus')
    def test_init_kmeans_plusplus_safe_duplicate_indices(self, kmeans_plusplus_mock, choice_mock):
        x = np.random.rand(100, 10)

        result_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 42, 42])  # 42 is not unique here
        kmeans_plusplus_mock.return_value = [x[result_indices], result_indices]

        with self.assertWarnsRegex(UserWarning,
                                   'kmeans_plusplus returned identical cluster centers'):
            centers, indices = init_kmeans_plusplus_safe(x, 10)

        kmeans_plusplus_mock.assert_called()
        choice_mock.assert_called()
        self.assertEqual((10, 10), centers.shape)
        self.assertEqual(10, indices.shape[0])

    @patch('small_text.utils.clustering.warnings.warn')
    @patch('small_text.utils.clustering.choice', wraps=np.random.choice)
    @patch('small_text.utils.clustering.kmeans_plusplus')
    def test_init_kmeans_plusplus_safe_duplicate_indices_warning_suppressed(
            self, kmeans_plusplus_mock, choice_mock, warn_mock):
        x = np.random.rand(100, 10)

        result_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 42, 42])  # 42 is not unique here
        kmeans_plusplus_mock.return_value = [x[result_indices], result_indices]

        centers, indices = init_kmeans_plusplus_safe(x, 10, suppress_warning=True)
        kmeans_plusplus_mock.assert_called()
        choice_mock.assert_called()
        warn_mock.assert_not_called()
        self.assertEqual((10, 10), centers.shape)
        self.assertEqual(10, indices.shape[0])


@pytest.mark.optional
class ClusteringUtilsRecentScikitLearnVersionTest(unittest.TestCase):

    @patch('small_text.utils.clustering.warnings.warn')
    @patch('small_text.utils.clustering.choice')
    @patch('small_text.utils.clustering.kmeans_plusplus')
    def test_init_kmeans_plusplus_safe_normal(self, kmeans_plusplus_mock, choice_mock, warn_mock):
        x = np.random.rand(100, 10)
        weights = np.random.rand(100,)

        result_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 42, 99])
        kmeans_plusplus_mock.return_value = [x[result_indices], result_indices]

        centers, indices = init_kmeans_plusplus_safe(x, 10, weights=weights)
        kmeans_plusplus_mock.assert_called()
        choice_mock.assert_not_called()
        warn_mock.assert_not_called()
        self.assertEqual((10, 10), centers.shape)
        self.assertEqual(10, indices.shape[0])
