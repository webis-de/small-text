import unittest

import numpy as np

from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer

from small_text.utils.classification import empty_result, prediction_result
from tests.utils.testing import assert_csr_matrix_equal


class ClassificationUtilsTest(unittest.TestCase):

    def test_prediction_result(self):
        proba = np.array([
            [0.1, 0.2, 0.6, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.3, 0.3, 0.2, 0.2],
            [0.3, 0.2, 0.5, 0.1],
        ])
        result = prediction_result(proba, False, proba.shape[1])
        expected = np.array([2, 0, 0, 2])
        assert_array_equal(expected, result)

    def test_prediction_result_with_proba(self):
        proba = np.array([
            [0.1, 0.2, 0.6, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.3, 0.3, 0.2, 0.2],
            [0.3, 0.2, 0.5, 0.1],
        ])
        result, proba_result = prediction_result(proba, False, proba.shape[1], return_proba=True)
        expected = np.array([2, 0, 0, 2])
        assert_array_equal(expected, result)
        assert_array_equal(proba, proba_result)

    def test_prediction_result_multilabel(self):
        proba = np.array([
            [0.1, 0.2, 0.6, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.3, 0.3, 0.2, 0.2],
            [0.3, 0.2, 0.5, 0.1],
        ])
        result = prediction_result(proba, True, proba.shape[1])
        expected = csr_matrix(np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]))
        assert_csr_matrix_equal(expected, result)

    def test_prediction_result_multilabel_with_proba(self):
        proba = np.array([
            [0.1, 0.2, 0.6, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.3, 0.3, 0.2, 0.2],
            [0.3, 0.2, 0.5, 0.1],
        ])
        result, proba_result = prediction_result(proba, True, proba.shape[1], return_proba=True)
        expected = csr_matrix(np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]))
        assert_csr_matrix_equal(expected, result)
        expected_proba = csr_matrix(np.array([
            [0, 0, 0.6, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]))
        assert_csr_matrix_equal(expected_proba, proba_result)

    # TODO: remove this in 2.0.0
    def test_prediction_result_multilabel_with_enc(self):
        all_labels = [[0], [0, 1], [2, 3]]
        enc = MultiLabelBinarizer()
        enc.fit(all_labels)

        proba = np.array([
            [0.1, 0.2, 0.6, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.3, 0.3, 0.2, 0.2],
            [0.3, 0.2, 0.5, 0.1],
        ])
        with self.assertWarnsRegex(DeprecationWarning,
                                   'The enc keyword argument has been deprecated since 1.1.0'):
            result = prediction_result(proba, True, proba.shape[1], enc=enc)
        expected = csr_matrix(np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]))
        assert_csr_matrix_equal(expected, result)

    # TODO: remove this in 2.0.0
    def test_prediction_result_multilabel_with_enc_and_proba(self):
        all_labels = [[0], [0, 1], [2, 3]]
        enc = MultiLabelBinarizer()
        enc.fit(all_labels)

        proba = np.array([
            [0.1, 0.2, 0.6, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.3, 0.3, 0.2, 0.2],
            [0.3, 0.2, 0.5, 0.1],
        ])

        with self.assertWarnsRegex(DeprecationWarning,
                                   'The enc keyword argument has been deprecated since 1.1.0'):
            result, proba_result = prediction_result(proba, True, proba.shape[1], enc=enc,
                                                     return_proba=True)
        expected = csr_matrix(np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]))
        assert_csr_matrix_equal(expected, result)
        expected_proba = csr_matrix(np.array([
            [0, 0, 0.6, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]))
        assert_csr_matrix_equal(expected_proba, proba_result)

    def test_empty_result_single_label_prediction(self):
        num_labels = 3
        prediction = empty_result(False, num_labels, return_proba=False)

        self.assertTrue(isinstance(prediction, np.ndarray))
        self.assertEqual(np.int64, prediction.dtype)
        self.assertEqual((0, 3), prediction.shape)

    def test_empty_result_single_label_proba(self):
        num_labels = 3
        proba = empty_result(False, num_labels, return_prediction=False)

        self.assertTrue(isinstance(proba, np.ndarray))
        self.assertEqual(float, proba.dtype)
        self.assertEqual((0, 3), proba.shape)

    def test_empty_result_single_label_both(self):
        num_labels = 3
        prediction, proba = empty_result(False, num_labels)

        self.assertTrue(isinstance(prediction, np.ndarray))
        self.assertEqual(np.int64, prediction.dtype)
        self.assertEqual((0, 3), prediction.shape)

        self.assertTrue(isinstance(proba, np.ndarray))
        self.assertEqual(float, proba.dtype)
        self.assertEqual((0, 3), proba.shape)

    def test_empty_result_invalid_call(self):
        num_labels = 3
        multi_label_args = [True, False]
        for multi_label in multi_label_args:
            with self.assertRaisesRegex(ValueError, 'Invalid usage: At least one of'):
                empty_result(multi_label, num_labels, return_prediction=False, return_proba=False)
