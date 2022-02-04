import unittest

import numpy as np

from scipy.sparse import csr_matrix

from small_text.classifiers.classification import ConfidenceEnhancedLinearSVC, SklearnClassifier
from small_text.data.datasets import SklearnDataset
from tests.utils.datasets import random_sklearn_dataset, random_matrix_data


class _ClassifierBaseFunctionalityTest(object):

    def _get_clf(self):
        raise NotImplementedError()

    def _is_multi_label(self):
        raise NotImplementedError()

    def test_predict_on_empty_data(self):
        train_set = random_sklearn_dataset(10,
                                           num_classes=3 if self._is_multi_label() else 2,
                                           multi_label=self._is_multi_label())
        test_set = SklearnDataset(np.array([], dtype=int), np.array([], dtype=int))

        clf = self._get_clf()
        clf.fit(train_set)

        predictions = clf.predict(test_set)
        self.assertEqual(0, predictions.shape[0])
        self.assertTrue(np.issubdtype(predictions.dtype, np.integer))

    def test_predict_proba_on_empty_data(self):
        train_set = random_sklearn_dataset(10,
                                           num_classes=3 if self._is_multi_label() else 2,
                                           multi_label=self._is_multi_label())
        test_set = SklearnDataset(np.array([], dtype=int), np.array([], dtype=int))

        clf = self._get_clf()
        clf.fit(train_set)

        proba = clf.predict_proba(test_set)
        self.assertEqual(0, proba.shape[0])
        if self._is_multi_label():
            self.assertTrue(isinstance(proba, csr_matrix))
        else:
            self.assertTrue(isinstance(proba, np.ndarray))
            self.assertTrue(np.issubdtype(proba.dtype, np.float))


class SklearnClassifierSingleLabelTest(unittest.TestCase, _ClassifierBaseFunctionalityTest):

    def _get_clf(self):
        return SklearnClassifier(ConfidenceEnhancedLinearSVC(), 2)

    def _is_multi_label(self):
        return False

    def test_fit_with_multi_label_data_on_single_label_classifier(self):
        train_set = SklearnDataset(*random_matrix_data('dense',
                                                       'sparse',
                                                       num_samples=10,
                                                       num_labels=3))
        clf = self._get_clf()

        expected_str = 'Given labeling is recognized as multi-label labeling but the classifier '
        with self.assertRaisesRegex(ValueError, expected_str):
            clf.fit(train_set)


class SklearnClassifierMultiLabelTest(unittest.TestCase, _ClassifierBaseFunctionalityTest):

    def _get_clf(self):
        return SklearnClassifier(ConfidenceEnhancedLinearSVC(),
                                 3,
                                 multi_label=True)

    def _is_multi_label(self):
        return True

    def test_fit_with_invalid_multi_label_data(self):
        train_set = SklearnDataset(*random_matrix_data('dense',
                                                       'sparse',
                                                       num_samples=10,
                                                       num_labels=3))
        # possibility of error: more than one unique value in y.data
        train_set.y.data = np.array([0] * 5 + [1] * 5, dtype=train_set.y.data.dtype)
        clf = self._get_clf()

        expected_str = 'Invalid input: Given labeling must be recognized as multi-label'
        with self.assertRaisesRegex(ValueError, expected_str):
            clf.fit(train_set)
