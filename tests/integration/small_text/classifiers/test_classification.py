import unittest
import pytest

import numpy as np

from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_is_fitted
from small_text.classifiers.classification import ConfidenceEnhancedLinearSVC
from small_text.classifiers.classification import SklearnClassifier
from small_text.data.datasets import SklearnDataset
from tests.utils.datasets import random_matrix_data


class _SklearnClassifierBaseTest(object):

    def get_dataset(self):
        raise NotImplementedError('get_dataset() must be overridden')

    def get_clf(self):
        raise NotImplementedError('get_clf() must be overridden')

    def test_fit_and_predict(self):
        ds = self.get_dataset()
        clf = self.get_clf()

        clf.fit(ds)
        check_is_fitted(clf.model)

        predictions = clf.predict(ds)
        if isinstance(ds.y, csr_matrix):
            self.assertTrue(isinstance(predictions, csr_matrix))
        else:
            self.assertTrue(isinstance(predictions, np.ndarray))

        proba = clf.predict_proba(ds)
        self.assertTrue(isinstance(proba, np.ndarray))


@pytest.mark.pytorch
class LinearSVCSingleLabelTest(unittest.TestCase, _SklearnClassifierBaseTest):

    def get_dataset(self, num_samples=100):
        return SklearnDataset(*random_matrix_data('dense', 'dense', num_samples=num_samples,
                                                  num_labels=3))

    def get_clf(self):
        return SklearnClassifier(ConfidenceEnhancedLinearSVC(), 3)

    def test_fit_and_predict_with_weights(self):
        ds = self.get_dataset()
        clf = self.get_clf()

        weights = np.random.randn(len(ds))
        weights = weights - weights.min() + 1e-8

        clf.fit(ds, weights=weights)
        check_is_fitted(clf.model)

        predictions = clf.predict(ds)
        if isinstance(ds.y, csr_matrix):
            self.assertTrue(isinstance(predictions, csr_matrix))
        else:
            self.assertTrue(isinstance(predictions, np.ndarray))

        proba = clf.predict_proba(ds)
        self.assertTrue(isinstance(proba, np.ndarray))


@pytest.mark.pytorch
class LinearSVCMultiLabelTest(unittest.TestCase, _SklearnClassifierBaseTest):

    def get_dataset(self, num_samples=100):
        return SklearnDataset(*random_matrix_data('dense', 'sparse', num_samples=num_samples,
                                                  num_labels=3))

    def get_clf(self):
        return SklearnClassifier(ConfidenceEnhancedLinearSVC(), 3, multi_label=True)

    def test_fit_and_predict_with_weights(self):
        ds = self.get_dataset()
        clf = self.get_clf()

        weights = np.random.randn(len(ds))
        weights = weights - weights.min() + 1e-8

        with self.assertRaisesRegex(ValueError, 'Sample weights are not supported'):
            clf.fit(ds, weights=weights)
