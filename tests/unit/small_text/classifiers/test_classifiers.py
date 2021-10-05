import unittest

import numpy as np

from small_text.classifiers.classification import ConfidenceEnhancedLinearSVC, SklearnClassifier
from small_text.data.datasets import SklearnDataSet
from tests.utils.datasets import random_sklearn_dataset


class _ClassifierBaseFunctionalityTest(object):

    def _get_clf(self):
        raise NotImplementedError()

    def test_predict_on_empty_data(self):
        train_set = random_sklearn_dataset(10)
        test_set = SklearnDataSet(np.array([], dtype=int), np.array([], dtype=int))

        clf = self._get_clf()
        clf.fit(train_set)

        predictions = clf.predict(test_set)
        self.assertEqual(0, predictions.shape[0])
        self.assertTrue(np.issubdtype(predictions.dtype, np.integer))

    def test_predict_proba_on_empty_data(self):
        train_set = random_sklearn_dataset(10)
        test_set = SklearnDataSet(np.array([], dtype=int), np.array([], dtype=int))

        clf = self._get_clf()
        clf.fit(train_set)

        predictions, proba = clf.predict_proba(test_set)
        self.assertEqual(0, predictions.shape[0])
        self.assertTrue(np.issubdtype(predictions.dtype, np.integer))
        self.assertEqual(0, proba.shape[0])
        self.assertTrue(np.issubdtype(proba.dtype, np.float))


class SklearnClassifierBaseFunctionalityTest(unittest.TestCase,_ClassifierBaseFunctionalityTest):

    def _get_clf(self):
        return SklearnClassifier(ConfidenceEnhancedLinearSVC())
