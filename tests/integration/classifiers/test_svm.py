import unittest

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from active_learning.classifiers import ConfidenceEnhancedLinearSVC


class ConfidenceEnhancedLinearSVCIntegrationTest(unittest.TestCase):

    def _get_20news_vectors(self, categories=None):
        train = fetch_20newsgroups(subset='train', categories=categories)
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(train.data)

        return X, train.target

    def test_predict_binary(self):

        clf = ConfidenceEnhancedLinearSVC()
        X, y = self._get_20news_vectors(categories=['comp.graphics', 'sci.med'])

        clf.fit(X, y)

        y_pred = clf.predict(X)

        self.assertEqual(1, len(y_pred.shape))
        self.assertEqual(X.shape[0], y_pred.shape[0])
        self.assertTrue(all((y_pred == 0) | (y_pred == 1)))

    def test_predict_binary_with_probas(self):

        clf = ConfidenceEnhancedLinearSVC()
        X, y = self._get_20news_vectors(categories=['comp.graphics', 'sci.med'])

        clf.fit(X, y)

        y_pred, proba = clf.predict(X, return_proba=True)

        self.assertEqual(1, len(y_pred.shape))
        self.assertEqual(X.shape[0], y_pred.shape[0])
        self.assertTrue(all((y_pred == 0) | (y_pred == 1)))

        self.assertEqual(2, len(proba.shape))
        self.assertEqual(X.shape[0], proba.shape[0])
        self.assertEqual(2, proba.shape[1])
        self.assertTrue(all((proba.flatten() >= 0) & (proba.flatten() <= 1)))

    def test_predict_multiclass(self):
        clf = ConfidenceEnhancedLinearSVC()
        categories = ['alt.atheism', 'comp.graphics', 'sci.med']
        X, y = self._get_20news_vectors(categories=categories)

        clf.fit(X, y)

        y_pred = clf.predict(X)

        self.assertEqual(1, len(y_pred.shape))
        self.assertEqual(X.shape[0], y_pred.shape[0])
        self.assertTrue(all((y_pred >= 0) & (y_pred <= 2)))

    def test_predict_multiclass_with_probas(self):
        clf = ConfidenceEnhancedLinearSVC()
        categories = ['alt.atheism', 'comp.graphics', 'sci.med']
        X, y = self._get_20news_vectors(categories=categories)

        clf.fit(X, y)

        y_pred, proba = clf.predict(X, return_proba=True)

        self.assertEqual(1, len(y_pred.shape))
        self.assertEqual(X.shape[0], y_pred.shape[0])
        self.assertTrue(all((y_pred >= 0) & (y_pred <= 2)))

        self.assertEqual(2, len(proba.shape))
        self.assertEqual(X.shape[0], proba.shape[0])
        self.assertEqual(len(categories), proba.shape[1])
        self.assertTrue(all((proba.flatten() >= 0) & (proba.flatten() <= 1)))

    def test_predict_proba_binary(self):

        clf = ConfidenceEnhancedLinearSVC()
        X, y = self._get_20news_vectors(categories=['comp.graphics', 'sci.med'])

        clf.fit(X, y)

        proba = clf.predict_proba(X)

        self.assertEqual(2, len(proba.shape))
        self.assertEqual(X.shape[0], proba.shape[0])
        self.assertEqual(2, proba.shape[1])
        self.assertTrue(all((proba.flatten() >= 0) & (proba.flatten() <= 1)))

    def test_predict_proba_multiclass(self):

        clf = ConfidenceEnhancedLinearSVC()
        categories = ['alt.atheism', 'comp.graphics', 'sci.med']
        X, y = self._get_20news_vectors(categories=categories)

        clf.fit(X, y)

        proba = clf.predict_proba(X)

        self.assertEqual(2, len(proba.shape))
        self.assertEqual(X.shape[0], proba.shape[0])
        self.assertEqual(len(categories), proba.shape[1])
        self.assertTrue(all((proba.flatten() >= 0) & (proba.flatten() <= 1)))
