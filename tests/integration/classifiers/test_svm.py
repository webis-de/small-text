import unittest

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from small_text.classifiers import ConfidenceEnhancedLinearSVC


class ConfidenceEnhancedLinearSVCIntegrationTest(unittest.TestCase):

    def _get_20news_vectors(self, categories=None):
        train = fetch_20newsgroups(subset='train', categories=categories)
        vectorizer = CountVectorizer()
        x = vectorizer.fit_transform(train.data)

        return x, train.target

    def test_predict_binary(self):

        clf = ConfidenceEnhancedLinearSVC()
        x, y = self._get_20news_vectors(categories=['comp.graphics', 'sci.med'])

        clf.fit(x, y)

        y_pred = clf.predict(x)

        self.assertEqual(1, len(y_pred.shape))
        self.assertEqual(x.shape[0], y_pred.shape[0])
        self.assertTrue(all((y_pred == 0) | (y_pred == 1)))

    def test_predict_binary_with_probas(self):

        clf = ConfidenceEnhancedLinearSVC()
        x, y = self._get_20news_vectors(categories=['comp.graphics', 'sci.med'])

        clf.fit(x, y)

        y_pred, proba = clf.predict(x, return_proba=True)

        self.assertEqual(1, len(y_pred.shape))
        self.assertEqual(x.shape[0], y_pred.shape[0])
        self.assertTrue(all((y_pred == 0) | (y_pred == 1)))

        self.assertEqual(2, len(proba.shape))
        self.assertEqual(x.shape[0], proba.shape[0])
        self.assertEqual(2, proba.shape[1])
        self.assertTrue(all((proba.flatten() >= 0) & (proba.flatten() <= 1)))

    def test_predict_multiclass(self):
        clf = ConfidenceEnhancedLinearSVC()
        categories = ['alt.atheism', 'comp.graphics', 'sci.med']
        x, y = self._get_20news_vectors(categories=categories)

        clf.fit(x, y)

        y_pred = clf.predict(x)

        self.assertEqual(1, len(y_pred.shape))
        self.assertEqual(x.shape[0], y_pred.shape[0])
        self.assertTrue(all((y_pred >= 0) & (y_pred <= 2)))

    def test_predict_multiclass_with_probas(self):
        clf = ConfidenceEnhancedLinearSVC()
        categories = ['alt.atheism', 'comp.graphics', 'sci.med']
        x, y = self._get_20news_vectors(categories=categories)

        clf.fit(x, y)

        y_pred, proba = clf.predict(x, return_proba=True)

        self.assertEqual(1, len(y_pred.shape))
        self.assertEqual(x.shape[0], y_pred.shape[0])
        self.assertTrue(all((y_pred >= 0) & (y_pred <= 2)))

        self.assertEqual(2, len(proba.shape))
        self.assertEqual(x.shape[0], proba.shape[0])
        self.assertEqual(len(categories), proba.shape[1])
        self.assertTrue(all((proba.flatten() >= 0) & (proba.flatten() <= 1)))

    def test_predict_proba_binary(self):

        clf = ConfidenceEnhancedLinearSVC()
        x, y = self._get_20news_vectors(categories=['comp.graphics', 'sci.med'])

        clf.fit(x, y)

        proba = clf.predict_proba(x)

        self.assertEqual(2, len(proba.shape))
        self.assertEqual(x.shape[0], proba.shape[0])
        self.assertEqual(2, proba.shape[1])
        self.assertTrue(all((proba.flatten() >= 0) & (proba.flatten() <= 1)))

    def test_predict_proba_multiclass(self):

        clf = ConfidenceEnhancedLinearSVC()
        categories = ['alt.atheism', 'comp.graphics', 'sci.med']
        x, y = self._get_20news_vectors(categories=categories)

        clf.fit(x, y)

        proba = clf.predict_proba(x)

        self.assertEqual(2, len(proba.shape))
        self.assertEqual(x.shape[0], proba.shape[0])
        self.assertEqual(len(categories), proba.shape[1])
        self.assertTrue(all((proba.flatten() >= 0) & (proba.flatten() <= 1)))
