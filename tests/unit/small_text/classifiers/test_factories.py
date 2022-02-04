import unittest

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.validation import check_is_fitted, NotFittedError
from small_text.classifiers.classification import SklearnClassifier
from small_text.classifiers.factories import SklearnClassifierFactory

from tests.utils.datasets import random_sklearn_dataset


class SklearnClassifierFactoryTest(unittest.TestCase):

    def test_init_with_invalid_estimator(self):
        class MyCustomEstimator(object):
            pass

        base_estimator = MyCustomEstimator()
        num_classes = 2

        with self.assertRaisesRegex(ValueError, 'Given classifier template must be a subclass'):
            SklearnClassifierFactory(base_estimator, num_classes)

    def test_new(self):
        base_estimator = LinearSVC()
        num_classes = 2

        clf_factory = SklearnClassifierFactory(base_estimator, num_classes)
        clf = clf_factory.new()

        self.assertTrue(isinstance(clf, SklearnClassifier))
        self.assertTrue(isinstance(clf.model, LinearSVC))
        self.assertEqual(num_classes, clf.num_classes)

    def test_new_with_multi_label_kwarg(self):
        kwargs = dict({'multi_label': True})
        base_estimator = LinearSVC()
        num_classes = 2

        clf_factory = SklearnClassifierFactory(base_estimator, num_classes, kwargs=kwargs)
        clf = clf_factory.new()

        self.assertTrue(isinstance(clf, SklearnClassifier))
        self.assertTrue(isinstance(clf.model, OneVsRestClassifier))
        self.assertEqual(num_classes, clf.num_classes)
        self.assertTrue(clf.multi_label)

    def test_clone_resets_classifier(self):
        base_estimator = LinearSVC(max_iter=10)
        num_classes = 2

        clf_factory = SklearnClassifierFactory(base_estimator, num_classes)
        clf = clf_factory.new()
        clf_two = clf_factory.new()
        self.assertNotEqual(clf, clf_two)

        with self.assertRaises(NotFittedError):
            check_is_fitted(clf.model)
        with self.assertRaises(NotFittedError):
            check_is_fitted(clf_two.model)

        ds = random_sklearn_dataset(100)
        clf.fit(ds)

        check_is_fitted(clf.model)
        with self.assertRaises(NotFittedError):
            check_is_fitted(clf_two.model)
