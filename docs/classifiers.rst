===========
Classifiers
===========

In order to use different models, query strategies, and stopping criteria from the active learner,
we provide classification abstractions to allow for a unified interface.

Interface
=========

The classifier interface is very simple and scikit-learn-like, with the difference that it operates on
:py:class:`Datasets<small_text.data.datasets.Dataset>` objects. Call the :code:`fit()` method with a
training set as argument to train your classifier, and use :code:`predict()` to obtain predictions.

.. literalinclude:: ../small_text/classifiers/classification.py
   :pyobject: Classifier

Example
=======

This is a simple example which shows the training of a tiny toy dataset.

.. testcode::

   import numpy as np
   from small_text.classifiers import ConfidenceEnhancedLinearSVC, SklearnClassifier
   from small_text.data import SklearnDataset

   # this is a linear which has been extended to return confidence estimates
   model = ConfidenceEnhancedLinearSVC()
   num_classes = 2
   clf = SklearnClassifier(model, num_classes)

   x = np.array([
       [0, 0],
       [0, 0.5],
       [0.5, 1],
       [1, 1]
   ])
   y = np.array([0, 0, 1, 1])
   train_set = SklearnDataset(x, y)
   clf.fit(train_set)

   """
   Generate predictions on the train set
   (Only for the purpose of demonstration;
    usually you would be more interested in obtaining predictions on new, unseen data.)
   """
   y_train_pred = clf.predict(train_set)
   print(y_train_pred)

*Output:*

.. testoutput::

   [0 0 1 1]

Factories
=========

To configure the active learner to use classifiers a factory object is required because
new classifier objects are created at each iteration (unless explicitly configured not to).
A factory creates new instances of an object, for which the knowledge of what to pass to
the constructor is required, which is why we need a factory.
Assuming all constructor took zero arguments we would not need factories here.

.. testcode::

   from small_text.classifiers import ConfidenceEnhancedLinearSVC
   from small_text.classifiers.factories import SklearnClassifierFactory

   clf_template = ConfidenceEnhancedLinearSVC()
   num_classes = 2

   clf_factory = SklearnClassifierFactory(clf_template, num_classes)
   clf = clf_factory.new()


This also means that any classifier parameters, e.g. for **multi-label** classification, are managed by the factory:


.. testcode::

   from small_text.classifiers import ConfidenceEnhancedLinearSVC
   from small_text.classifiers.factories import SklearnClassifierFactory

   clf_template = ConfidenceEnhancedLinearSVC()
   num_classes = 2
   classifier_kwargs = {'multi_label': True}

   clf_factory = SklearnClassifierFactory(clf_template, num_classes, kwargs=classifier_kwargs)
   clf = clf_factory.new()
