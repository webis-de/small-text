===============
Classifier API
===============

.. contents:: Overview
   :depth: 2
   :local:
   :backlinks: none

----

Classifiers
===========

Core
----

.. currentmodule:: small_text.classifiers.classification

.. autoclass:: Classifier
    :members: fit, predict, predict_proba

.. autoclass:: SklearnClassifier
    :members: fit, predict, predict_proba
    :member-order: bysource
    :special-members: __init__

Pytorch Integration
-------------------

.. currentmodule:: small_text.integrations.pytorch.classifiers

.. _api-classifiers-kimcnn-classifier:

.. autoclass:: KimCNNClassifier
    :members: fit, predict, predict_proba, validate
    :member-order: bysource
    :special-members: __init__

Transformers Integration
------------------------

.. currentmodule:: small_text.integrations.transformers.classifiers

.. _api-classifiers-transformer-based-classification:

.. autoclass:: TransformerBasedClassification
    :members: fit, predict, predict_proba, validate
    :member-order: bysource
    :special-members: __init__

.. autoclass:: small_text.integrations.transformers.classifiers.classification.TransformerModelArguments
   :special-members: __init__
   :members:

.. autoclass:: small_text.integrations.transformers.classifiers.classification.FineTuningArguments
   :special-members: __init__
   :members:

.. autoclass:: small_text.integrations.transformers.classifiers.setfit.SetFitClassification
   :special-members: __init__
   :members:

.. autoclass:: small_text.integrations.transformers.classifiers.setfit.SetFitModelArguments
   :special-members: __init__
   :members:


Factories
=========

Core
----

.. currentmodule:: small_text.classifiers.factories

.. autoclass:: SklearnClassifierFactory
    :members: new
    :member-order: bysource
    :special-members: __init__

Pytorch Integration
-------------------

.. currentmodule:: small_text.integrations.pytorch.classifiers.factories

.. autoclass:: KimCNNFactory
    :members: new
    :member-order: bysource
    :special-members: __init__

Transformers Integration
------------------------

.. currentmodule:: small_text.integrations.transformers.classifiers.factories

.. autoclass:: TransformerBasedClassificationFactory
    :members: new
    :member-order: bysource
    :special-members: __init__


.. autoclass:: SetFitClassificationFactory
    :members: new
    :member-order: bysource
    :special-members: __init__
