.. py:currentmodule:: small_text.query_strategies

================
Query Strategies
================

Query strategies select data samples from the set of unlabeled data.

Overview
========

General
-------

* :py:class:`LeastConfidence`
* :py:class:`PredictionEntropy`
* :py:class:`BreakingTies`
* :py:class:`EmbeddingKMeans`
* :py:class:`RandomSampling`

.. py:currentmodule:: small_text.integrations.pytorch.query_strategies

Pytorch
-------
* :py:class:`ExpectedGradientLength`
* :py:class:`ExpectedGradientLengthMaxWord`
* :py:class:`ExpectedGradientLengthLayer`
* :py:class:`BADGE`


Helpers
=======

Constraints
-----------

.. testcode::

    from small_text.query_strategies import constraints, QueryStrategy

    @constraints(classification_type='single-label')
    class MyQueryStrategy(QueryStrategy):
        pass


Classes
=======

.. py:module:: small_text.query_strategies

.. autoclass:: LeastConfidence
    :inherited-members:

.. autoclass:: PredictionEntropy
    :inherited-members:

.. autoclass:: EmbeddingKMeans
    :inherited-members:

.. autoclass:: RandomSampling
    :inherited-members:

.. py:module:: small_text.integrations.pytorch.query_strategies

.. autoclass:: ExpectedGradientLength
    :inherited-members:

.. autoclass:: ExpectedGradientLengthMaxWord
    :inherited-members:

.. autoclass:: ExpectedGradientLengthLayer
    :inherited-members:
