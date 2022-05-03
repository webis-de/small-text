================
Query Strategies
================

Query strategies select data samples from the set of unlabeled data.

Overview
========

You can use the following pre-implemented query strategies:

General
-------

.. py:currentmodule:: small_text.query_strategies.strategies

* :py:class:`LeastConfidence`
* :py:class:`PredictionEntropy`
* :py:class:`BreakingTies`
* :py:class:`EmbeddingKMeans`
* :py:class:`~small_text.query_strategies.coresets.GreedyCoreset`
* :py:class:`~small_text.query_strategies.coresets.LightweightCoreset`
* :py:class:`ContrastiveActiveLearning`
* :py:class:`DiscriminativeActiveLearning`
* :py:class:`SEALS`
* :py:class:`RandomSampling`


Pytorch
-------

.. py:currentmodule:: small_text.integrations.pytorch.query_strategies.strategies

* :py:class:`ExpectedGradientLength`
* :py:class:`ExpectedGradientLengthMaxWord`
* :py:class:`ExpectedGradientLengthLayer`
* :py:class:`BADGE`


Helpers
=======

Some query strategies may be formulated so that are only applicable to either single-label or
multi-label data. As a safeguard against using such strategies on data which is not supported,
the `constraints()` decorator intercepts the `query()`. If the given labels cannot be handled,
`RuntimeError` is raised.

.. note:: For all pre-implemented query strategies, don't equate an absence of an constraint
          as an indicator of capibility, since we will sparingly use this in the main library
          in order to not restrict the user unnecessarily.
          For your own projects and applications, however, this is highly recommended.

Constraints
-----------

.. testcode::

    from small_text.query_strategies import constraints, QueryStrategy

    @constraints(classification_type='single-label')
    class MyQueryStrategy(QueryStrategy):
        pass


Classes
=======

Base
----

.. py:module:: small_text.query_strategies.strategies

.. autoclass:: LeastConfidence

.. autoclass:: PredictionEntropy

.. autoclass:: BreakingTies

.. autoclass:: EmbeddingKMeans
    :special-members: __init__

.. py:module:: small_text.query_strategies.coresets

.. autoclass:: GreedyCoreset
    :special-members: __init__

.. autoclass:: LightweightCoreset
    :special-members: __init__

.. py:module:: small_text.query_strategies.strategies
    :noindex:

.. autoclass:: ContrastiveActiveLearning
    :special-members: __init__

.. autoclass:: DiscriminativeActiveLearning
    :special-members: __init__

.. autoclass:: SEALS
    :special-members: __init__

.. autoclass:: RandomSampling


Pytorch Integration
-------------------

.. py:module:: small_text.integrations.pytorch.query_strategies.strategies

.. autoclass:: ExpectedGradientLength
    :special-members: __init__

.. autoclass:: ExpectedGradientLengthMaxWord
    :special-members: __init__

.. autoclass:: ExpectedGradientLengthLayer
    :special-members: __init__

.. autoclass:: BADGE
    :special-members: __init__
