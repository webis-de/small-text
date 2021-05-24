.. py:currentmodule:: active_learning.query_strategies

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
* :py:class:`RandomSampling`

.. py:currentmodule:: active_learning.integrations.pytorch.query_strategies

Pytorch
-------
* :py:class:`ExpectedGradientLength`
* :py:class:`ExpectedGradientLengthMaxWord`
* :py:class:`ExpectedGradientLengthLayer`
* :py:class:`BADGE`


Classes
-------

.. py:module:: active_learning.query_strategies

.. autoclass:: LeastConfidence
    :inherited-members:

.. autoclass:: RandomSampling
    :inherited-members:

.. py:module:: active_learning.integrations.pytorch.query_strategies

.. autoclass:: ExpectedGradientLength
    :inherited-members:

.. autoclass:: ExpectedGradientLengthMaxWord
    :inherited-members:

.. autoclass:: ExpectedGradientLengthLayer
    :inherited-members:
