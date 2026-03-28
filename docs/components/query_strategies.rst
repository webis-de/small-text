================
Query Strategies
================

Query strategies decide which samples from the unlabeled pool should be labeled next.
small-text provides a broad set of strategies, ranging from simple baselines to
embedding-based, multi-label, and gradient-based methods.

----

Choosing a Strategy
===================

When you compare or build active learning setups, it helps to first decide which signal a
strategy should rely on:

- Prediction uncertainty:
  :py:class:`~small_text.query_strategies.strategies.LeastConfidence`,
  :py:class:`~small_text.query_strategies.strategies.PredictionEntropy`,
  :py:class:`~small_text.query_strategies.strategies.BreakingTies`, and
  :py:class:`~small_text.query_strategies.bayesian.BALD` use model predictions to find
  uncertain instances.
- Embedding structure:
  :py:class:`~small_text.query_strategies.strategies.EmbeddingKMeans`,
  :py:class:`~small_text.query_strategies.coresets.GreedyCoreset`,
  :py:class:`~small_text.query_strategies.coresets.LightweightCoreset`,
  :py:class:`~small_text.query_strategies.strategies.ContrastiveActiveLearning`,
  and :py:class:`~small_text.query_strategies.vector_space.ProbCover` rely on dense
  representations to encourage diversity or coverage.
- Labeled-vs-unlabeled pool separation:
  :py:class:`~small_text.query_strategies.strategies.DiscriminativeActiveLearning`
  explicitly learns to distinguish the labeled pool from the unlabeled pool and selects
  instances that look most like the remaining unlabeled data.
- Pool reduction and balancing:
  :py:class:`~small_text.query_strategies.class_balancing.ClassBalancer`,
  :py:class:`~small_text.query_strategies.subsampling.SEALS`, and
  :py:class:`~small_text.query_strategies.subsampling.AnchorSubsampling` wrap another
  strategy or reduce the candidate pool before querying.
- Multi-label specific selection:
  :py:class:`~small_text.query_strategies.multi_label.CategoryVectorInconsistencyAndRanking`,
  :py:class:`~small_text.query_strategies.multi_label.LabelCardinalityInconsistency`, and
  :py:class:`~small_text.query_strategies.multi_label.AdaptiveActiveLearning` are tailored
  to multi-label data.
- PyTorch-specific methods:
  :py:class:`~small_text.integrations.pytorch.query_strategies.strategies.ExpectedGradientLength`,
  :py:class:`~small_text.integrations.pytorch.query_strategies.strategies.ExpectedGradientLengthMaxWord`,
  :py:class:`~small_text.integrations.pytorch.query_strategies.strategies.ExpectedGradientLengthLayer`,
  :py:class:`~small_text.integrations.pytorch.query_strategies.strategies.BADGE`, and
  :py:class:`~small_text.integrations.pytorch.query_strategies.strategies.DiscriminativeRepresentationLearning`
  are available through the PyTorch integration.

If you are unsure where to start, a practical progression is:

1. Begin with :py:class:`~small_text.query_strategies.strategies.RandomSampling` as a baseline.
2. Compare it against one or two uncertainty-based strategies.
3. Add an embedding-based strategy when your classifier exposes useful embeddings.
4. Introduce wrappers such as :py:class:`~small_text.query_strategies.subsampling.SEALS` or
   :py:class:`~small_text.query_strategies.class_balancing.ClassBalancer` if runtime or class
   distribution becomes a bottleneck.

Quick Overview
==============

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Family
     - Strategies
     - Typical use
   * - Baselines and uncertainty
     - :py:class:`~small_text.query_strategies.strategies.RandomSampling`,
       :py:class:`~small_text.query_strategies.strategies.LeastConfidence`,
       :py:class:`~small_text.query_strategies.strategies.PredictionEntropy`,
       :py:class:`~small_text.query_strategies.strategies.BreakingTies`,
       :py:class:`~small_text.query_strategies.bayesian.BALD`
     - Strong starting points for benchmarking or for setups where model predictions are the
       main signal.
   * - Embedding-based
     - :py:class:`~small_text.query_strategies.strategies.EmbeddingKMeans`,
       :py:class:`~small_text.query_strategies.coresets.GreedyCoreset`,
       :py:class:`~small_text.query_strategies.coresets.LightweightCoreset`,
       :py:class:`~small_text.query_strategies.strategies.ContrastiveActiveLearning`,
       :py:class:`~small_text.query_strategies.vector_space.ProbCover`
     - Useful when diversity, representation coverage, or pool structure matters and the
       classifier can provide embeddings.
   * - Pool-structure based
     - :py:class:`~small_text.query_strategies.strategies.DiscriminativeActiveLearning`
     - Useful when selection should focus on the separation between the labeled and
       unlabeled pools rather than only uncertainty or embedding coverage.
   * - Wrappers and pool reduction
     - :py:class:`~small_text.query_strategies.class_balancing.ClassBalancer`,
       :py:class:`~small_text.query_strategies.subsampling.SEALS`,
       :py:class:`~small_text.query_strategies.subsampling.AnchorSubsampling`
     - Useful when you want to rebalance selections or limit a more expensive base strategy to a
       smaller candidate subset.
   * - Multi-label
     - :py:class:`~small_text.query_strategies.multi_label.CategoryVectorInconsistencyAndRanking`,
       :py:class:`~small_text.query_strategies.multi_label.LabelCardinalityInconsistency`,
       :py:class:`~small_text.query_strategies.multi_label.AdaptiveActiveLearning`
     - Use these when the task is inherently multi-label and the query signal should reflect
       label cardinality or per-label ranking behavior.
   * - PyTorch integration
     - :py:class:`~small_text.integrations.pytorch.query_strategies.strategies.ExpectedGradientLength`,
       :py:class:`~small_text.integrations.pytorch.query_strategies.strategies.ExpectedGradientLengthMaxWord`,
       :py:class:`~small_text.integrations.pytorch.query_strategies.strategies.ExpectedGradientLengthLayer`,
       :py:class:`~small_text.integrations.pytorch.query_strategies.strategies.BADGE`,
       :py:class:`~small_text.integrations.pytorch.query_strategies.strategies.DiscriminativeRepresentationLearning`
     - Useful for PyTorch-based models when gradient information or embedding-derived gradient
       surrogates are part of the acquisition logic.

Pre-implemented Strategies
==========================

General-Purpose Baselines
-------------------------

These strategies are easy to compare and often serve as the first baselines in an experiment.

- :py:class:`~small_text.query_strategies.strategies.RandomSampling`

Uncertainty-Based Strategies
----------------------------

These strategies rank instances by uncertainty derived from model predictions.

- :py:class:`~small_text.query_strategies.strategies.LeastConfidence`
- :py:class:`~small_text.query_strategies.strategies.PredictionEntropy`
- :py:class:`~small_text.query_strategies.strategies.BreakingTies`
- :py:class:`~small_text.query_strategies.bayesian.BALD`

Embedding-Based Strategies
--------------------------

These strategies rely on vector representations and are useful when coverage, diversity, or
neighborhood structure matters.

- :py:class:`~small_text.query_strategies.strategies.EmbeddingKMeans`
- :py:class:`~small_text.query_strategies.coresets.GreedyCoreset`
- :py:class:`~small_text.query_strategies.coresets.LightweightCoreset`
- :py:class:`~small_text.query_strategies.vector_space.ProbCover`
- :py:class:`~small_text.query_strategies.strategies.ContrastiveActiveLearning`

Pool-Structure-Based Strategies
-------------------------------

These strategies reason about the relation between the labeled and unlabeled pools directly.

- :py:class:`~small_text.query_strategies.strategies.DiscriminativeActiveLearning`

Wrappers and Pool Reduction
---------------------------

These strategies delegate to a base strategy or restrict the set of candidates before querying.

- :py:class:`~small_text.query_strategies.class_balancing.ClassBalancer`
- :py:class:`~small_text.query_strategies.subsampling.SEALS`
- :py:class:`~small_text.query_strategies.subsampling.AnchorSubsampling`

Multi-Label Strategies
----------------------

These strategies are designed specifically for multi-label classification.

- :py:class:`~small_text.query_strategies.multi_label.CategoryVectorInconsistencyAndRanking`
- :py:class:`~small_text.query_strategies.multi_label.LabelCardinalityInconsistency`
- :py:class:`~small_text.query_strategies.multi_label.AdaptiveActiveLearning`

PyTorch Integration
-------------------

These strategies live in the PyTorch integration and extend the general query strategy
interface with gradient-based or embedding-based methods for neural models.

- :py:class:`~small_text.integrations.pytorch.query_strategies.strategies.ExpectedGradientLength`
- :py:class:`~small_text.integrations.pytorch.query_strategies.strategies.ExpectedGradientLengthMaxWord`
- :py:class:`~small_text.integrations.pytorch.query_strategies.strategies.ExpectedGradientLengthLayer`
- :py:class:`~small_text.integrations.pytorch.query_strategies.strategies.BADGE`
- :py:class:`~small_text.integrations.pytorch.query_strategies.strategies.DiscriminativeRepresentationLearning`

Interface
=========

All query strategies revolve around the :code:`query()` method. Implementations may use all of
its positional arguments, but they are free to focus only on the parts they actually need.

.. literalinclude:: ../../small_text/query_strategies/base.py
   :pyobject: QueryStrategy

- `clf` provides the model-side signal, such as predictions, probabilities, embeddings, or
  gradients.
- `dataset` contains the full pool, regardless of whether individual samples are currently
  labeled or unlabeled.
- `indices_unlabeled` and `indices_labeled` describe the active learning state as index
  partitions over `dataset`.
- `y` contains the labels for the labeled pool.
- `n` controls how many instances should be queried in the current step.

.. note:: The union of `indices_unlabeled` and `indices_labeled` does not need to cover all
          samples in `dataset`. This makes it possible to keep instances in the dataset while
          excluding them from the active learning loop.

Helpers
=======

Some query strategies are only meaningful for single-label or multi-label data. The
`constraints()` decorator can be used to enforce that a strategy is only called in supported
settings.

.. note:: Not every strategy in the library uses explicit constraints. The absence of a declared
          constraint should therefore not be interpreted as a guarantee that every possible setup
          is supported.

Constraints
-----------

.. testcode::

    from small_text.query_strategies import constraints, QueryStrategy

    @constraints(classification_type='single-label')
    class MyQueryStrategy(QueryStrategy):
        pass


API Reference
=============

Baselines and Uncertainty
-------------------------

.. py:module:: small_text.query_strategies.strategies

.. autoclass:: RandomSampling

.. autoclass:: LeastConfidence

.. autoclass:: PredictionEntropy

.. autoclass:: BreakingTies

.. py:module:: small_text.query_strategies.bayesian

.. autoclass:: BALD
    :special-members: __init__

Embedding-Based
---------------

.. py:module:: small_text.query_strategies.strategies
    :noindex:

.. autoclass:: EmbeddingKMeans
    :special-members: __init__

.. autoclass:: ContrastiveActiveLearning
    :special-members: __init__

.. py:module:: small_text.query_strategies.coresets

.. autoclass:: GreedyCoreset
    :special-members: __init__

.. autoclass:: LightweightCoreset
    :special-members: __init__

.. py:module:: small_text.query_strategies.vector_space

.. autoclass:: ProbCover
    :special-members: __init__

Pool-Structure-Based
--------------------

.. py:module:: small_text.query_strategies.strategies
    :noindex:

.. autoclass:: DiscriminativeActiveLearning
    :special-members: __init__

Wrappers and Pool Reduction
---------------------------

.. py:module:: small_text.query_strategies.class_balancing

.. autoclass:: ClassBalancer
    :special-members: __init__

.. py:module:: small_text.query_strategies.subsampling

.. autoclass:: SEALS
    :special-members: __init__

.. autoclass:: AnchorSubsampling
    :special-members: __init__

Multi-Label
-----------

.. py:module:: small_text.query_strategies.multi_label

.. autoclass:: CategoryVectorInconsistencyAndRanking
    :special-members: __init__

.. autoclass:: LabelCardinalityInconsistency
    :special-members: __init__

.. autoclass:: AdaptiveActiveLearning
    :special-members: __init__

PyTorch Integration
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

.. autoclass:: DiscriminativeRepresentationLearning
    :special-members: __init__

Functions
=========

.. autofunction:: small_text.query_strategies.coresets.greedy_coreset

.. autofunction:: small_text.query_strategies.coresets.lightweight_coreset
