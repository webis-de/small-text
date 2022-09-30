.. py:currentmodule:: small_text.stopping_criteria

=================
Stopping Criteria
=================

Stopping criteria indicate when to exit the active learning loop.

----

Pre-implemented Stopping Criteria
=================================

- :py:class:`~small_text.stopping_criteria.base.DeltaFScore`
- :py:class:`~small_text.stopping_criteria.change.ClassificationChange`
- :py:class:`~small_text.stopping_criteria.kappa.KappaAverage`
- :py:class:`~small_text.stopping_criteria.uncertainty.OverallUncertainty`
- :py:class:`~small_text.stopping_criteria.utility.MaxIterations`

----

Interface
=========

This interface is one of the trickiest, since you might stop on any information available within
the active learning process
(excluding experiment only information like the test set of course).
Therefore all arguments here are optional and :code:`None` by default, and the interface only provides a
very loose frame on how stopping criteria should be built.

.. literalinclude:: ../../small_text/stopping_criteria/base.py
   :pyobject: StoppingCriterion

For an example, see the :py:class:`~small_text.stopping_criteria.kappa.KappaAverage`,
which stops when the change in the `predictions` over multiple iterations falls below a fixed threshold.

Classes
=======

.. py:module:: small_text.stopping_criteria.kappa

.. autoclass:: KappaAverage
    :special-members: __init__
    :inherited-members:

.. py:module:: small_text.stopping_criteria.base

.. autoclass:: DeltaFScore
    :special-members: __init__
    :inherited-members:

.. py:module:: small_text.stopping_criteria.change

.. autoclass:: ClassificationChange
    :special-members: __init__
    :inherited-members:

.. py:module:: small_text.stopping_criteria.uncertainty

.. autoclass:: OverallUncertainty
    :special-members: __init__
    :inherited-members:


.. py:module:: small_text.stopping_criteria.utility

.. autoclass:: MaxIterations
    :special-members: __init__
    :inherited-members:
