.. py:currentmodule:: small_text.stopping_criteria

=================
Stopping Criteria
=================

Stopping criteria provide you an indication when to exit the active learning loop.

Pre-implemented Stopping Criteria
=================================

- :py:class:`~small_text.stopping_criteria.base.DeltaFScore`
- :py:class:`~small_text.stopping_criteria.kappa.KappaAverage`

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
