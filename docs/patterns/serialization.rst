=============
Serialization
=============

In real-world applications, an active learner will have a longer lifespan, and therefore it can be useful
to save the current object on disk (or load it from disk). This process is called serialization (deserialization).
While you could always rely on python's own pickle library to achieve this,
the :py:class:`~small_text.active_learner.PoolBasedActiveLearner` class provides convenience methods for both serialization and deserialization.

See
===

* :py:meth:`~small_text.active_learner.PoolBasedActiveLearner.save`
* :py:meth:`~small_text.active_learner.PoolBasedActiveLearner.load`

Note
====
Both and load :py:meth:`~small_text.active_learner.PoolBasedActiveLearner.save` take
a string, a file, or a Path as input argument.

Example
=======

.. code-block:: python

    from small_text.active_learner import PoolBasedActiveLearner

    active_learner = <...>   # care, this does not run;
                             # active_learner is assumed to be a trained active learner

    active_learner.save('active_leaner.pkl')

    active_learner = PoolBasedActiveLearner.load('active_leaner.pkl')
