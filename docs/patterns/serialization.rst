=============
Serialization
=============

Functionality to save / load your active learner.

----

In some active learning applications, an active learner might have a longer lifespan. For example, you might want
to save your active learner to resume your annotation process at a later time.
This process in which the current object is saved to disk is called serialization.

To allow saving / loading, :py:class:`~small_text.active_learner.PoolBasedActiveLearner` provides
:py:meth:`~small_text.active_learner.PoolBasedActiveLearner.save` and
:py:meth:`~small_text.active_learner.PoolBasedActiveLearner.load`. The only mandataroy argument is a folder
to which the active learner is saved to or loaded from respectively.

----

Usage
=====

(De-)serialization is straightforward to use. Models on the GPU need to be transferred to the CPU first
to avoid errors during deserialization.

.. note::
    - Serialization has changed in v2.0.0 and is not backwards compatible with files saved using small-text v1.x.
    - You likely need the same small-text version (and depending on the model also similar dependencies).
      This might be improved in future releases.


See
===

* :py:meth:`~small_text.active_learner.PoolBasedActiveLearner.save`
* :py:meth:`~small_text.active_learner.PoolBasedActiveLearner.load`

----

Example
=======

.. code-block:: python

    from small_text import PoolBasedActiveLearner

    active_learner = <...>   # care, this does not run;
                             # active_learner is assumed to be a trained active learner

    # Only for models on the GPU: transfer them to the CPU before serialization
    active_learner.classifier.model = active_learner.classifier.model.cpu()

    project_folder = '/tmp/active-learning-project'
    active_learner.save(project_folder)

    active_learner = PoolBasedActiveLearner.load(project_folder)
