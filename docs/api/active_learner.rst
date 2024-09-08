=================
ActiveLearner API
=================

Everything in small-text revolves around the active learner.

.. contents:: Overview
   :depth: 2
   :local:
   :backlinks: none

----

Pool-Based Active Learner
=========================

.. py:currentmodule:: small_text.active_learner

.. autoclass:: PoolBasedActiveLearner

  .. automethod:: initialize
  .. automethod:: query
  .. automethod:: update
  .. automethod:: update_label_at
  .. automethod:: remove_label_at
  .. automethod:: save
  .. automethod:: load
