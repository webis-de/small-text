=============
ActiveLearner
=============

.. py:currentmodule:: active_learning.active_learner

.. autoclass:: PoolBasedActiveLearner

  .. automethod:: initialize_data
  .. automethod:: query
  .. automethod:: update
  .. automethod:: update_label_at
  .. automethod:: remove_label_at
  .. automethod:: save
  .. automethod:: load


=======
DataSet
=======

.. py:currentmodule:: active_learning.data

.. autoclass:: SklearnDataSet

  .. autoattribute:: x
  .. autoattribute:: y
  .. autoattribute:: target_labels
