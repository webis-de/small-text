========
Training
========

Apart from active learning, small-text includes several helpers for classification.

.. contents:: Overview
   :depth: 1
   :local:
   :backlinks: none

----

Early Stopping
==============

Early stopping is a mechanism which tries to avoid overfitting when training a model.
For this purpose, an early stopping mechanism monitors certain metrics during the training process
---usually after each epoch---in order to check if early stopping should be triggered.
If the early stopping handler deems an early stop to be necessary according to the given
contraints then it returns `True` when :code:`check_early_stop()` is called.
This response has to be subsequently handled in the respective classifier.

Interface
---------
.. literalinclude:: ../../small_text/training/early_stopping.py
   :pyobject: EarlyStoppingHandler

.. py:currentmodule:: small_text.training.early_stopping


Example Usage
-------------

1. Monitoring validation loss (lower is better):

.. testcode::

   from small_text.training.early_stopping import EarlyStopping
   from small_text.training.metrics import Metric

   early_stopping = EarlyStopping(Metric('val_loss'), patience=2)

   print(early_stopping.check_early_stop(1, {'val_loss': 0.060}))
   print(early_stopping.check_early_stop(2, {'val_loss': 0.061}))  # no improvement, don't stop
   print(early_stopping.check_early_stop(3, {'val_loss': 0.060}))  # no improvement, don't stop
   print(early_stopping.check_early_stop(3, {'val_loss': 0.060}))  # no improvement, stop

*Output:*

..  testoutput::

    False
    False
    False
    True


2. Monitoring training accuracy (higher is better) with `patience=1`:

.. testcode::

   from small_text.training.early_stopping import EarlyStopping
   from small_text.training.metrics import Metric

   early_stopping = EarlyStopping(Metric('val_acc', lower_is_better=False), patience=1)

   print(early_stopping.check_early_stop(1, {'val_acc': 0.80}))
   print(early_stopping.check_early_stop(3, {'val_acc': 0.79}))  # no improvement, don't stop
   print(early_stopping.check_early_stop(2, {'val_acc': 0.81}))  # improvement
   print(early_stopping.check_early_stop(3, {'val_acc': 0.81}))  # no improvement, don't stop
   print(early_stopping.check_early_stop(3, {'val_acc': 0.80}))  # no improvement, stop

*Output:*

..  testoutput::

    False
    False
    False
    False
    True

Combining Early Stopping Conditions
-----------------------------------

What if we want to early stop based on either one of two conditions? For example,
if validation loss does not change during the last 3 checks or training accuracy crosses `0.99`?
This can be easily done by using :py:class:`EarlyStoppingOrCondition`
which sequentially applies a list of early stopping handlers.

.. testcode::

   from small_text.training.early_stopping import EarlyStopping, EarlyStoppingOrCondition
   from small_text.training.metrics import Metric

   early_stopping = EarlyStoppingOrCondition([
       EarlyStopping(Metric('val_loss'), patience=3),
       EarlyStopping(Metric('train_acc', lower_is_better=False), threshold=0.99)
   ])

:py:class:`EarlyStoppingOrCondition`  returns `True`, i.e. triggers an early stop, iff at least one of the early stopping handlers
within the given list returns `True`. Similarly, we have :py:class:`EarlyStoppingAndCondition`
which stops only when all of the early stopping handlers return `True`.

Implementations
---------------

.. autoclass:: EarlyStopping
    :special-members: __init__
    :members:

.. note::
   Currently, supported metrics are validation accuracy (:code:`val_acc`),
   validation loss (:code:`val_loss`), training accuracy (:code:`train_acc`),
   and training loss (:code:`train_loss`). For the accuracy metric, a higher value is better,
   i.e. :code:`patience` triggers only when the respective metric has not exceeded the previous
   best value, and for loss metrics when the respective metric has not fallen below the previous
   best value respectively.

.. autoclass:: EarlyStoppingAndCondition
    :special-members: __init__
    :members:

.. autoclass:: EarlyStoppingOrCondition
    :special-members: __init__
    :members:

.. autoclass:: NoopEarlyStopping
    :members:

----

Model Selection
===============

Given a set of models that have been trained on the same data, model selection chooses
the model that is considered best according to some criterion.
In the context of neural networks, a typical use case for this is the training process,
where the set of models is given by the respetive model after each epoch,
or hyperparameter search, where one model for each hyperparameter configuration is trained.

Interface
---------
.. literalinclude:: ../../small_text/training/model_selection.py
   :pyobject: ModelSelectionManager

Example Usage
-------------

.. testcode::

   from small_text.training.model_selection import ModelSelection

   model_selection = ModelSelection()

   measured_values = {'val_acc': 0.87, 'train_acc': 0.89, 'val_loss': 0.123}
   model_selection.add_model('model_id_1', 'model_1.bin', measured_values)
   measured_values = {'val_acc': 0.88, 'train_acc': 0.91, 'val_loss': 0.091}
   model_selection.add_model('model_id_2', 'model_2.bin', measured_values)
   measured_values = {'val_acc': 0.87, 'train_acc': 0.92, 'val_loss': 0.101}
   model_selection.add_model('model_id_3', 'model_3.bin', measured_values)

   print(model_selection.select(select_by='val_acc'))
   print(model_selection.select(select_by='train_acc'))
   print(model_selection.select(select_by=['val_acc', 'train_acc']))


*Output:*

..  testoutput::

   ModelSelectionResult('model_id_2', 'model_2.bin', {'val_loss': 0.091, 'val_acc': 0.88, 'train_loss': nan, 'train_acc': 0.91}, {'early_stop': False})
   ModelSelectionResult('model_id_3', 'model_3.bin', {'val_loss': 0.101, 'val_acc': 0.87, 'train_loss': nan, 'train_acc': 0.92}, {'early_stop': False})
   ModelSelectionResult('model_id_2', 'model_2.bin', {'val_loss': 0.091, 'val_acc': 0.88, 'train_loss': nan, 'train_acc': 0.91}, {'early_stop': False})


Implementations
---------------

.. py:currentmodule:: small_text.training.model_selection

.. autoclass:: ModelSelection
    :special-members: __init__
    :members:

.. autoclass:: NoopModelSelection
    :members:
