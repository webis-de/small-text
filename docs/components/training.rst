========
Training
========

Apart from active learning, small-text includes several helpers for classification.

.. contents:: Overview
   :depth: 1
   :local:
   :backlinks: none

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

   from small_text.training.early_stopping import EarlyStopping, SequentialEarlyStopping

   early_stopping = EarlyStopping('val_loss', patience=2)

   print(early_stopping.check_early_stop(1, {'val_loss': 0.060}))
   print(early_stopping.check_early_stop(2, {'val_loss': 0.061}))  # no improvement, don't stop
   print(early_stopping.check_early_stop(3, {'val_loss': 0.060}))  # no improvement, stop

..  testoutput::

    False
    False
    True


2. Monitoring training accuracy (higher is better):

.. testcode::

   from small_text.training.early_stopping import EarlyStopping, SequentialEarlyStopping

   early_stopping = EarlyStopping('val_acc', patience=2)

   print(early_stopping.check_early_stop(1, {'val_acc': 0.80}))
   print(early_stopping.check_early_stop(3, {'val_acc': 0.79}))  # no improvement, don't stop
   print(early_stopping.check_early_stop(2, {'val_acc': 0.81}))  # improvment
   print(early_stopping.check_early_stop(3, {'val_acc': 0.81}))  # no improvement, don't stop
   print(early_stopping.check_early_stop(3, {'val_acc': 0.80}))  # no improvement, don't stop

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
This can be easily done by using :py:class:`SequentialEarlyStopping`
which sequentially applies a list of early stopping handlers.

.. testcode::

   from small_text.training.early_stopping import EarlyStopping, SequentialEarlyStopping

   early_stopping = SequentialEarlyStopping([
       EarlyStopping('val_loss', patience=3),
       EarlyStopping('train_acc', threshold=0.99)
   ])

:py:class:`SequentialEarlyStopping`  returns `True`, i.e. triggers an early stop, iff at least one of the early stopping handlers
within the given list returns `True`.

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


.. autoclass:: SequentialEarlyStopping
    :special-members: __init__
    :members:

.. autoclass:: NoopEarlyStopping
    :members:
