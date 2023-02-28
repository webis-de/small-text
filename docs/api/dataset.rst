===========
Dataset API
===========

All datset implementations inherit from the the abstract class :py:class:`~small_text.data.datasets.Dataset`.
Several such implementations are available, depending on the choice of classifier (and on the :ref:`installed optional dependencies <install:Optional Dependencies>`).

.. contents:: Overview
   :depth: 1
   :local:
   :backlinks: none

----

Core
====

.. currentmodule:: small_text.data.datasets

.. autoclass:: Dataset

  .. autoproperty:: x
  .. autoproperty:: y
  .. autoproperty:: target_labels

.. autoclass:: small_text.data.datasets.SklearnDataset
  :members:
  :special-members: __init__

.. autoclass:: small_text.data.datasets.TextDataset
  :members:
  :special-members: __init__

----

Pytorch Integration
===================

.. currentmodule:: small_text.integrations.pytorch.datasets

.. _api-datasets-pytorch-text-classification-dataset:

.. autoclass:: PytorchTextClassificationDataset
    :members: x, y, target_labels, data, vocab, clone, to, from_arrays
    :member-order: bysource
    :special-members: __init__

----

Transformers Integration
========================


.. currentmodule:: small_text.integrations.transformers.datasets

.. _api-transformers-dataset:

.. autoclass:: TransformersDataset
    :members: x, y, target_labels, data, clone, to, from_arrays
    :member-order: bysource
    :special-members: __init__
