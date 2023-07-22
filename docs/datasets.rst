========
Datasets
========

Small-Text's basic data structures for data are called :py:class:`Datasets<small_text.data.datasets.Dataset>` and
represent text data for :doc:`single-label and multi-label classification<classifiers>`.
Besides features and labels, these datasets also hold meta information about the underlying data, namely the number of classes and
whether the labeling is single- or multi-label.

.. contents:: Overview
   :depth: 1
   :local:
   :backlinks: none

----

Dataset Overview
================

While all the other components are mostly unified, for the datasets the respective
dataset and classifier have to match, since the underlying representations can be quite different.

.. table::
   :widths: 50 50

   +----------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
   | Dataset Implementation                                                                 | Classifier(s)                                                                                               |
   +========================================================================================+=============================================================================================================+
   | :py:class:`~small_text.data.datasets.SklearnDataset`                                   | :py:class:`~small_text.classifiers.classification.SklearnClassifier`                                        |
   +----------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
   | :py:class:`~small_text.integrations.pytorch.datasets.PytorchTextClassificationDataset` | :py:class:`~small_text.integrations.pytorch.classifiers.kimcnn.KimCNNClassifier`                            |
   +----------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
   | :py:class:`~small_text.data.datasets.TextDataset`                                      | :py:class:`~small_text.integrations.transformers.classifiers.setfit.SetFitClassification`                   |
   +----------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
   | :py:class:`~small_text.integrations.transformers.datasets.TransformersDataset`         | :py:class:`~small_text.integrations.transformers.classifiers.classification.TransformerBasedClassification` |
   +----------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+

SklearnDatasets
~~~~~~~~~~~~~~~

Disregarding any integrations, small-text's core is built around dense (numpy) and sparse (scipy)
matrices, which can be easily used for active learning via :py:class:`~small_text.data.datasets.SklearnDataset`.
This dataset is compatible with :py:class:`~small_text.classifiers.classification.SklearnClassifier` classifiers.

The form of the features and labels can vary as follows:

- The features can either be dense or sparse.
- The labeling can either be single- or multi-label targets.


.. note:: Despite all integration efforts, at the end it comes down to the model in use,
          which combinations of dense/sparse features and single-/multi-label are supported.


Sparse Features
~~~~~~~~~~~~~~~

Traditional text classification methods relied on the Bag-of-Words representation,
which can be efficiently represented as a sparse matrix.

.. testcode::

   import numpy as np
   from scipy.sparse import csr_matrix, random
   from small_text.data import SklearnDataset

   # create exemplary features and labels randomly
   x = random(100, 2000, density=0.15, format='csr')
   y = np.random.randint(0, 2, size=100)

   dataset = SklearnDataset(x, y, target_labels=np.arange(2))

Dense Features
~~~~~~~~~~~~~~

Or similarly with dense features:

.. testcode::

   import numpy as np
   from small_text.data import SklearnDataset

   # create exemplary features and labels randomly
   x = np.random.rand(100, 30)
   y = np.random.randint(0, 2, size=100)

   dataset = SklearnDataset(x, y, target_labels=np.arange(2))

Multi-Label
~~~~~~~~~~~

The previous two examples were single-label datasets, i.e. each instance had exactly
one label assigned. If you want to classify multi-label problems, you need to pass a scipy
csr_matrix. This matrix must be a multi-label indicator matrix, i.e. a matrix in the shape of
(num_documents, num_labels) where each non-zero entry is exactly 1 and represents a label.

.. testcode::

   import numpy as np
   from scipy import sparse
   from small_text.data import SklearnDataset

   x = sparse.random(100, 2000, density=0.15, format='csr')
   # a random sparse matrix
   y = sparse.random(100, 5, density=0.5, format='csr')
   # convert non-zero entries to 1, making it an indicator
   y.data[np.s_[:]] = 1

   dataset = SklearnDataset(x, y, target_labels=np.arange(5))

Indexing and Views
------------------

Accessing an data object by index or range such as :code:`dataset[selector]` is called indexing,
where selector can be an index (:code:`dataset[10]`), a range (:code:`dataset[2:10]`), or an array
of indices (:code:`dataset[[1, 5, 10]]`).
Similarly to `numpy indexing <https://numpy.org/doc/stable/user/basics.indexing.html#basics-indexing>`_,
dataset indexing does not create a copy of the selected subset but creates a view thereon.
:py:class:`~small_text.data.datasets.DatasetView` objects behave similarly to Datasets, but are readonly.

.. testcode::

   import numpy as np
   from small_text.data import SklearnDataset

   # create exemplary features and labels randomly
   x = np.random.rand(100, 30)
   y = np.random.randint(0, 2, size=100)

   dataset = SklearnDataset(x, y, target_labels=np.arange(2))

   # returns a DatasetView of the first ten items in x
   dataset_sub = dataset[0:10]


In the multi-label case, this is for once simpler, and here no separate handling is needed.
An unlabeled instance just has no label in the corresponding row of the indicator matrix.

Copying a Dataset
~~~~~~~~~~~~~~~~~

While indexing creates a view instead of copying, there are cases where you want a copy instead.

.. testcode::

   dataset_copy = dataset.clone()
   print(type(dataset_copy).__name__)

*Output*:

.. testoutput::

   SklearnDataset

This also works on :py:class:`~small_text.data.datasets.DatasetView` instances, however,
the :code:`clone()` operation dissolves a view and returns a dataset again:

.. testcode::

   dataset_view = dataset[0:5]
   dataset_view_copy = dataset_view.clone()
   print(type(dataset_view_copy).__name__)

*Output*:

.. testoutput::

   SklearnDataset

----

Constructing an Unlabeled Dataset
=================================

Unless you are doing a simulated experiment, you will need to deal with (partly or
completely) unlabeled data. We show how to construct an unlabeled dataset at the example of
:py:class:`~small_text.data.datasets.SklearnDataset`, but the concept is the same for
:py:class:`~small_text.integrations.pytorch.datasets.PytorchTextClassificationDataset`
and
:py:class:`~small_text.integrations.transformers.datasets.TransformersDataset`.

For this, it must be distinguished between the single- and multi-label setting. For the single-label case,
there is a special label constant :code:`LABEL_UNLABELED`,
which indicates that an instance is unlabeled:

.. testcode::

   import numpy as np
   from small_text.base import LABEL_UNLABELED
   from small_text.data import SklearnDataset

   x = np.random.rand(100, 30)
   # a label array of size 100 where each entry means "unlabeled"
   y = np.array([LABEL_UNLABELED] * 100)

   dataset = SklearnDataset(x, y, target_labels=np.arange(2))


For the multi-label case, creating unlabeled datasets is trivial. The sparse label matrix works as
usual, and unlabeled instances simply correspond to empty rows:

.. testcode::

    import numpy as np
    from scipy import sparse
    from small_text.data import SklearnDataset

    num_labels = 3

    x = sparse.random(100, 2000, density=0.15, format='csr')
    y = sparse.csr_matrix((100, num_labels))  # <-- this a sparse empty matrix

    dataset = SklearnDataset(x, y, target_labels=np.arange(num_labels))

For partially labeled data, the sparse label matrix `y` has empty and non-empty rows.

----

Integration Data Structures
===========================

Both the :doc:`Pytorch Integration <libraries/pytorch_integration>` the :doc:`Transformers Integration <libraries/transformers_integration>`
bring their own Datasets (each subclassing :py:class:`~small_text.data.datasets.Dataset`),
which rely on different representations and bring additional methods for handling GPU-related operations.
See the respective integration's page for more information.

----

Building your own Dataset implementation
========================================

In general, any data structure handled by your classifier can be implemented.
Custom Datasets should work with existing parts of the library, providing the following
conditions are met:

1. Indexing (using integers, lists, ndarray, slices) must be supported
2. Iteration must be supported
3. The length of dataset (`__len__`) must return the number of data instances

See :py:class:`small_text.integrations.transformers.datasets.TransformersDataset` for an example.
