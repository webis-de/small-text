===============
Data Structures
===============

Small-Text's basic data structures for data are called :py:class:`Datasets<small_text.data.datasets.Dataset>` and
represent text data for :doc:`single-label and multi-label classification<classification>`.
Besides features and labels, these datasets also hold meta information about the underlying data, namely the number of classes and
whether the labeling is single- or multi-label.

Basic Data Structures
=====================

Disregarding any integrations, small-text's core is built around dense (numpy) and sparse (scipy)
matrices, which can be easily used for active learning via :py:class:`~small_text.data.datasets.SklearnDataset`.
This dataset is compatible with :py:class:`~small_text.classifiers.classification.SklearnClassifier` classifiers.

The form of the features and labels can vary as follows:

- The features can either be dense or sparse.
- The labeling can either be single- or multi-label targets.


.. note:: Despite all integration efforts, at the end it comes down to the model in use,
          which combinations of dense/sparse features and single-/multi-label are supported.


Sparse Features
---------------

Traditional text classification methods relied on the Bag-of-Words representation,
which can be efficiently represented as a sparse matrix.

.. testcode::

   import numpy as np
   from scipy.sparse import csr_matrix, random
   from small_text.data import SklearnDataset

   # create exemplary features and labels randomly
   x = random(100, 2000, density=0.15, format='csr')
   y = np.random.randint(0, 1, size=100)

   dataset = SklearnDataset(x, y)

Dense Features
--------------

Or similarly with dense features:

.. testcode::

   import numpy as np
   from small_text.data import SklearnDataset

   # create exemplary features and labels randomly
   x = np.random.rand(100, 30)
   y = np.random.randint(0, 1, size=100)

   dataset = SklearnDataset(x, y)

Multi-Label
-----------

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

   dataset = SklearnDataset(x, y)


Unlabeled Data
--------------

Sometimes you cannot or will not assign a label an instance. To indicate this special status in the single-label scenario
there is a special label constant :code:`LABEL_UNLABELED`, which indicates that an instance is unlabeled:

.. testcode::

   import numpy as np
   from small_text.base import LABEL_UNLABELED
   from small_text.data import SklearnDataset

   x = np.random.rand(100, 30)
   # a label array of size 100 where each entry means "unlabeled"
   y = np.array([LABEL_UNLABELED] * 100)

   dataset = SklearnDataset(x, y)


Indexing and Views
==================

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
   y = np.random.randint(0, 1, size=100)

   dataset = SklearnDataset(x, y)

   # returns a DatasetView of the first ten items in x
   dataset_sub = dataset[0:10]


In the multi-label case, this is for once simpler, and here no separate handling is needed.
An unlabeled instance just has no label in the corresponding row of the indicator matrix.

Copying a Dataset
=================

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



Integration Data Structures
===========================

Both the :doc:`Pytorch Integration <libraries/pytorch_main>` the :doc:`Transformers Integration <libraries/transformers_main>`
bring their own Datasets (each subclassing :py:class:`~small_text.data.datasets.Dataset`),
which rely on different representations and bring additional methods for handling GPU-related operations.


Further Extensions
==================

In general, any data structure handled by your classifier can be implemented.
Custom Datasets should work with existing parts of the library, providing the following
conditions are met:

1. Indexing (using integers, lists, ndarray, slices) must be supported
2. Iteration must be supported
3. The length of dataset (`__len__`) must return the number of data instances

See :py:class:`small_text.integrations.transformers.datasets.TransformersDataset` for an example.
