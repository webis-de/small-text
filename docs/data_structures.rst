===============
Data Structures
===============

In order to make the integrated libraries and all extensions accessible in the same way,
classifiers (and more specialized query strategies as well) rely on dataset abstractions based on
the interface :py:class:`small_text.data.Dataset`.


Scikit-learn
============

:py:class:`small_text.data.SklearnDataset` is a dataset abstraction for both sparse and dense features.

Sparse Vectors
--------------

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

Dense Vectors
-------------

:py:class:`~small_text.data.SklearnDataset` also supports dense features:

.. testcode::

   import numpy as np
   from small_text.data import SklearnDataset

   # create exemplary features and labels randomly
   x = np.random.rand(100, 30)
   y = np.random.randint(0, 1, size=100)

   dataset = SklearnDataset(x, y)


Pytorch Integration and Transformers Integration
================================================

Both the :doc:`Pytorch Integration <libraries/pytorch_main>` the :doc:`Transformers Integration <libraries/transformers_main>`
bring their own Datasets (each subclassing :py:class:`small_text.data.Dataset`).
:py:class:`torchtext.datasets.PytorchTextClassificationDataset` and :py:class:`small_text.integrations.transformers.TransformersDataset`
adapt the data structures to work the respective classifiers, and offer some convenience methods
to transfer features from/to the GPU.

Further Extensions
==================

In general, any data structure handled by your classifier can be implemented.
Custom Datasets should work with existing parts of the library, providing the following
conditions are met:

1. Indexing (using integers, lists, ndarray, slices) must be supported
2. Iteration must be supported
3. The length of dataset (`__len__`) must return the number of data instances

See :py:class:`small_text.integrations.transformers.TransformersDataset` for an example.
