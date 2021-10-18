===============
Data Structures
===============

In order to make the integrated libraries and all extensions accessible in the same way,
classifiers (and more specialized query strategies as well) rely on dataset abstractions based on
the interface :py:class:`~small_text.data.datasets.Dataset`.

Basic Data Structures
=====================

Dense (numpy) and sparse (scipy) matrices can be easily used within datasets in combination with :py:class:`~small_text.data.datasets.SklearnDataset`,
which is compatible with all :py:class:`~small_text.classifiers.classification.SklearnClassifier` classifiers.

Sparse Vectors
^^^^^^^^^^^^^^

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

Or similarly with dense features:

.. testcode::

   import numpy as np
   from small_text.data import SklearnDataset

   # create exemplary features and labels randomly
   x = np.random.rand(100, 30)
   y = np.random.randint(0, 1, size=100)

   dataset = SklearnDataset(x, y)


Integration Data Structures
---------------------------

Both the :doc:`Pytorch Integration <libraries/pytorch_main>` the :doc:`Transformers Integration <libraries/transformers_main>`
bring their own Datasets (each subclassing :py:class:`~small_text.data.datasets.Dataset`),
which rely on different representations and bring additional methods for handling GPU-related operations.


Indexing and Views
==================

.. testcode::

   import numpy as np
   from small_text.data import SklearnDataset

   # create exemplary features and labels randomly
   x = np.random.rand(100, 30)
   y = np.random.randint(0, 1, size=100)

   dataset = SklearnDataset(x, y)

   # returns a DatasetView of the first ten items in x
   dataset_sub = dataset[0:10]


Similarly to numpy, indexing does not create a copy of the selected subset but creates a view thereon.
:py:class:`~small_text.data.datasets.DatasetView` objects behave similarly to Datasets, but are readonly.

Further Extensions
==================

In general, any data structure handled by your classifier can be implemented.
Custom Datasets should work with existing parts of the library, providing the following
conditions are met:

1. Indexing (using integers, lists, ndarray, slices) must be supported
2. Iteration must be supported
3. The length of dataset (`__len__`) must return the number of data instances

See :py:class:`small_text.integrations.transformers.datasets.TransformersDataset` for an example.
