===============
Data Structures
===============

Most basic components are tailored towards
:py:class:`active_learning.data.SklearnDataset` (which supports :py:class:`np.array` or :py:class:`scipy.sparse.csr_matrix`).
In the end, it is each component's own responsibility which data structures they support.

Pytorch Integration and Transformers Integration
================================================

Both the :doc:`Pytorch Integration <libraries/pytorch_main>` the :doc:`Transformers Integration <libraries/transformers_main>`
bring their own Datasets (each subclassing :py:class:`active_learning.data.Dataset`).
:py:class:`torchtext.datasets.PytorchTextClassificationDataset` and :py:class:`active_learning.integrations.transformers.TransformersDataset`
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

See :py:class:`active_learning.integrations.transformers.TransformersDataset` for an example.
