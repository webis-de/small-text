===================
Pytorch Integration
===================

The Pytorch integration provides access to Pytorch classifiers and functionality surrounding them.

----

.. contents:: Overview
   :depth: 1
   :local:
   :backlinks: none

----

Installation
=============

Before you can use the pytorch integration :ref:`make sure you have the optional dependencies installed <installation-pytorch>`.

----

Contents
========

With the integration you will have access to the following additional components:

+------------------+--------------------------------------------------------------------------------------------+
| Components       | Resources                                                                                  |
+==================+============================================================================================+
| Datasets         | :ref:`KimCNNClassifier <api-classifiers-kimcnn-classifier>`                                |
+------------------+--------------------------------------------------------------------------------------------+
| Classifiers      | :ref:`PytorchTextClassificationDataset <api-datasets-pytorch-text-classification-dataset>` |
+------------------+--------------------------------------------------------------------------------------------+
| Query Strategies | (See :doc:`Query Strategies </components/query_strategies>`)                               |
+------------------+--------------------------------------------------------------------------------------------+

----

Code Example
============

A code example showing a pytorch multiclass classification is given in :file:`examples/examplecode/pytorch_multiclass_classification.py`.

.. literalinclude:: ../../examples/examplecode/pytorch_multiclass_classification.py
   :language: python
