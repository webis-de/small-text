========================
Transformers Integration
========================

The Transformers Integration makes :ref:`transformer-based classification <libraries/transformers_integration:Transformer-based Classification>` and
:ref:`sentence transformer finetuning <libraries/transformers_integration:Sentence Transformer Finetuning>` usable in small-text.
It relies on the :doc:`Pytorch Integration<pytorch_integration>` which is a prerequisite.

.. note:: Some implementation make use of :ref:`optional dependencies <install:Optional Dependencies>`.

----

.. contents:: Overview
   :depth: 1
   :local:
   :backlinks: none

----

Installation
=============

Before you can use the transformers integration :ref:`make sure the required dependencies have been installed <installation-transformers>`.

----

Contents
========

With the integration you will have access to the following additional components:

+------------------+------------------------------------------------------------------------------------------+
| Components       | Resources                                                                                |
+==================+==========================================================================================+
| Datasets         | :ref:`TransformersDataset <api-transformers-dataset>`                                    |
+------------------+------------------------------------------------------------------------------------------+
| Classifiers      | :ref:`TransformerBasedClassification <api-classifiers-transformer-based-classification>` |
+------------------+------------------------------------------------------------------------------------------+
| Query Strategies | (See :doc:`Query Strategies </components/query_strategies>`)                             |
+------------------+------------------------------------------------------------------------------------------+

-----

Examples
========

Transformer-based Classification
--------------------------------

An example is provided in :file:`examples/examplecode/transformers_multiclass_classification.py`:

.. literalinclude:: ../../examples/examplecode/transformers_multiclass_classification.py
   :language: python


Sentence Transformer Finetuning
-------------------------------

An example is provided in :file:`examples/examplecode/setfit_multiclass_classification.py`:

.. literalinclude:: ../../examples/examplecode/setfit_multiclass_classification.py
   :language: python

