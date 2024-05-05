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

----

Compatible Models
=================

While this integration is tailored to the `transformers library <https://github.com/huggingface/transformers>`_,
but since models (and their corresponding) tokenizers can vary considerably, not all models are applicable for small-text classifiers.
To help you with finding a suitable model, we list a subset of compatible models in the following which you can use as a starting point:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Size
     - Models
   * - < 1B Parameters
     - BERT, T5, DistilRoBERTa, DistilBERT, ELECTRA, BioGPT

**English Models**

- BERT models: `bert-base-uncased <https://huggingface.co/google-bert/bert-base-uncased>`_, `bert-large-uncased <https://huggingface.co/google-bert/bert-large-uncased>`_,
  `bert-base-uncased <https://huggingface.co/google-bert/bert-base-uncased>`_
- T5: `t5-small <google-t5/t5-small>`_, `t5-base <google-t5/t5-base>`_, `t5-large <google-t5/t5-large>`_
- DistilRoBERTa: `distilroberta-base <https://huggingface.co/distilbert/distilroberta-base>`_
- DistilBERT: `distilbert-base-uncased <https://huggingface.co/distilbert/distilbert-base-uncased>`_,
  `distilroberta-base-cased <https://huggingface.co/distilbert/distilroberta-base-cased>`_
- ELECTRA: `google/electra-base-discriminator <google/electra-base-discriminator>`_, `google/electra-small-discriminator <google/electra-small-discriminator>`_
- BioGPT: `microsoft/biogpt <microsoft/biogpt>`_

This list is not exhaustive. Let us know when you have tested other models that might belong on these lists.

----

TransformerBasedClassification: Extended Functionality
======================================================

Layer-specific Fine-tuning
--------------------------

Layer-specific fine-tuning can be enabled by setting :py:class:`~small_text.integrations.transformers.classifiers.classification.FineTuningArguments` during the construction of :py:class:`~small_text.integrations.transformers.classifiers.classification.TransformerBasedClassification`. With this, you can enable layerwise gradient decay and gradual unfreezing:

- Layerwise gradient decay: learning rates decrease the lower the layer's level is.
- Gradual unfreezing: lower layers are frozen at the start of the training and become gradually unfrozen with each epoch.

See [HR18]_ for more details on these methods.

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

