========================
Transformers Integration
========================

The Transformers Integration makes :ref:`transformer-based classification <libraries/transformers_integration:Transformer-based Classification>` and
:ref:`sentence transformer finetuning <libraries/transformers_integration:Sentence Transformer Finetuning>` usable in small-text.
In contrast to classical machine learning models, transformer models provide superior performance in most scenarios.
However, this comes at the cost of increased computational requirements.
These models are generally trained on GPUs, making the :doc:`Pytorch Integration<pytorch_integration>` a prerequisite for this integration.

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

Transformers
============

This integration supports both encoder and decoder transformer model architectures.
To learn more about the difference between these two transformer architectures we refer to `blog post by Sebastian Raschka on encoder and decoder models <https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder>`_.
In general, encoder models are better suited for classification, the key task of this library.
Nevertheless, decoder models are equally applicable as well, and sometimes the decoder model might even be better
or you might not have a choice e.g., when requiring a model in a specific language.

Compatible Model Weights
------------------------

The transformers integration is tailored to the `transformers library <https://github.com/huggingface/transformers>`_.
In theory, all architectures should be usable from small-text, but practically limitations may arise due to deviations
in the implementation of models and tokenizers.

To help you with finding a suitable model, we provide a (non-exhaustive) curated list of compatible models below:

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
- T5: `t5-small <https://huggingface.co/google-t5/t5-small>`_, `t5-base <https://huggingface.co/google-t5/t5-base>`_, `t5-large <https://huggingface.co/google-t5/t5-large>`_
- DistilRoBERTa: `distilroberta-base <https://huggingface.co/distilbert/distilroberta-base>`_
- DistilBERT: `distilbert-base-uncased <https://huggingface.co/distilbert/distilbert-base-uncased>`_,
  `distilroberta-base <https://huggingface.co/distilbert/distilroberta-base>`_
- ELECTRA: `google/electra-base-discriminator <https://huggingface.co/google/electra-base-discriminator>`_, `google/electra-small-discriminator <https://huggingface.co/google/electra-small-discriminator>`_
- BioGPT: `microsoft/biogpt <https://huggingface.co/microsoft/biogpt>`_

Let us know when you have successfully tested other models that should be listed here.

----

Sentence Transformers
=====================

Compatible Model Weights
------------------------

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
