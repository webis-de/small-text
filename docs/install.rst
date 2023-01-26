.. _installation:

============
Installation
============

You can easily install small-text using pip or conda:

**Using Pip**

.. code-block:: console

    pip install small-text

**Using Conda**

.. code-block:: console

    conda install small-text

This installs a minimal setup **without** any integrations. By installing the :ref:`integrations<install:Optional Integrations>`
you can enable larger scopes of gpu-based functionality.
Further :ref:`optional dependencies<install:Optional Dependencies>`, i.e. dependencies which are only needed for one to a few strategies
and are not installed by default (to avoid bloated dependencies), might be required.

.. _installation-optional-dependencies:

----

Optional Integrations
=====================

The small-text library is designed to be usable in combination with as many classifiers/classification libraries as possible.
Whenever possible, we try to keep most dependencies optional to avoid dependency bloat.
Dependending on the classifier of your choice, you might need additional python libraries.

.. note:: The `Pytorch` and `Transformers` Integrations are best used with a CUDA-capable GPU.
          You need CUDA version |CUDA_VERSION| or newer, and your GPU must also support that specific version.


.. _installation-pytorch:

Pytorch Integration
-------------------

**Using Pip**

To enable the Pytorch Integration, install the library with the `pytorch` extra:

.. code-block:: console

    pip install small-text[pytorch]

**Using Conda**

.. code-block:: console

    conda install small-text "torch>=1.6.0" "torchtext>=0.7.0"

.. note:: After installation, make sure the installed `torchtext and Pytorch versions are compatible <https://github.com/pytorch/text#user-content-installation>`_.

.. _installation-transformers:

Transformers Integration
------------------------

**Using Pip**

To enable the Transformers Integration, install the library with the `transformers` extra:

.. code-block:: console

    pip install small-text[transformers]

**Using Conda**

.. code-block:: console

    conda install small-text "torch>=1.6.0" "torchtext>=0.7.0" "transformers>=4.0.0"

The Transformers Integration also requires Pytorch, so installing this automatically
entails an installation of the Pytorch Integration.

----

Optional Dependencies
=====================

We keep certain python dependencies optional when they are either only required
for very specific (query or stopping) strategies or are purely convenience functions.

An overview of such dependencies is given in table below:

+-------------------------+------------------------------------------------------------------------------------------------+
| Dependency              | Required by                                                                                    |
+=========================+================================================================================================+
| `hnswlib`_              | :py:class:`~small_text.query_strategies.strategies.SEALS`                                      |
+-------------------------+------------------------------------------------------------------------------------------------+
| `setfit`_               | :py:class:`~small_text.integrations.transformers.classifiers.setfit.SetFitClassification`      |
+-------------------------+------------------------------------------------------------------------------------------------+

.. _hnswlib: https://pypi.org/project/hnswlib/

.. _setfit: https://github.com/huggingface/setfit
