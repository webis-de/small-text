.. _installation:

============
Installation
============

Active Learning Library is not public yet. You can install the library directly from git using:

.. code-block:: console

    pip install small-text

In the future there will be a pypi package available.


.. _installation-optional-dependencies:

Optional Dependencies
=====================

This library is designed to be usable in combination with as many classifiers/classification libraries as possible.
Whenever possible, we try to keep most dependencies optional to avoid dependency bloat.
Dependending on the classifier of your choice, you might need additional python libraries.

.. note:: The `Pytorch` and `Transformers` Integrations are best used with a CUDA-capable GPU.
          You need a CUDA version |CUDA_VERSION| or newer, and your GPU must also support that specific version.


.. _installation-pytorch:

Pytorch Integration
-------------------

To enable the Pytorch Integration, install the library with the `pytorch` extra:

.. code-block:: console

    pip install small-text[pytorch]


.. _installation-transformers:

Transformers Integration
------------------------

To enable the Transformers Integration, install the library with the `transformers` extra:

.. code-block:: console

    pip install small-text[transformers]


The Transformers Integration also requires Pytorch, so installing this automatically
entails an installation of the Pytorch Integration.
