==========
small-text
==========

[|LIBRARY_VERSION|]

`small-text` provides :doc:`active_learning` for Text Classification.
It is designed to offer a robust and modular set of components for both experimental and applied Active Learning.

Getting Started
===============

.. toctree::
   :caption: Getting Started
   :maxdepth: 1
   :hidden:

   install
   active_learning
   data_structures

For now, the best way to get started is the `examples/ <https://github.com/webis-de/small-text/tree/master/examples>`_ folder in the github directory.

Active Learning Components
==========================

.. toctree::
   :caption: Components
   :maxdepth: 1
   :hidden:

   components/initialization
   components/query_strategies
   components/stopping_criteria

All components are based around the `ActiveLearner <https://github.com/webis-de/small-text/blob/master/small_text/active_learner.py>`_ class.
You can mix and match different many Initialization Strategies, Query Strategies and Classifiers.


Integrations
============

Optional Integrations allow you to use models from other libraries such as pytorch or transformers.

.. toctree::
   :caption: Integrations
   :maxdepth: 1
   :hidden:

   libraries/pytorch_main
   libraries/pytorch_classes
   libraries/transformers_main
   libraries/transformers_classes


Common Patterns
===============

We provide patterns to common challenges when building experiments and/or applications,
such as :doc:`patterns/pool` and :doc:`patterns/serialization`.

.. toctree::
   :caption: Common Patterns
   :maxdepth: 1
   :hidden:

   patterns/pool
   patterns/serialization


API
===

.. toctree::
   :caption: Common Patterns
   :maxdepth: 1
   :hidden:

   api/active_learner
   api/data_set

:doc:`api/active_learner`
:doc:`api/data_set`


Citation
========

A preprint which introduces small-text is available here:
`Small-text: Active Learning for Text Classification in Python <https://arxiv.org/abs/2107.10314>`_.

.. code-block:: text

    @misc{schroeder2021smalltext,
        title={Small-text: Active Learning for Text Classification in Python},
        author={Christopher Schröder and Lydia Müller and Andreas Niekler and Martin Potthast},
        year={2021},
        eprint={2107.10314},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }

----
:ref:`genindex` | :ref:`modindex` | :ref:`search`
