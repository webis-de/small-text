==========
Small-Text
==========

`Small-Text` provides :doc:`active learning<active_learning>` for text classification.
It is designed to offer a robust and modular set of components for both experimental and applied active learning.

Getting Started
===============

.. toctree::
   :caption: Getting Started
   :maxdepth: 1
   :hidden:

   install
   active_learning
   data_structures
   classification

For now, the best way to get started is checking out the folders `examples/notebooks/ <https://github.com/webis-de/small-text/tree/master/examples/notebooks>`_ and `examples/examplecode/ <https://github.com/webis-de/small-text/tree/master/examples/examplecode>`_ in the github directory.

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
You can mix and match different many initialization strategies, query strategies, and Classifiers.


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

------

.. toctree::
   :caption: API
   :maxdepth: 1
   :hidden:

   api/active_learner
   api/classifier
   api/data_set


.. toctree::
   :caption: Other
   :maxdepth: 0
   :hidden:

   changelog
   reproducibility_notes
   bibliography

----

:ref:`genindex` | :ref:`modindex` | :ref:`search`
