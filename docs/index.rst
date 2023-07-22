==========
Small-Text
==========

`Small-Text` provides :doc:`active learning<active_learning>` for text classification.
It is designed to offer a robust and modular set of components *for both experimental and applied active learning*.

Why Small-Text?
===============

- **Interchangeable components**: All components are based around the `ActiveLearner <https://github.com/webis-de/small-text/blob/main/small_text/active_learner.py>`_ class.
  You can mix and match different many initialization strategies, query strategies, and classifiers.
- Integrations: Optional Integrations allow you to use **GPU-based models** from the pytorch and transformers libraries.
- **Common patterns**: We provide solutions to common challenges when building experiments and/or applications,
  such as :doc:`patterns/pool` and :doc:`patterns/serialization`.
- Multiple scientifically evaluated components are **pre-implemented and ready to use**
  (:doc:`query strategies<components/query_strategies>`, :doc:`initialization strategies<components/initialization>`, and :doc:`stopping criteria<components/stopping_criteria>`).

----

Getting Started
===============

.. toctree::
   :caption: Getting Started
   :maxdepth: 1
   :hidden:

   install
   active_learning
   datasets

- Start: :doc:`install` | :doc:`Active Learning Overview<active_learning>`
- Examples: `Notebooks <https://github.com/webis-de/small-text/tree/v1.3.1/examples/notebooks>`_ | `Code Examples <https://github.com/webis-de/small-text/tree/v1.3.1/examples/examplecode>`_

.. toctree::
   :caption: Components
   :maxdepth: 1
   :hidden:

   components/initialization
   components/query_strategies
   components/stopping_criteria

.. toctree::
   :caption: Classification
   :maxdepth: 1
   :hidden:

   classifiers
   components/training

.. toctree::
   :caption: Integrations
   :maxdepth: 1
   :hidden:

   libraries/pytorch_integration
   libraries/transformers_integration

.. toctree::
   :caption: Common Patterns
   :maxdepth: 1
   :hidden:

   patterns/pool
   patterns/serialization

----

Citation
========

| A preprint which introduces small-text is available here:
| `Small-Text: Active Learning for Text Classification in Python <https://arxiv.org/abs/2107.10314>`_.

.. code-block:: text

    @misc{schroeder2021smalltext,
        title={Small-Text: Active Learning for Text Classification in Python},
        author={Christopher Schröder and Lydia Müller and Andreas Niekler and Martin Potthast},
        year={2021},
        eprint={2107.10314},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }

------

License
========

`MIT License <https://github.com/webis-de/small-text/blob/main/LICENSE>`_

------

.. toctree::
   :caption: API
   :maxdepth: 1
   :hidden:

   api/active_learner
   api/classifier
   api/dataset


.. toctree::
   :caption: Other
   :maxdepth: 0
   :hidden:

   changelog
   showcase
   reproducibility_notes
   bibliography

:ref:`genindex` | :ref:`modindex` | :ref:`search`
