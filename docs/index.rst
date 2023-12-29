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
- Examples: `Notebooks <https://github.com/webis-de/small-text/tree/v1.3.3/examples/notebooks>`_ | `Code Examples <https://github.com/webis-de/small-text/tree/v1.3.3/examples/examplecode>`_

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

Small-Text has been introduced in detail in the EACL23 System Demonstration Paper `Small-Text: Active Learning for Text Classification in Python <https://aclanthology.org/2023.eacl-demo.11/>`_ which can be cited as follows:

.. code-block:: text

    @inproceedings{schroeder2023small-text,
        title = "Small-Text: Active Learning for Text Classification in Python",
        author = {Schr{\"o}der, Christopher  and  M{\"u}ller, Lydia  and  Niekler, Andreas  and  Potthast, Martin},
        booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
        month = may,
        year = "2023",
        address = "Dubrovnik, Croatia",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.eacl-demo.11",
        pages = "84--95"
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
   errata
   bibliography

:ref:`genindex` | :ref:`modindex` | :ref:`search`
