.. py:currentmodule:: small_text.initialization.strategies

==============
Initialization
==============

Active learning has a mutual dependency between the model, which is trained on the instances in the labeled pool,
and the instances in the labeled pool, which are (generally) queried using the model.
To bootstrap active learning, we must provide one or the other.

1. Initialize with labels
2. Initialize with a classifier
3. Use a query strategy that does not depend on a model (during the first iteration)

----

Application Scenario
====================

In the application scenario with a human annotation, there basically three ways to bootstrap the active learning process:

Option 1: Provide Initial Labels
--------------------------------

The simplest way to provide an initial model is to provide a few examples for each class.
On these labeled instances an initial model is created, and the active learning process can be started.

**This is the preferred approach whenever possible** and also helps to test and refine annotation guidelines.
Situations where this is infeasible includes problems with a large number of classes, e.g., providing 10 samples per
class for a 20,000-class classification would be very time-consuming.

Option 2: Provide an Initial Model
----------------------------------

Another option is to simply provide an initial model.
The model gets replaced after the next call to `query()` (unless retraining is prevented),
i.e., you could possible also use a different type of model to bootstrap the active learning process.s

Here, "model" refers simply to the API terminology. You could for example also pass a labeling heuristic,
as long as it adheres to the interface. This example would already be a cold start approach.

Option 3: Cold Start
--------------------

A full cold start just sets the classifier to `None`, but still prepares the ActiveLearner object.
However, this approach is incompatible with most strategies unless the strategy operates without a model.

----

Experimental Scenario
=====================

In the experimental scenario, we simulate active learning on pre-labeled datasets.
Initializing active learning is then as simple as selecting a few instances from the dataset, and using
their labels to create the initial model.
This can either be done with random sampling, which has obvious disadvantages of a possibly highly
skewed label distribution.


Initialization Strategies
-------------------------

For the single-label classification:

* :py:func:`random_initialization`
* :py:func:`random_initialization_balanced`

For single-label and multi-label classification:

* :py:func:`random_initialization_stratified`


Methods
-------

.. autofunction:: random_initialization
.. autofunction:: random_initialization_stratified
.. autofunction:: random_initialization_balanced
