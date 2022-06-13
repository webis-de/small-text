.. py:currentmodule:: small_text.initialization.strategies

==============
Initialization
==============

Initialization (sampling) strategies provide the initial labelings from which the first classifier is created.
Some of them may require knowledge about the true labels and therefore they are merely intended for experimental purposes.

In an application setting you must provide an initial set of labels instead (or use a cold start approach, which is not yet supported).

Initialization Strategies
-------------------------

For the single-label scenario:

* :py:func:`random_initialization`
* :py:func:`random_initialization_balanced`

For single-label and multi-label scenarios:

* :py:func:`random_initialization_stratified`


Methods
-------

.. autofunction:: random_initialization
.. autofunction:: random_initialization_stratified
.. autofunction:: random_initialization_balanced
