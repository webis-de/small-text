.. py:currentmodule:: small_text.initialization.strategies

==============
Initialization
==============

Initialization strategies provide the initial labelings from which the first classifier is created.
They are merely intended for experimental purposes and therefore some of them may require knowledge about the true labels.

Initialization Strategies
-------------------------

For single-label scenarios:

* :py:func:`random_initialization`
* :py:func:`random_initialization_balanced`

For single-label and multi-label scenarios:

* :py:func:`random_initialization_stratified`


Methods
-------

.. autofunction:: random_initialization
.. autofunction:: random_initialization_stratified
.. autofunction:: random_initialization_balanced
