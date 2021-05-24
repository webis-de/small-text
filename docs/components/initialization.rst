.. py:currentmodule:: active_learning.initialization.strategies

==============
Initialization
==============

Initialization strategies provide the initial labelings from which the first classifier is created.
They are merely intended for experimental purposes and therefore some of them may require knowledge about the true labels.

Initialization Strategies
-------------------------

* :py:func:`random_initialization`
* :py:func:`random_initialization_stratified`
* :py:func:`random_initialization_balanced`


Functions
---------

.. autofunction:: random_initialization
.. autofunction:: random_initialization_stratified
.. autofunction:: random_initialization_balanced
