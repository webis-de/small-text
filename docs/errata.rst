======
Errata
======

In this section, we will document changes that affected a method's correctness.

Although a lot of effort is spent in this repository to ensure correctness (through extensive unit and integration testing), errors will happen. 
The best we can do is to leverage the shared knowledge of all contributors, to spot and fix these issues.
Most importantly, we provide full transparency by documenting such cases here.

In case you are using small-text in a scientific context, make sure to document the version that has been used.


Classifiers
===========

- TransformerBasedClassification: parameter groups were omitted when using the layer-specific fine-tuning functionality (`#38 <https://github.com/webis-de/small-text/pull/38>`_; fixed in v1.3.1).

Stopping Critera
================

- DeltaFScore: the implementation deviated from the original paper by taking agreement of the negative label into account (`#51 <https://github.com/webis-de/small-text/pull/51>`_; fixed in v1.3.3).
- KappaAverage: an attempted fix to prevent division by zero errors had made the implementation incorrect (`#52 <https://github.com/webis-de/small-text/pull/52>`_; fixed in v1.3.3).
