# Migration Guide

The migration guide is intended to help you adapt your code after major version releases, which can and are likely to break some interfaces. 
This is not an exhaustive list of changes, but here we try to collect changes that break existing code.

## 1.3.0 to 2.0.0

*Work in progress. If in doubt, look at the code and docstrings.*

### Interfaces Changes
- [PoolBasedActiveLearner](https://small-text.readthedocs.io/en/latest/api/active_learner.html#activelearner-api):  
  `initialize_data()` has been changed to `initialize()`. The method now takes a list of initial indices or an initialized first (proxy-)model.

- SetFitClassification: `model_kwargs` and `trainer_kwargs` are now attached to `SetFitModelArguments` instead of `SetFitClassification`.

### Renamed Classes

The following classes and variables have been renamed for consistency:

- KimCNNFactory -> KimCNNClassifierFactory

- KimCNNClassifierFactory, TransformerBasedClassificationFactory: the argument for classification keyword arguments (`kwargs`) in the respective `__init__()` has been renamed to `classification_kwargs`

### Moved Classed / Changed Import Paths

- SEALS has been moved from `small_text.query_strategies.strategies` to `small_text.query_strategies.subsampling`. You either have to fix the import or switch to use the convenience report:
  ```
  from small_text import SEALS
  ```
