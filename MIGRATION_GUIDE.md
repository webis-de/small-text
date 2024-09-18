# Migration Guide

## 1.3.0 to 2.0.0


### Renamed Classes

The following classes amd variables have been renamed for consistency:

- KimCNNFactory -> KimCNNClassifierFactory

- KimCNNClassifierFactory, TransformerBasedClassificationFactory: the argument for classification keyword arguments (`kwargs`) in the respective `__init__()` has been renamed to `classification_kwargs`

### Moved Classed / Changed Import Paths

- SEALS has been moved from `small_text.query_strategies.strategies` to `small_text.query_strategies.subsampling`. You either have to fix the import or switch to use the convenience report:
  ```
  from small_text import SEALS
  ```
