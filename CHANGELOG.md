# Changelog

## [Next] - unreleased

### Changed

- Code examples:
  - Code structure was  unified.
  - Number of iterations can now be passed via an cli argument.

## [1.0.0b3] - 2022-03-06

### Added

- New query strategy: [ContrastiveActiveLearning](https://github.com/webis-de/small-text/blob/v1.0.0b3/small_text/query_strategies/strategies.py).
- Added [Reproducibility Notes](https://small-text.readthedocs.io/en/v1.0.0b3/reproducibility_notes.html).

### Changed

- Cleaned up and unified argument naming: The naming of variables related to datasets and 
  indices has been improved and unified. The naming of datasets had been inconsistent, 
  and the previous `x_` notation for indices was a relict of earlier versions of this library and 
  did not reflect the underlying object anymore.
  - `PoolBasedActiveLearner`:
    - attribute `x_indices_labeled` was renamed to `indices_labeled`
    - attribute `x_indices_ignored` was unified to `indices_ignored`
    - attribute `queried_indices` was unified to `indices_queried`
    - attribute `_x_index_to_position` was named to `_index_to_position`
    - arguments `x_indices_initial`, `x_indices_ignored`, and `x_indices_validation` were
      renamed to `indices_initial`, `indices_ignored`, and `indices_validation`. This affects most 
      methods of the `PoolBasedActiveLearner`.
    
  - `QueryStrategy`
    - old: `query(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n=10)`
    - new: `query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10)`
    
  - `StoppingCriterion`
    - old: `stop(self, active_learner=None, predictions=None, proba=None, x_indices_stopping=None)`
    - new: `stop(self, active_learner=None, predictions=None, proba=None, indices_stopping=None)`

- Renamed environment variable which sets the small-text temp folder from `ALL_TMP` to `SMALL_TEXT_TEMP`


## [1.0.0b2] - 2022-02-22

Bugfix release.

### Fixed

- Fix links to the documentation in README.md and notebooks.


## [1.0.0b1] - 2022-02-22

First beta release with multi-label functionality and stopping criteria.

### Added

- Added a changelog.
- All provided classifiers are now capable of multi-label classification.

### Changed

- Documentation has been overhauled considerably.
- `PoolBasedActiveLearner`: Renamed `incremental_training` kwarg to `reuse_model`.
- `SklearnClassifier`: Changed `__init__(clf)` to `__init__(model, num_classes, multi_Label=False)`
- `SklearnClassifierFactory`: `__init__(clf_template, kwargs={})` to `__init__(base_estimator, num_classes, kwargs={})`.
- Refactored `KimCNNClassifier` and `TransformerBasedClassification`.

### Removed

- Removed `device` kwarg from `PytorchDataset.__init__()`, 
`PytorchTextClassificationDataset.__init__()` and `TransformersDataset.__init__()`.
