# Changelog


## [1.0.0b3] - unreleased

### Added

- New query strategy: [ContrastiveActiveLearning](small_text/query_strategies/strategies.py).

### Changed

The naming of variables related to datasets and indices has been unified 
(and also improved, since the previous `x_` notation, stemming from earlier versions of this library,
does not reflect the underlying object anymore).

- `PoolBasedActiveLearner`
  - attribute `x_indices_labeled` was renamed to `indices_labeled`
  - attribute `x_indices_ignored` was unified to `indices_ignored`
  - attribute `queried_indices` was unified to `indices_queried`
  - attribute `_x_index_to_position` was named to `_index_to_position`
  - `initialize_data(self, indices_initial, y_initial, x_indices_ignored=None, x_indices_validation=None, retrain=True)` changed to `initialize_data(self, indices_initial, y_initial, indices_ignored=None, indices_validation=None, retrain=True)`
  - `query(self, num_samples=10, x=None, query_strategy_kwargs=None)` changed to `query(self, num_samples=10, representation=None, query_strategy_kwargs=None)`
  - `update(self, y, x_indices_validation=None)` changed to `update(self, y, indices_validation=None)`
  - `update_label_at(self, x_index, y, retrain=False, x_indices_validation=None):` changed to `update_label_at(self, index, y, retrain=False, indices_validation=None):`  
  - `ignore_sample_at(self, x_index, retrain=False, x_indices_validation=None)` changed to `ignore_sample_at(self, index, retrain=False, indices_validation=None)`
  - `remove_label_at(self, x_index, retrain=False, x_indices_validation=None)` changed to `remove_label_at(self, x_index, retrain=False, indices_validation=None)`  
 

- `QueryStrategy`
  - old: `query(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n=10)`
  - new: `query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10)`


- `StoppingCriterion`
  - `stop(self, active_learner=None, predictions=None, proba=None, x_indices_stopping=None)` changed to `stop(self, active_learner=None, predictions=None, proba=None, indices_stopping=None)`

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
