# Changelog

## [1.0.0b3] - unreleased

### Changed

The naming of variables related to datasets and indices has been unified 
(and also improved, since the previous `x_` notation, stemming from earlier versions of this library,
does not reflect the underlying object anymore).

- `QueryStrategy`
  - old: `query(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n=10)`
  - new: `query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10)`

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
