# Changelog

## Version 1.3.1 - 2023-07-22

### Fixed

- Fixed a bug where parameter groups were omitted when using `TransformerBasedClassification`'s layer-specific fine-tuning functionality. ([#36](https://github.com/webis-de/small-text/issues/36), [#38](https://github.com/webis-de/small-text/pull/38))
- Fixed a bug where class weighting resulted in `nan` values. ([#39](https://github.com/webis-de/small-text/issues/39))

### Contributors

[@JP-SystemsX](https://github.com/JP-SystemsX)

---

## Version 1.3.0 - 2023-02-21

### Added

- Added dropout sampling to `SetFitClassification <https://github.com/webis-de/small-text/blob/v1.3.0/small_text/integrations/transformers/classifiers/setfit.py>`__.

### Fixed

- Fixed broken link in README.md.
- Fixed typo in README.md. ([#26](https://github.com/webis-de/small-text/pull/26))

### Changed

- The `ClassificationChange <https://github.com/webis-de/small-text/blob/v1.3.0/small_text/stopping_criteria/change.py>`__ stopping criterion now supports multi-label classification.
- Documentation:
  - Updated the active learning setup figure.
  - The documentation of integrations has been reorganized.

### Contributors

[@rmitsch](https://github.com/rmitsch)

---

## Version 1.2.0 - 2023-02-04

### Added

- Added new classifier: [SetFitClassification](https://github.com/webis-de/small-text/blob/v1.2.0/small_text/integrations/transformers/classifiers/setfit.py) which wraps [huggingface/setfit](https://github.com/huggingface/setfit).
- Active Learner:
  - [PoolBasedActiveLearner](https://github.com/webis-de/small-text/blob/v1.2.0/small_text/active_learner.py) now handles keyword arguments passed to the classifier's `fit()` during the `update()` step.
- Query Strategies:
  - New strategy: [BALD](https://github.com/webis-de/small-text/blob/v1.2.0/small_text/query_strategies/bayesian.py).
  - [SubsamplingQueryStrategy](https://github.com/webis-de/small-text/blob/v1.2.0/small_text/query_strategies/strategies.py) now uses the remaining unlabeled pool when more samples are requested than are available.
- Notebook Examples:
  - Revised both existing notebook examples.
  - Added a notebook example for active learning with SetFit classifiers.
  - Added a notebook example for cold start initialization with SetFit classifiers.
- Documentation:
  - A [showcase](https://small-text.readthedocs.io/en/v1.2.0/showcase.html) section has been added to the documentation.

### Fixed

- Distances in [lightweight_coreset](https://github.com/webis-de/small-text/blob/v1.2.0/small_text/query_strategies/coresets.py) were not correctly projected onto the [0, 1] interval (but ranking was unaffected).

### Changed

- [Coreset implementations](https://github.com/webis-de/small-text/blob/v1.2.0/small_text/query_strategies/coresets.py) now use the distance-based  (as opposed to the similarity-based) formulation.

---

## Version 1.1.1 - 2022-10-14

### Fixed

- Model selection raised an error in cases where no model was available for selection ([#21](https://github.com/webis-de/small-text/issues/21)).

---

## Version 1.1.0 - 2022-10-01

### Added

- General:
  - Small-Text package is now available via [conda-forge](https://anaconda.org/conda-forge/small-text). 
  - Imports have been reorganized. You can import all public classes and methods from the top-level package (`small_text`):
    ```
    from small_text import PoolBasedActiveLearner
    ```

- Classification:
  - All classifiers now support weighting of training samples.
  - [Early stopping](https://small-text.readthedocs.io/en/v1.1.0/components/training.html) has been reworked, improved, and documented ([#18](https://github.com/webis-de/small-text/issues/18)).
  - [Model selection](https://small-text.readthedocs.io/en/v1.1.0/components/training.html) has been reworked and documented.
  - **[!]** `KimCNNClassifier.__init()__`: The default value of the (now deprecated) keyword argument `early_stopping_acc` has been changed from `0.98` to `-1` in order to match `TransformerBasedClassification`.
  - **[!]** Removed weight renormalization after gradient clipping.

- Datasets:
  - The `target_labels` keyword argument in `__init()__` will now raise a warning if not passed.
  - Added `from_arrays()` to `SklearnDataset`, `PytorchTextClassificationDataset`, and `TransformersDataset` to construct datasets more conveniently.

- Query Strategies:
  - New multi-label strategy: [CategoryVectorInconsistencyAndRanking](https://github.com/webis-de/small-text/blob/v1.1.0/small_text/query_strategies/multi_label.py).

- Stopping Criteria:
  - New stopping criteria: [ClassificationChange](https://github.com/webis-de/small-text/blob/v1.1.0/small_text/stopping_criteria/change.py), 
    [OverallUncertainty](https://github.com/webis-de/small-text/blob/v1.1.0/small_text/stopping_criteria/uncertainty.py), and 
    [MaxIterations](https://github.com/webis-de/small-text/blob/v1.1.0/small_text/stopping_criteria/utility.py).

### Deprecated

- `small_text.integrations.pytorch.utils.misc.default_tensor_type()` is deprecated without replacement ([#2](https://github.com/webis-de/small-text/issues/2)).
- `TransformerBasedClassification` and `KimCNNClassifier`:
  The keyword arguments for early stopping (early_stopping / early_stopping_no_improvement, early_stopping_acc) that are passed to `__init__()` are now deprecated. Use the `early_stopping`
  keyword argument in the `fit()` method instead ([#18](https://github.com/webis-de/small-text/issues/18)). 


### Fixed
- Classification:
  - `KimCNNClassifier.fit()` and `TransformerBasedClassification.fit()` now correctly
    process the `scheduler` keyword argument ([#16](https://github.com/webis-de/small-text/issues/16)).

### Removed
- Removed the strict check that every target label has to occur in the training data.
  *(This is intended for multi-label settings with many labels; apart from that it is still recommended to make sure that all labels occur.)*

## Version 1.0.1 - 2022-09-12

Minor bug fix release.

### Fixed

Links to notebooks and code examples will now always point to the latest release instead of the latest main branch.

---

## Version 1.0.0 - 2022-06-14

First stable release.

### Changed

- Datasets:
  - `SklearnDataset` now checks if the dimensions of the features and labels match.
- Query Strategies:
  - [ExpectedGradientLengthMaxWord](https://github.com/webis-de/small-text/blob/main/small_text/integrations/pytorch/query_strategies/strategies.py): Cleaned up code and added checks to detect invalid configurations.
- Documentation:
  - The documentation is now available in full width.
- Repository:
  - Versions in this can now be referenced using the respective [Zenodo DOI](https://zenodo.org/record/6641063).

---

## [1.0.0b4] - 2022-05-04

### Added

- General:
  - We now have a concept for [optional dependencies](https://small-text.readthedocs.io/en/v1.0.0b4/install.html#optional-dependencies) which 
    allows components to rely on soft dependencies, i.e. python dependencies which can be installed on demand
    (and only when certain functionality is needed).
- Datasets:
  - The `Dataset` interface now has a `clone()` method 
    that creates an identical copy of the respective dataset.
- Query Strategies:
  - New strategies: [DiscriminativeActiveLearning](https://github.com/webis-de/small-text/blob/v1.0.0b4/small_text/query_strategies/strategies.py) 
    and [SEALS](https://github.com/webis-de/small-text/blob/v1.0.0b4/small_text/query_strategies/strategies.py).

### Changed

- Datasets:
  - Separated the previous `DatasetView` implementation into interface (`DatasetView`) 
    and implementation (`SklearnDatasetView`).
  - Added `clone()` method which creates an identical copy of the dataset.
- Query Strategies:
  - `EmbeddingBasedQueryStrategy` now only embeds instances that are either in the label
    or in the unlabeled pool (and no longer the entire dataset).
- Code examples:
  - Code structure was  unified.
  - Number of iterations can now be passed via an cli argument.
- `small_text.integrations.pytorch.utils.data`:
  - Method `get_class_weights()` now scales the resulting multi-class weights so that the smallest
    class weight is equal to `1.0`.

---

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

---

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
