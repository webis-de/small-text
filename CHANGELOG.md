# Changelog

## Version 2.0.0.dev3 - 2025-08-17

*This section is going to be updated and will become v2.0.0 eventually.*

This is the first release with breaking changes, coming sooner than we would have liked. 

The need for this came from relying on some legacy interfaces in torchtext for too long, which now have been dropped. The result was that with never PyTorch Versions, which also requires a matching torchtext version, which in turn broke some parts of the PyTorch integration.

On the other hand, this also allowed us to deal with further issues that contain breaking changes but needed to be done eventually. **All of this should not cause you too much trouble**, but still to make the transition as easy as possible there is a [migration guide](https://github.com/webis-de/small-text/blob/v2.0.0.dev2/MIGRATION_GUIDE.md), which lists all breaking changes.

### Added

- General
  - Python requirements raised to Python 3.9 since Python 3.8 has reached [end of life on 2024-10-07](https://devguide.python.org/versions/).
  - Dropped torchtext as an integration dependency. For individual use cases it can of course still be used.
  - Added environment variables `SMALL_TEXT_PROGRESS_BARS` and `SMALL_TEXT_OFFLINE` to control the default behavior for progress bars and model downloading.
  - Minimum required SetFit version has been raised to `1.1.2` ([#71](https://github.com/webis-de/small-text/issues/71)).
- PoolBasedActiveLearner:
  - `initialize_data()` has been replaced by `initialize()` which can now also be used to provide an initial model in cold start scenarios. ([#10](https://github.com/webis-de/small-text/pull/10))
- Datasets
  - Validation has been added to prevent TextDataset objects from containing None items (instead of str) either during initialization or when setting the x property. ([#73](https://github.com/webis-de/small-text/pull/73))
- Classification:
  - All PyTorch-classifiers (KimCNN, TransformerBasedClassification, SetFitClassification) now support `torch.compile()` which can be enabled on demand. (Requires PyTorch >= 2.0.0). 
  - All PyTorch-classifiers (KimCNN, TransformerBasedClassification, SetFitClassification) now support Automatic Mixed Precision. 
  - All classifiers have gained an additional convenience check that raises an error on single-/multi-label.
  - `SetFitClassification.__init__() <https://github.com/webis-de/small-text/blob/v2.0.0.dev2/small_text/integrations/transformers/classifiers/setfit.py>`__ now has a verbosity parameter (similar to `TransformerBasedClassification`) through which you can control the progress bar output of `SetFitClassification.fit()`.
  - TransformerBasedClassification:
    - Removed unnecessary `token_type_ids` keyword argument in model call.
    - Additional keyword args for config, tokenizer, and model can now be configured.
  - SetFitClassification:
    - Additional keyword args for trainer and model are now attached to `SetFitModelArguments` instead of `SetFitClassification`.
    - Removed `setfit_train_kwargs` from `SetFitClassification.fit()`.
    - Using a differentiable head no longer requires a validation set.

- Embeddings:
  - Prevented unnecessary gradient computations for some embedding types and unified code structure.
- Pytorch:
  - Added an `inference_mode()` context manager that applies `torch.inference_mode` or `torch.no_grad` for older Pytorch versions.
- Query Strategies:
  - New strategies: [DiscriminativeRepresentationLearning](https://github.com/webis-de/small-text/blob/v2.0.0.dev2/small_text/integrations/pytorch/query_strategies/strategies.py), [LabelCardinalityInconsistency](https://github.com/webis-de/small-text/blob/v2.0.0.dev2/small_text/query_strategies/multi_label.py), [ClassBalancer](https://github.com/webis-de/small-text/blob/v2.0.0.dev2/small_text/query_strategies/class_balancing.py), and [ProbCover](https://github.com/webis-de/small-text/blob/v2.0.0.dev2/small_text/query_strategies/vector_space.py).
  - Query strategies now have a [tie-breaking mechanism](https://github.com/webis-de/small-text/blob/v2.0.0.dev2/small_text/query_strategies/base.py) to randomly permutate when there is a tie in scores.
  - Added `ScoringMixin` to enable a reusable scoring mechanism for query strategies.
  - [LightweightCoreset](https://github.com/webis-de/small-text/blob/v2.0.0.dev2/small_text/query_strategies/coresets.py) can now process input in batches. ([#23](https://github.com/webis-de/small-text/issues/23))
- Vector Index Functionality:
  - A new vector index API provides implementations over a unified interface to use different implementations for k-nearest neighbor search.
  - Existing strategies that used a hard-coded vector search ([ContrastiveActiveLearning][contrastive_active_learning], [SEALS][seals], [AnchorSubsampling][anchor_subsampling]) have been adapted and can now be used with different vector index implementations.

### Fixed 

- Fixed a bug where the `clone()` operation wrapped the labels, which then raised an error. This affected the single-label scenario for PytorchTextClassificationDataset and TransformersDataset. ([#35](https://github.com/webis-de/small-text/issues/35))
- Fixed a bug where the batching in `greedy_coreset()` and `lightweight_coreset()` resulted in incorrect batch sizes. ([#50](https://github.com/webis-de/small-text/issues/50))
- Fixed a bug where `lightweight_coreset()` failed when computing the norm of the elementwise mean vector.

### Changed

- Minimum required Python version was raised to 3.9.
- General
  - Moved `split_data()` method from `small_text.data.datasets` to `small_text.data.splits`.
- Dependencies
  - Raised minimum required setfit version to 1.1.0.
- Classification:
  - The `initialize()` methods of all PyTorch-classifiers (KimCNN, TransformerBasedClassification, SetFitClassification) are now more unified. ([#57](https://github.com/webis-de/small-text/issues/57))
  - KimCNNClassifier / TransformerBasedClassification: model selection is now disabled by default. Also, it no longer saves models when disabled, thereby greatly reducing the runtime.
- Utils
  - `init_kmeans_plusplus_safe()` now supports weighted kmeans++ initialization for `scikit-learn>=1.3.0`.
  - `get_progress_bars_default()` has been renamed to `get_show_progress_bar_default()`. It's functionality has been slightly generalized and always returns a boolean.
- Documentation:
  - Extended the section on initializing the active learning process.

### Removed

- Deprecated functionality
  - Removed `default_tensor_type()` method.
  - Removed `small_text.utils.labels.get_flattened_unique_labels()`.
  - Removed `small_text.integrations.pytorch.utils.labels.get_flattened_unique_labels()`.
  - Classification
    - Removed early stopping legacy arguments in `__init__()` for KimCNN and TransformerBasedClassification. (Use `fit()` keyword arguments instead.) 
    - Removed model selection legacy argument in `TransformerBasedClassification.__init__()`.
- The explicit installation instruction for conda was removed, but the small-text conda-forge package will remain.
- Removed gensim dependency in the PyTorch multiclass example.

---

## Version 1.4.1 - 2024-08-18

### Fixed

- Fixed an out of bounds error that occurred when `DiscriminativeActiveLearning` queries all remaining unlabeled data.
- Fixed typos/wording in PoolBasedActiveLearner docstrings.
- Pinned SetFit version in notebook example. ([#64](https://github.com/webis-de/small-text/issues/64))
- Fixed an out of bounds error that could occur in `SetFitClassification` for both 32bit systems and Windows. ([#66](https://github.com/webis-de/small-text/issues/66))
- Fixed errors in notebook examples that occurred with more recent seaborn / matplotlib versions.

### Changed

- Documentation: added links to bibliography. ([#65](https://github.com/webis-de/small-text/issues/65))

### Contributors

[@pdhall99](https://github.com/pdhall99)

---

## Version 1.4.0 - 2024-06-09

### Added

- New query strategy: [AnchorSubsampling][anchor_subsampling].

### Fixed

- Changed the way how the seed is controlled  in `SetFitClassification` since the seed was fixed unless explicitly set via the respective trainer keyword argument.

### Changed


---

## Version 1.3.3 - 2023-12-29

### Changed

- An [errata](https://small-text.readthedocs.io/en/v1.3.3/errata.html) section was added to the documentation.

### Fixed

- Fixed a deviation from the paper, where `DeltaFScore` also took into account the agreement in predictions of the negative label. ([#51](https://github.com/webis-de/small-text/pull/51))
- Fixed a bug in `KappaAverage` that affected the stopping behavior. ([#52](https://github.com/webis-de/small-text/pull/52))

### Contributors

[@zakih2](https://github.com/zakih2), [@vmanc](https://github.com/vmanc)

---

## Version 1.3.2 - 2023-08-19

### Fixed

- Fixed a bug in `TransformerBasedClassification`, where `validations_per_epoch>=2` left the model in eval mode. ([#40](https://github.com/webis-de/small-text/issues/40))

---

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
    and [SEALS][seals].

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

- New query strategy: [ContrastiveActiveLearning][contrastive_active_learning].
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


[contrastive_active_learning]: https://github.com/webis-de/small-text/blob/v2.0.0.dev2/small_text/query_strategies/strategies.py
[anchor_subsampling]: https://github.com/webis-de/small-text/blob/v2.0.0.dev2/small_text/query_strategies/subsampling.py
[seals]: https://github.com/webis-de/small-text/blob/v1.0.0b4/small_text/query_strategies/strategies.py
