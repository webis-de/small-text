# Changelog

## [1.0.0b1] - unreleased

First beta release with multi-label functionality.

### Added

- Added a changelog.
- All provided classifiers are now capable of multi-label classification.

### Changed

- Documentation has been overhauled considerably.

### Removed

- Removed `device` kwarg from `PytorchDataset.__init__()`, 
`PytorchTextClassificationDataset.__init__()` and `TransformersDataset.__init__()`
