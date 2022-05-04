# Contributing to Small-Text

Contributions are always welcome.

**Important:** By contributing you agree to the following:

1. License: This project is licensed under the MIT License (see [LICENSE](LICENSE)).
    By contributing you agree to license your contributions under this license as well.
2. All contributions are subject to the [Developer Certificate of Origin](DCO.md).

## How to Contribute

### Pull Request Checklist

1. Check that the documentation can be built:

    ```bash
    cd docs && make html
    ```

2. Check that the documentation examples run:
    
    ```bash
    cd docs && make doctest
    ```

## Development

### Code Conventions

1. Code style should adhere to the [.flake8 config](.flake8) (except in justified cases).

```bash
flake8
```

### Building the Documentation

The documentation (currently work in progress) can be generated using sphinx:

```bash
pip install sphinx sphinx-rtd-theme
cd docs/
make
sphinx-build . _build/ -b linkcheck
sphinx-build . _build/ -b doctest
```

### Continuous Integration

We use a CI Pipeline which is unfortunately not public yet, but this might change in the future.

This pipeline checks the following:
- Checks if small-text can be installed with and without extra requirements.
- Checks if the unit/integration tests complete successfully.
- Checks if the examples can be executed successfully.
- Checks whether the documentation can be built.

## Documentation Conventions

### Spellings

- multi-label instead of multi label (analogous: multi-class)
- dataset instead of data set (but: train set, test set)


## Release Checklist

### Raising the Version

- Update small_text/version.json
- examples/notebooks: Set the small-text version to the new release.
- README.md
  - Documentation Badge should link to the version of the most recent release (link AND img)
  - Link references at the bottom should point to the most recent release
- Set the version and date in CHANGELOG.md
  - Make sure the changelog is complete
- Update the "News" section in README.md

### Checking for Correctness

- Check every step of the above Pull Request Checklist
- Check if the CI Pipeline runs successfully
- Run the sphinx doctests

  - ```bash
    sphinx-build SOURCEDIR OUTPUTDIR -b doctest
    ```

### Finalizing

- Create a git tag: `v<VERSION>` (e.g., v1.0.0)
- Push tags and code to github
- Create new version at Read the Docs
  - Run sphinx linkcheck:
  - ```bash
    sphinx-build SOURCEDIR OUTPUTDIR -b linkcheck
    ```
- Create a release on testpypi
  - If successful: Create a release on pypi
- Create a release on github

## Contributors

Thanks goes to...

- Erik Körner ([@Querela](https://github.com/querela)), for the help with packaging and CI.

And to the many testers who gave their feedback.
