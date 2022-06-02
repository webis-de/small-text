# Contributing to Small-Text

Contributions are always welcome. We need help, among others, in the form of code or 
documentation.

❤️ Even if you tend to doubt yourself, we encourage you to try and pick a small part 
of small-text that can be improved.
*Every little bit of effort is appeciated and we welcome everyone who is willing to learn.*

[Click here to see the list of contributors.](CONTRIBUTORS.md)

---

## Table of Contents

1. [How to Contribute](#how-to-contribute)
2. [Development](#development)
3. [Documentation](#documentation)
4. [Workflows](#workflows)

---

## License / Preliminaries

**Important:** By contributing you agree to the following:

1. License: This project is licensed under the MIT License (see [LICENSE](LICENSE)).
    By contributing you agree to license your contributions under this license as well.
2. All contributions are subject to the [Developer Certificate of Origin](DCO.md).

You are responsible to guarantee that the code is a) either written by yourself or,
or b) that it is both usable under the MIT License **and** properly attributed.
*Please mark any code fragments that you did not write yourself in the code and mention them in 
the pull request so that a second person can review this.*

---

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

---

## Development

### Code Conventions

1. Code style should adhere to the [.flake8 config](.flake8) (except in justified cases).

```bash
flake8
```

### Building the Documentation

The documentation can be generated using sphinx:

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

---

## Documentation

This section refers to the (sphinx) online documentation. 
This may include adapting certain parts of the docstrings but the focus here is the final result.
See the subsection under [Development](#development) on [how to build the documentation](#building-the-documentation).

### Documentation Conventions

The paradigma here is that 1. the same thing should unambiguously be referred to in the same way 
(i.e. don't use different names or spellings for one single concept) and 2. similar things should be treated in the same way.

#### General

We use [numpydoc docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).

#### Spellings

- "small-text" is preferred over "Small-Text" in cases where the first letter is capitalized the latter should be used
- multi-label instead of multi label (analogous: multi-class)
- dataset instead of data set (but: train set, test set)

---

## Workflows

### Release Checklist

The following steps need to be done before a new release can be created. 

#### Raising the Version

- Update small_text/version.json
- examples/notebooks: Set the small-text version to the new release.
- README.md
  - Documentation Badge should link to the version of the most recent release (link AND img)
  - Link references at the bottom should point to the most recent release
- Set the version and date in CHANGELOG.md
  - Make sure the changelog is complete
- Update the "News" section in README.md

#### Checking for Correctness

- Check every step of the above Pull Request Checklist
- Check if the CI Pipeline runs successfully
- Run the sphinx doctests

  - ```bash
    sphinx-build SOURCEDIR OUTPUTDIR -b doctest
    ```

### Create a Release

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
