# Contributing to Small-Text

Happy to see you here. Contributions are always welcome. We need help, among others, 
in the form of code or documentation.

❤️ Even if you tend to doubt yourself, we encourage you to try and pick a small part 
of small-text that can be improved.
*Every little bit of effort is appeciated and we welcome everyone who is willing to learn.*

[Click here to see the list of contributors.](CONTRIBUTORS.md)

---

## Table of Contents

1. [How to Contribute](#how-to-contribute)
2. [Support us](#support-us)
3. [Development](#development)
4. [Documentation](#documentation)
5. [Workflows](#workflows)

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

For contributions, there are both code and non-code contributions. Both are equally welcome.
As a small token of appreciation, with your first contribution you can put yourself on the [list of contributors](CONTRIBUTORS.md)
(if the commit contributes to code or documentation).

### Types of Contribution

- Report Bugs
- Suggest Enhancements
- Documentation
- Code

### Support us

Apart from contributions, if you like our project you can also support us by:

- Consider giving us a github star
- Use small-text in your applications or experiments.
- Write a blog post small-text or your specific use case
- Share small-text on Twitter or LinkedIn.

### How to work on an issue (bug, feature, documentation, other)
 
1. Search for an existing issue related to your problem.
   If none exists yet then create an issue stating the problem and that you are working on it.
2. Fork the repository. Create a new branch that branches off the **dev** branch.
3. Apply your changes.
4. Make sure the unit / integration tests succeed.
5. Create a pull request. (You can also create the pull requets earlier if you mark it as "work in progress").

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

### Testing

If you do this for the first time you have to install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

#### Running the unit tests

```bash
pytest -m 'not optional' --cov small_text --cov-append tests/unit/
```

Without pytorch tests:

```bash
pytest -m 'not pytorch and not optional' --cov small_text --cov-append tests/unit/
```

#### Running the integration tests:

```bash
pytest -m 'not optional' --cov small_text --cov-append tests/integration
```

Without pytorch tests:

```bash
pytest -m 'not pytorch and not optional' --cov small_text --cov-append tests/integration
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

The same thing should unambiguously be referred to in the same way 
(i.e. don't use different names or spellings for one single concept) and similar things should be treated in the same way.

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
  - Check notebook using a temporary small-text installation via github.
- docs/index.rst: Set the small-text version to the new release.
- README.md (Links "Notebooks" / "Code Examples")
  - Documentation Badge should link to the version of the most recent release (link AND img)
  - Link references at the bottom should point to the most recent release
  - Notebook links should point to the most recent release
- Set the version and date in CHANGELOG.md
  - Make sure the changelog is complete
- Update the "News" section in README.md

#### Checking for Correctness

- Check every step of the above Pull Request Checklist
- Check if the CI Pipeline runs successfully
- Documentation
  - Check if conda versions in docs/install.rst match those in setup.py
  - Run the sphinx doctests
    - ```bash
      sphinx-build SOURCEDIR OUTPUTDIR -b doctest
      ```
  - Run sphinx linkcheck:
    - ```bash
      sphinx-build SOURCEDIR OUTPUTDIR -b linkcheck
      ```
    - Links containing the new version tag (e.g., v1.0.0) will fail.

### Create a Release

- Create a git tag: `v<VERSION>` (e.g., v1.0.0)
- Push tags and code to github
- Create new version at Read the Docs
  - Run sphinx linkcheck (yes, again):
  - ```bash
    sphinx-build SOURCEDIR OUTPUTDIR -b linkcheck
    ```

- Create a release on testpypi
  - If successful:
    - Create a new Zenodo version and update DOI in README.md
    - Delete and create the tag again at the current head commit
    - Create a release on pypi
- Create a release on github
- Create a new condaforge release
