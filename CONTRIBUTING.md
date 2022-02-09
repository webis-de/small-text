# Contributing to active-learning

1. License: This project is licensed under the MIT License (see [LICENSE](LICENSE)).
    By contributing you agree to license your contributions under this license as well.
2. All contributions are subject to the [Developer Certificate of Origin](DCO.md).

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
```


## Documentation Conventions

### Spellings

- multi-label instead of multi label (analogous: multi-class)
- dataset instead of data set (but: train set, test set)


## Release Checklist

### Raising the Version

- Update small_text/version.json
- README.md
  - Documentation Badge should link to the version of the most recent release (link AND img)
  - Link references at the bottom should point to the most recent release

## Contributors

Thanks goes to...

- Erik Körner ([@Querela](https://github.com/querela)), for the help with packaging and CI.

And to the many testers who gave their feedback.
