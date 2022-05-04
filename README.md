[![PyPI](https://img.shields.io/pypi/v/small-text)](https://pypi.org/project/small-text/)
[![codecov](https://codecov.io/gh/webis-de/small-text/branch/master/graph/badge.svg?token=P86CPABQOL)](https://codecov.io/gh/webis-de/small-text)
[![Documentation Status](https://readthedocs.org/projects/small-text/badge/?version=v1.0.0b4)](https://small-text.readthedocs.io/en/v1.0.0b4/) 
![Maintained Yes](https://img.shields.io/badge/maintained-yes-green)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](CONTRIBUTING.md)
![GitHub](https://img.shields.io/github/license/webis-de/small-text)

<p align="center">
<img width="372" height="80" src="https://raw.githubusercontent.com/webis-de/small-text/master/docs/_static/small-text-logo.png" alt="small-text logo" />
</p>

> Active Learning for Text Classifcation in Python.
<hr>

[Installation](#installation) | [Quick Start](#quick-start) | [Changelog][changelog] | [**Docs**][documentation_main]

Small-Text provides state-of-the-art **Active Learning** for Text Classification. 
Several components are provided, which are abstracted via generic interfaces, 
so that you can easily mix and match many classifiers and query strategies 
to build active learning experiments or applications.

**What is Active Learning?**
[Active Learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) allows you to efficiently label training data in a small-data scenario.


## Features

- Provides unified interfaces for Active Learning so that you can 
  easily mix and match query strategies with classifiers provided by [sklearn](https://scikit-learn.org/), [Pytorch](https://pytorch.org/), or [transformers](https://github.com/huggingface/transformers).
- Supports GPU-based [Pytorch](https://pytorch.org/) models and integrates [transformers](https://github.com/huggingface/transformers) 
  so that you can use state-of-the-art Text Classification models for Active Learning.
- GPU is optional: In case of a CPU-only use case, 
  a lightweight installation only requires a minimal set of dependencies.
- Multiple scientifically evaluated components are pre-implemented and ready to use (Query Strategies, Initialization Strategies, and Stopping Criteria).

## News

The news section provides only a brief update showing a small selection of highlights. Read the [changelog][changelog] for a complete list of changes.

- **May Beta Release** ([v1.0.0b4][changelog_1.0.0b4]) - May 04, 2022
  - Two new query strategies: [DiscriminativeActiveLearning](https://github.com/webis-de/small-text/blob/v1.0.0b4/small_text/query_strategies/strategies.py), 
    [SEALS](https://github.com/webis-de/small-text/blob/v1.0.0b4/small_text/query_strategies/strategies.py).
  - `Dataset` interface now has a `clone()` method.
  - Added a concept for [optional dependencies](https://small-text.readthedocs.io/en/v1.0.0b4/install.html#optional-dependencies).

- **March Beta Release** ([v1.0.0b3][changelog_1.0.0b3]) - March 06, 2022
  - Consolidated interfaces: Renamed and unified some arguments before v1.0.0.
  - New query strategy: [ContrastiveActiveLearning](https://github.com/webis-de/small-text/blob/v1.0.0b3/small_text/query_strategies/strategies.py).  

- 🎉 **Beta Release** ([v1.0.0b1][changelog_1.0.0b1]) - February 22, 2022
  - New features: multi-label classification and stopping criteria are now supported.
  - Added/revised large parts of the documentation.

## Installation

Small-text can be easily installed via pip:

```bash
pip install small-text
```

For a full installation include the transformers extra requirement:

```bash
pip install small-text[transformers]
```

Requires Python 3.7 or newer. For using the GPU, CUDA 10.1 or newer is required. 
More information regarding the installation can be found in the 
[documentation][documentation_install].


## Quick Start

For a quick start, see the provided examples for [binary classification](examples/examplecode/binary_classification.py),
[pytorch multi-class classification](examples/examplecode/pytorch_multiclass_classification.py), and 
[transformer-based multi-class classification](examples/examplecode/transformers_multiclass_classification.py),
or check out the notebooks.

### Notebooks

| # | Notebook | |
| --- | -------- | --- |
| 1 | [Intro: Active Learning for Text Classification with Small-Text](examples/notebooks/01-active-learning-for-text-classification-with-small-text-intro.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/webis-de/small-text/blob/master/examples/notebooks/01-active-learning-for-text-classification-with-small-text-intro.ipynb) |
| 2 | [Using Stopping Criteria for Active Learning](examples/notebooks/02-active-learning-with-stopping-criteria.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/webis-de/small-text/blob/master/examples/notebooks/02-active-learning-with-stopping-criteria.ipynb) |
## Documentation

Read the latest documentation (currently work in progress) [here][documentation_main].

## Alternatives

[modAL](https://github.com/modAL-python/modAL), [ALiPy](https://github.com/NUAA-AL/ALiPy), [libact](https://github.com/ntucllab/libact)

## Contribution

Contributions are welcome. Details can be found in [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgments

This software was created by Christopher Schröder ([@chschroeder](https://github.com/chschroeder)) at Leipzig University's [NLP group](http://asv.informatik.uni-leipzig.de/) 
which is a part of the [Webis](https://webis.de/) research network. 
The encompassing project was funded by the Development Bank of Saxony (SAB) under project number 100335729.

## Citation

A preprint which introduces small-text is available here:  
[Small-Text: Active Learning for Text Classification in Python](https://arxiv.org/abs/2107.10314). 

```
@misc{schroeder2021smalltext,
    title={Small-Text: Active Learning for Text Classification in Python}, 
    author={Christopher Schröder and Lydia Müller and Andreas Niekler and Martin Potthast},
    year={2021},
    eprint={2107.10314},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## License

[MIT License](LICENSE)


[documentation_main]: https://small-text.readthedocs.io/en/v1.0.0b4/
[documentation_install]: https://small-text.readthedocs.io/en/v1.0.0b4/install.html
[changelog]: https://small-text.readthedocs.io/en/latest/changelog.html]
[changelog_1.0.0b1]: https://small-text.readthedocs.io/en/latest/changelog.html#b1-2022-02-22
[changelog_1.0.0b3]: https://small-text.readthedocs.io/en/latest/changelog.html#b3-2022-03-06
[changelog_1.0.0b4]: https://small-text.readthedocs.io/en/latest/changelog.html#b4-2022-05-04
