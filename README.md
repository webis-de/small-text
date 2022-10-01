[![PyPI](https://img.shields.io/pypi/v/small-text/v1.1.0)](https://pypi.org/project/small-text/)
[![Conda Forge](https://img.shields.io/conda/v/conda-forge/small-text?label=conda-forge)](https://anaconda.org/conda-forge/small-text)
[![codecov](https://codecov.io/gh/webis-de/small-text/branch/master/graph/badge.svg?token=P86CPABQOL)](https://codecov.io/gh/webis-de/small-text)
[![Documentation Status](https://readthedocs.org/projects/small-text/badge/?version=v1.1.0)](https://small-text.readthedocs.io/en/v1.1.0/) 
![Maintained Yes](https://img.shields.io/badge/maintained-yes-green)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](CONTRIBUTING.md)
[![MIT License](https://img.shields.io/github/license/webis-de/small-text)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7072841.svg)](https://zenodo.org/record/7072841)
[![Twitter URL](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fwebis-de%2Fsmall-text)](https://twitter.com/intent/tweet?text=https%3A%2F%2Fgithub.com%2Fwebis-de%2Fsmall-text)

<p align="center">
<img width="372" height="80" src="https://raw.githubusercontent.com/webis-de/small-text/master/docs/_static/small-text-logo.png" alt="small-text logo" />
</p>

> Active Learning for Text Classifcation in Python.
<hr>

[Installation](#installation) | [Quick Start](#quick-start) | [Contribution](CONTRIBUTING.md) | [Changelog][changelog] | [**Docs**][documentation_main]

Small-Text provides state-of-the-art **Active Learning** for Text Classification. 
Several pre-implemented Query Strategies, Initialization Strategies, and Stopping Critera are provided, 
which can be easily mixed and matched to build active learning experiments or applications.

**What is Active Learning?**  
[Active Learning](https://small-text.readthedocs.io/en/latest/active_learning.html) allows you to efficiently label training data in a small data scenario.


## Features

- Provides unified interfaces for Active Learning so that you can 
  easily mix and match query strategies with classifiers provided by [sklearn](https://scikit-learn.org/), [Pytorch](https://pytorch.org/), or [transformers](https://github.com/huggingface/transformers).
- Supports GPU-based [Pytorch](https://pytorch.org/) models and integrates [transformers](https://github.com/huggingface/transformers) 
  so that you can use state-of-the-art Text Classification models for Active Learning.
- GPU is supported but not required. In case of a CPU-only use case, 
  a lightweight installation only requires a minimal set of dependencies.
- Multiple scientifically evaluated components are pre-implemented and ready to use (Query Strategies, Initialization Strategies, and Stopping Criteria).

## News

- **Version 1.1.0** ([v1.1.0][changelog_1.1.0]): Highlights - October 01, 2022
  - A small-text package on [conda-forge](https://anaconda.org/conda-forge/small-text) is now available.
  - Early stopping and model selection have been reworked.
  - One new query strategy and three new stopping criteria have been added.

- **Version 1.0.1** ([v1.0.1][changelog_1.0.1]): Highlights - September 12, 2022
  - Minor bug fix release that fixes notebook and code example links that caused problems by pointing to the latest main branch.

- **Use Small-Text from the Rubrix User Interface** - July 16, 2022
  - We are happy to announce that the great team at [rubrix][rubrix] has worked hard to provide a  
    comprehensive [tutorial on how to use small-text from within the rubrix user interface][rubrix_al_tutorial].

- **Version 1.0.0** ([v1.0.0][changelog_1.0.0]): Highlights - June 13, 2022
  - We're out of beta 🎉!
  - This release mainly consists of code cleanup, documentation, and repository organization.


[For a complete list of changes, see the change log.][changelog]

## Installation

Small-Text can be easily installed via pip:

```bash
pip install small-text
```

For a full installation include the transformers extra requirement:

```bash
pip install small-text[transformers]
```

It requires Python 3.7 or newer. For using the GPU, CUDA 10.1 or newer is required. 
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
| 1 | [Intro: Active Learning for Text Classification with Small-Text](https://github.com/webis-de/small-text/blob/v1.1.0/examples/notebooks/01-active-learning-for-text-classification-with-small-text-intro.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/webis-de/small-text/blob/v1.0.0/examples/notebooks/01-active-learning-for-text-classification-with-small-text-intro.ipynb) |
| 2 | [Using Stopping Criteria for Active Learning](https://github.com/webis-de/small-text/blob/v1.1.0/examples/notebooks/02-active-learning-with-stopping-criteria.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/webis-de/small-text/blob/v1.0.0/examples/notebooks/02-active-learning-with-stopping-criteria.ipynb) |

### Showcase

- [Tutorial: 👂 Learn actively, and listen carefully to small-text.][rubrix_al_tutorial] (Use small-text conveniently from the [rubrix][rubrix] UI.)  

## Documentation

Read the latest documentation [here][documentation_main]. Noteworthy pages include:

- [Overview of Query Strategies][documentation_query_strategies]
- [Reproducibility Notes][documentation_reproducibility_notes]


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


[documentation_main]: https://small-text.readthedocs.io/en/v1.1.0/
[documentation_install]: https://small-text.readthedocs.io/en/v1.1.0/install.html
[documentation_query_strategies]: https://small-text.readthedocs.io/en/v1.1.0/components/query_strategies.html
[documentation_reproducibility_notes]: https://small-text.readthedocs.io/en/v1.1.0/reproducibility_notes.html
[changelog]: https://small-text.readthedocs.io/en/latest/changelog.html
[changelog_1.0.0]: https://small-text.readthedocs.io/en/latest/changelog.html#version-1-0-0-2022-06-14
[changelog_1.0.1]: https://small-text.readthedocs.io/en/latest/changelog.html#version-1-0-1-2022-09-12
[changelog_1.1.0]: https://small-text.readthedocs.io/en/latest/changelog.html#version-1-1-0-2022-10-01
[rubrix]: https://github.com/recognai/rubrix
[rubrix_al_tutorial]: https://rubrix.readthedocs.io/en/stable/tutorials/active_learning_with_small_text.html
