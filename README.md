[![PyPI](https://img.shields.io/pypi/v/small-text/v1.3.3)](https://pypi.org/project/small-text/)
[![Conda Forge](https://img.shields.io/conda/v/conda-forge/small-text?label=conda-forge)](https://anaconda.org/conda-forge/small-text)
[![codecov](https://codecov.io/gh/webis-de/small-text/branch/master/graph/badge.svg?token=P86CPABQOL)](https://codecov.io/gh/webis-de/small-text)
[![Documentation Status](https://readthedocs.org/projects/small-text/badge/?version=v1.3.3)](https://small-text.readthedocs.io/en/v1.3.3/) 
![Maintained Yes](https://img.shields.io/badge/maintained-yes-green)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](CONTRIBUTING.md)
[![MIT License](https://img.shields.io/github/license/webis-de/small-text)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10443361.svg)](https://zenodo.org/records/10443361)
[![Twitter URL](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fwebis-de%2Fsmall-text)](https://twitter.com/intent/tweet?text=https%3A%2F%2Fgithub.com%2Fwebis-de%2Fsmall-text)

<p align="center">
<img width="372" height="80" src="https://raw.githubusercontent.com/webis-de/small-text/master/docs/_static/small-text-logo.png" alt="small-text logo" />
</p>

> Active Learning for Text Classification in Python.
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

- **Version 1.3.3** ([v1.3.3][changelog_1.3.3]) - December 29th, 2023
  - Bugfix release.

- **Version 1.3.2** ([v1.3.2][changelog_1.3.2]) - August 19th, 2023
  - Bugfix release.

- **Paper accepted at EACL 2023 🎉**
  - The [paper][paper_arxiv] introducing small-text has been accepted at [EACL 2023](https://2023.eacl.org/). Meet us at the conference in May!
  - Update: the paper was awarded [EACL Best System Demonstration](https://aclanthology.org/2023.eacl-demo.11/). Thank you, for your support!

- **Version 1.3.0** ([v1.3.0][changelog_1.3.0]): Highlights - February 20th, 2023
  - Added dropout sampling to [SetFitClassification](https://github.com/webis-de/small-text/blob/v1.3.0/small_text/integrations/transformers/classifiers/setfit.py).
  
- **Version 1.2.0** ([v1.2.0][changelog_1.2.0]): Highlights - February 4th, 2023
  - Make [huggingface/setfit](https://github.com/huggingface/setfit) (SetFit) usable as a small-text classifier.
  - New query strategy: [BALD](https://github.com/webis-de/small-text/blob/v1.2.0/small_text/query_strategies/bayesian.py).
  - Added two new SetFit notebooks, and also updated existing notebooks.

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

| # | Notebook                                                                                                                                                                                                       |                                                                                                                                                                                                                                                  |
| --- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | [Intro: Active Learning for Text Classification with Small-Text](https://github.com/webis-de/small-text/blob/v1.3.3/examples/notebooks/01-active-learning-for-text-classification-with-small-text-intro.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/webis-de/small-text/blob/v1.3.3/examples/notebooks/01-active-learning-for-text-classification-with-small-text-intro.ipynb) |
| 2 | [Using Stopping Criteria for Active Learning](https://github.com/webis-de/small-text/blob/v1.3.3/examples/notebooks/02-active-learning-with-stopping-criteria.ipynb)                                           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/webis-de/small-text/blob/v1.3.3/examples/notebooks/02-active-learning-with-stopping-criteria.ipynb)                        |
| 3 | [Active Learning using SetFit](https://github.com/webis-de/small-text/blob/v1.3.3/examples/notebooks/03-active-learning-with-setfit.ipynb)                                                                     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/webis-de/small-text/blob/v1.3.3/examples/notebooks/03-active-learning-with-setfit.ipynb)                                   |
| 4 | [Using SetFit's Zero Shot Capabilities for Cold Start Initialization](https://github.com/webis-de/small-text/blob/v1.3.3/examples/notebooks/04-zero-shot-cold-start.ipynb)                                     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/webis-de/small-text/blob/v1.3.3/examples/notebooks/04-zero-shot-cold-start.ipynb)                                          |

### Showcase

- [Tutorial: 👂 Active learning for text classification with small-text][argilla_al_tutorial] (Use small-text conveniently from the [argilla][argilla] UI.)

A full list of showcases can be found [in the docs][documentation_showcase].

🎀 **Would you like to share your use case?** Regardless if it is a paper, an experiment, a practical application, a thesis, a dataset, or other, let us know and we will add you to the [showcase section][documentation_showcase] or even here.

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

Small-Text has been introduced in detail in the EACL23 System Demonstration Paper ["Small-Text: Active Learning for Text Classification in Python"](https://aclanthology.org/2023.eacl-demo.11/) which can be cited as follows:
```
@inproceedings{schroeder2023small-text,
    title = "Small-Text: Active Learning for Text Classification in Python",
    author = {Schr{\"o}der, Christopher  and  M{\"u}ller, Lydia  and  Niekler, Andreas  and  Potthast, Martin},
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-demo.11",
    pages = "84--95"
}
```

## License

[MIT License](LICENSE)


[documentation_main]: https://small-text.readthedocs.io/en/v1.3.3/
[documentation_install]: https://small-text.readthedocs.io/en/v1.3.3/install.html
[documentation_query_strategies]: https://small-text.readthedocs.io/en/v1.3.3/components/query_strategies.html
[documentation_showcase]: https://small-text.readthedocs.io/en/v1.3.3/showcase.html
[documentation_reproducibility_notes]: https://small-text.readthedocs.io/en/v1.3.3/reproducibility_notes.html
[changelog]: https://small-text.readthedocs.io/en/latest/changelog.html
[changelog_1.2.0]: https://small-text.readthedocs.io/en/latest/changelog.html#version-1-2-0-2023-02-04
[changelog_1.3.0]: https://small-text.readthedocs.io/en/latest/changelog.html#version-1-3-0-2023-02-21
[changelog_1.3.2]: https://small-text.readthedocs.io/en/latest/changelog.html#version-1-3-2-2023-08-19
[changelog_1.3.3]: https://small-text.readthedocs.io/en/latest/changelog.html#version-1-3-3-2023-12-29
[argilla]: https://github.com/argilla-io/argilla
[argilla_al_tutorial]: https://docs.argilla.io/en/latest/tutorials/notebooks/training-textclassification-smalltext-activelearning.html
[paper_arxiv]: https://arxiv.org/abs/2107.10314
