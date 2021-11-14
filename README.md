[![PyPI](https://img.shields.io/pypi/v/small-text)](https://pypi.org/project/small-text/)
[![codecov](https://codecov.io/gh/webis-de/small-text/branch/master/graph/badge.svg?token=P86CPABQOL)](https://codecov.io/gh/webis-de/small-text)
[![Documentation Status](https://readthedocs.org/projects/small-text/badge/?version=latest)](https://small-text.readthedocs.io/en/latest/?badge=latest) 
![GitHub](https://img.shields.io/github/license/webis-de/small-text)


<p align="center">
<img width="372" height="80" src="https://raw.githubusercontent.com/webis-de/small-text/master/docs/_static/small-text-logo.png" alt="small-text logo" />
</p>

> Active Learning for Text Classifcation in Python.
<hr>

[Installation](#installation) | [Quick Start](#quick-start) | [Docs](https://small-text.readthedocs.io/en/latest/)

Small-Text provides state-of-the-art **Active Learning** for Text Classification. 
This allows you to easily mix and match many classifiers and query strategies 
to build active learning experiments or applications.

**What is Active Learning?**

[Active Learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) allows you to efficiently label training data in a small-data scenario.


## Features

- Provides unified interfaces for Active Learning so that you can easily use any classifier provided by [sklearn](https://scikit-learn.org/).
- (Optionally) As an optional feature, you can also use [pytorch](https://pytorch.org/) classifiers, including [transformer](https://github.com/huggingface/transformers) models.
- Multiple scientifically-proven strategies re-implemented: Query Strategies, Initialization Strategies

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
[documentation](https://small-text.readthedocs.io/en/latest/install.html).


## Quick Start

For a quick start, see the provided examples for [binary classification](examples/code/binary_classification.py), 
[pytorch multi-class classification](examples/code/pytorch_multiclass_classification.py), and 
[transformer-based multi-class classification](examples/code/transformers_multiclass_classification.py), 
or check out the notebooks.

### Notebooks

| # | Notebook | |
| --- | -------- | --- |
| 1 | [Intro: Active Learning for Text Classification with Small-Text](examples/notebooks/01-active-learning-for-text-classification-with-small-text-intro.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/webis-de/small-text/blob/master/examples/notebooks/01-active-learning-for-text-classification-with-small-text-intro.ipynb) |

## Documentation

Read the latest documentation (currently work in progress) [here](https://small-text.readthedocs.io/en/latest/).

## Alternatives

- [modAL](https://github.com/modAL-python/modAL)
- [ALiPy](https://github.com/NUAA-AL/ALiPy)
- [libact](https://github.com/ntucllab/libact)

## Contribution

Contributions are welcome. Details can be found in [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgments

This software was created by [@chschroeder](https://github.com/chschroeder) at Leipzig University's [NLP group](http://asv.informatik.uni-leipzig.de/) which is a part of the [Webis](https://webis.de/) research network. The encompassing project was funded by the Development Bank of Saxony (SAB) under project number 100335729.

## Citation

A preprint which introduces small-text is available here:  
[Small-text: Active Learning for Text Classification in Python](https://arxiv.org/abs/2107.10314). 

```
@misc{schroeder2021smalltext,
    title={Small-text: Active Learning for Text Classification in Python}, 
    author={Christopher Schröder and Lydia Müller and Andreas Niekler and Martin Potthast},
    year={2021},
    eprint={2107.10314},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## License

[MIT License](LICENSE)
