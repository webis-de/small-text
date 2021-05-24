
<center>
<img src="./docs/_static/small-text-logo.png" alt="small-text logo" />
</center>

> Active Learning for Text Classifcation in Python.
<hr>

[Installation](#installation) | [Quick Start](#quick-start) | [Docs](#docs)

<br>
Active Learning allows you to efficiently label training data in a small-data scenario.

This library provides state-of-the-art **active learning** for text classification, 
built with modularity and extensibility in mind.



## Features

- Provides unified interfaces for Active Learning so that you can easily use any classifier provided by Integrates [sklearn](https://scikit-learn.org/).
- (Optionally) As an optional feature, you can also use [pytorch](https://pytorch.org/) classifiers, including [transformers](https://github.com/huggingface/transformers) models.
- Multiple scientifically-proven strategies re-implemented: Query Strategies, Initialization Strategies

## Installation

```bash
pip install small-text
```

Requires Python 3.7 or newer. For using the GPU, CUDA 10.1 or newer is required.


## Quick Start

For a quick start, see the provided examples for [binary classification](examples/binary_classification.py), 
[pytorch multi-class classification](examples/pytorch_multiclass_classification.py), or 
[transformer-based multi-class classification](examples/transformers_multiclass_classification.py)

## Docs

The API docs (currently work in progress) can be generated using sphinx:

```bash
pip install sphinx sphinx-rtd-theme
cd docs/
make
```

## Alternatives

- [modAL](https://github.com/modAL-python/modAL)
- [ALiPy](https://github.com/NUAA-AL/ALiPy)
- [libact](https://github.com/ntucllab/libact)

## Contribution

Contributions are welcome. Details can be found in [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgments

This software was created by [@chschroeder](https://github.com/chschroeder) at Leipzig University's [NLP group](http://asv.informatik.uni-leipzig.de/) which is a part of the [Webis](https://webis.de/) research network. The encompassing project was funded by the Development Bank of Saxony (SAB) under project number 100335729.

## License

[MIT License](LICENSE)

