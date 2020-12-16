# lightRaven -- Offline RL with Maximum Speed
[![PyPI version lightRaven](https://badge.fury.io/py/lightRaven.svg)](https://pypi.org/project/lightRaven/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/lightRaven.svg)](https://pypi.org/project/lightRaven/)
![Python package](https://github.com/M0gician/lightRaven/workflows/Python%20package/badge.svg)

This library provides convenient tools for people to create their own seldonian algorithms with optimum performance. A detailed example is also included in `dynamic_training.ipynb`. Performance test is in `ci_performance.ipynb`.

## Dependencies  [![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
- `gym==0.17.3`
- `numpy==1.19.1`
- `scipy==1.5.2`
- `numba == 0.51.2`

## Supplementary Materials
- Definition of Seldonian Framework
  - [Preventing undesirable behavior of intelligent machines](https://science.sciencemag.org/content/366/6468/999)
  - [High Confidence Policy Improvement](https://people.cs.umass.edu/~pthomas/papers/Thomas2015.pdf)
- Definition of different Importance Sampling estimators
  - [High Confidence Off-Policy Evaluation](https://people.cs.umass.edu/~pthomas/papers/Thomas2015.pdf)
- Definition of the new concentration bound 
  - [A New Confidence Interval for the Mean of a Bounded Random Variable](https://arxiv.org/abs/1905.06208)