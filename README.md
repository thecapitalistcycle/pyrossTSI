# PyRossTSI: Time since infection models in Python  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rajeshrinet/pyrosstsi/master?filepath=examples)

![Installation](https://github.com/rajeshrinet/pyrosstsi/workflows/Installation/badge.svg) 
[![Documentation Status](https://readthedocs.org/projects/pyrosstsi/badge/?version=latest)](https://pyrosstsi.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/pyrosstsi.svg)](https://badge.fury.io/py/pyrosstsi) 
[![Downloads](https://pepy.tech/badge/pyrosstsi)](https://pepy.tech/project/pyrosstsi)
![stars](https://img.shields.io/github/stars/rajeshrinet/pyrosstsi) 
![forks](https://img.shields.io/github/forks/rajeshrinet/pyrosstsi) 
 




## About


* [PyRoss](https://github.com/rajeshrinet/pyross) is a companion library that offers tools for both deterministic and stochastic simulation of macroscopic compartmental models, as well as a complete
suite of inference and optimal control tools.
* [PyRossGeo](https://github.com/lukastk/PyRossGeo) is another companion library that supports spatially resolved compartment models with explicit commuting networks.


## Installation
You can take PyRossTSI for a spin **without installation**: [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/rajeshrinet/pyrosstsi/master?filepath=examples). Please be patient while [Binder](https://mybinder.org/v2/gh/rajeshrinet/pyrosstsi/master?filepath=examples) loads.

PyRossTSI requires the following software

- Python 2.6+ or Python 3.4+
- [Cython](http://docs.cython.org/en/latest/index.html) |  [Matplotlib](https://matplotlib.org) | [NumPy](http://www.numpy.org) |   [SciPy](https://www.scipy.org/)  | [cma](https://github.com/CMA-ES/pycma) |  [NLopt](https://anaconda.org/conda-forge/nlopt)
- Optional:  integrators ([OdesPy](https://github.com/rajeshrinet/odespy)) |  parallelizing inference ([pathos](https://anaconda.org/conda-forge/pathos)) |  data files ([Pandas](https://pandas.pydata.org/), [xlrd](https://xlrd.readthedocs.io/en/latest/))


Clone (or download) the repository and use a terminal to install PyRoss (above [requirements](https://github.com/rajeshrinet/pyrosstsi/blob/master/requirements.txt) need to be satisfied)
```bash
>> git clone https://github.com/rajeshrinet/pyrosstsi.git
>> cd pyrosstsi
>> python setup.py install
```


Alternatively, use [anaconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/macos.html) to install PyRossTSI and its dependencies in a pyrosstsi [environment](https://github.com/rajeshrinet/pyrosstsi/blob/master/environment.yml)
```bash
>> git clone https://github.com/rajeshrinet/pyrosstsi.git
>> cd pyrosstsi
>> make env
>> conda activate pyrosstsi
>> make
```
#### Pip

```
pip install -e git+https://github.com/rajeshrinet/pyrosstsi.git#egg=pyrosstsi
```
