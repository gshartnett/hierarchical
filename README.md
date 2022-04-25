# The Hierarchical Parity Model
Code repository for the paper "The Hierarchical Parity Model: A Computationally Tractable Spin-Glass"

## Introduction
This repository contains the code used for the numerical analyses of the hierarchical parity model reported on in the paper.

## Contents
The key contents of this repository are:
- `hierarchical parity model.nb` a Mathematica notebook which contains some useful calculations.
- `hierarchical parity model.ipynb` A Python Jupyter notebook which contains the Jax automatic differentiation (autodiff) code used to perform the analysis of the disordered model (Section 5 of the paper).
- `utils.py` a Python script containing useful helper functions.

## Installation
The Mathematica notebook `hierarchical parity model.nb` requires no installation beyond the fact that Mathematica must already be installed. The Jupyter notebook `hierarchical parity model.ipynb` requires that [Google Jax](https://github.com/google/jax) be installed (the installation instructions are system-dependent and may be found on the Jax website). The other necessary packages may be installed by running `pip install -r requirements.txt`. It is recommended to perform the Python installation within a virtual environment.

## License
This code is provided under the MIT license. See `LICENSE` for more information.

## Contact
Developed by [Gavin Hartnett](https://www.rand.org/about/people/h/hartnett_gavin_s.html) (email: hartnett@rand.org).
