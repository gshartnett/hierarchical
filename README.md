# The Hierarchical Parity Model
Code repository for the paper "The Hierarchical Parity Model: A Computationally Tractable Spin-Glass"

## Introduction
This repository contains the code used for the numerical analyses of the hierarchical parity model reported on in the paper.

## Contents
The key contents of this repository are:
- `emp.jl`: a Julia script containing the functions used to implement the EMP model
- `EMP Notebook.ipynb` a Jupyter notebook which walks through the math and implementation of the EMP model
- `EMP Calculations.nb` a Mathematica notebook which contains some useful calculations (relating to Compton scattering and the problem geometry).
- `Plotting Notebook.ipynb` a Jupyter notebook used for making Python plots of the variation of the EMP peak magnitude over a region on the surface of the Earth.
- `Seiler Digitized Data` a directory containing digitized data from select figures in the original Seiler report. The data was digitized using [this online tool](https://apps.automeris.io/wpd/).

## Installation
To clone the repository and install the required Julia and Python packages, follow these instructions. It is recommended to perform the Python installation within an Anaconda virtual environment, but this is not necessary.

Clone the repo:
```
git clone git@github.com:gshartnett/karzas-latter-emp.git
cd karzas-latter-emp
```

Create a python virtual environment and create a corresponding Jupyter kernel (optional). The Jupyter kernel is required to run the `Plotting Notebook.ipynb` file.
```
conda create --name karzas-latter-emp python=3.8
conda activate karzas-latter-emp
pip intall ipykernel
python -m ipykernel install --user --name karzas-latter-emp --display-name "Python (karzas-latter-emp)"
```

Install the Python packages
```
pip install -r requirements.txt
```

Install the Julia packages (enter this command in the Pkg manager, accessible from the Julia REPL by entering `]`)
```
add IJulia DelimitedFiles DifferentialEquations JSON LaTeXStrings Plots Printf QuadGK
```

Add the Julia kernel to Jupyter (this is necessary to run the `EMP Notebook.ipynb` file). Enter the Julia REPL and enter
```
using IJulia
installkernel("Julia")
```

## License
This code is Copyright (C) 2022 RAND Corporation, and provided under the MIT license. See `LICENSE` for more information.

## Contact
Developed by [Gavin Hartnett](https://www.rand.org/about/people/h/hartnett_gavin_s.html) (email: hartnett@rand.org).
