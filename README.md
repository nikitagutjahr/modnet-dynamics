# Griffiths phase width control in modular networks

This code is part of an article in progress by Nikita Gutjahr and Aline Viol.

## Synopsis

This code contains python implementations of the SIS epidemic spreading model and the contact process [1], as well as quasistationary [2] versions of both. Network generation procedures for monodisperse modular networks [3] and hierarchical modular networks [4] are given in networks.py.

[1] Cota, Wesley, and Silvio C. Ferreira. “Optimized Gillespie Algorithms for the Simulation of Markovian Epidemic Processes on Large and Heterogeneous Networks.” Computer Physics Communications 219 (2017): 303–312 [[ArXiv](https://arxiv.org/abs/1704.01557)]

[2] Martins de Oliveira, Marcelo and Dickman, Ronald "How to simulate the quasistationary state" Phys. Rev. E 71 (1) (2005) [[ArXiv](https://arxiv.org/abs/cond-mat/0407797)]

[3] Cota, Wesley, Géza Ódor, and Silvio C. Ferreira. “Griffiths Phases in Infinite-Dimensional, Non-Hierarchical Modular Networks.” Scientific Reports 8.1 (2018) [[ArXiv](https://arxiv.org/abs/1801.06406)]

[4] Moretti, Paolo, and Miguel A. Muñoz. “Griffiths Phases and the Stretching of Criticality in Brain Networks.” Nature Communications 4.1 (2013) [[ArXiv](https://arxiv.org/abs/1308.6661)]

## Installation

Requires the NumPy, SciPy, Cython and graph-tool libraries.
The dynamics.pyx module needs to be compiled due to optimization with Cython.
For that, run ```python setup.py build_ext --inplace```, after which dynamics can be imported as usual.

## Use

The usage is showcased in the jupyter notebook example.ipynb.
