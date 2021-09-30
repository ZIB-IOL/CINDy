# CINDy: Conditional gradient-based Identification of Non-linear Dynamics – Noise-robust recovery

This reproduces the experiments of the [CINDy: Conditional gradient-based Identification of Non-linear Dynamics – Noise-robust recovery](https://arxiv.org/pdf/2101.02630.pdf) paper.


## Implemented Algorithms

Most of the algorithms included in the package solve a [Least Absolute Shrinkage and Selection Operator](http://www-stat.stanford.edu/~tibs/lasso/lasso.pdf) (LASSO) formulation of the sparse recovery problem, where the `l-1` norm regularization happens either in the feasible region (CINDy, IPM) or in the objective function (SR3, FISTA). In the case of the SR3 algorithm, one can substitute the `l-1` norm regularization for `l-0` norm regularization (although the former is not technically a norm). The CINDy, IPM and SR3 algorithms can also add a series of arbitrary linear constraints on the problem, either through the feasible region (CINDy, IPM), or through the objective function (SR3).

### CINDy

Implementation of the [Blended Conditional Gradients](https://arxiv.org/abs/1805.07311) (BCG) algorithm to solve a least squares problem subject to an `l-1` norm feasible region constraint, and a series of additional linear constraints.

### SINDy

Implementation of the Sequentially-Thresholded Ridge Regression formulation in the [Sparse Identification of Non-linear Dynamics](https://www.pnas.org/content/113/15/3932) (SINDy) framework. Based on the code in the [PDE_FIND](https://github.com/snagcliffs/PDE-FIND) Github repository.

### SR3

Implementation of the [Sparse Relaxed Regularized Regression](https://arxiv.org/abs/1906.10612) (SR3) algorithm. Based on the code in the [SINDySR3](https://github.com/kpchamp/SINDySR3) Github repository, with the correction of aspects in the mathematical formulation. This algorithm solves a least-squares problem with `l-1` or `l-0` norm regularization. Additional linear constraints are also enforced through penalty terms in the objective function.

### FISTA

Implementation of the [Fast Iterative Shrinkage-Thresholding Algorithm](https://www.ceremade.dauphine.fr/~carlier/FISTA) (FISTA) algorithm. This algorithm is used to solve a least squares problem with `l-1` norm regularization (in the objective function). 

### IPM

We include a least-squares problem formulation with an `l-1` norm feasible region constraint, and a series of additional linear constraints, that is solved with the [Interior-Point Method](https://people.compute.dtu.dk/~mskan/publications/mlbook.pdf) (IPM) included in the [CVXOPT](https://cvxopt.org/) Python package.

## Dynamics to recover

We benchmark the above algorithms on three dynamics, namely:

### Kuramoto model

ODE model that describes the angular movement of a series of weakly coupled identical oscillators that differ in their angular frequency. We consider the case where there is external forcing in the system see  [this paper](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.77.137) for the details on the mathematical formulation). The true underlying dynamic can be described using a combination of sines and cosines dependent on the angular position of the particles. In the experiment we consider a system with 5, and with 10 oscillators.

### Fermi-Pasta-Ulam-Tsingou model

This physical model describes a system of one dimensional particles connected through springs and subject to a nonlinear forcing term. The mathematical description of the system can be found in [this technical report](https://www.osti.gov/servlets/purl/4376203). We consider two cases, one in which the there are a total of 5 particles, and one in which there are a total of 10 particles.

### Michaelis-Menten model

The last model we benchmark our algorithm on describes enzyme reaction kynetics, in which several chemical species are formed at a speed proportional to the quantity of the different species. There are a total of 4 species in this example, and we use the mathematical formulation described in [this paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1259181/).

## Citing

When using the SINDy, SR3, FISTA and IPM algorithms please cite the appropiate papers. When using CINDy, please use the CITATION.bib BibTeX entry in the github repository.
