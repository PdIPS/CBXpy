![cbx](https://github.com/PdIPS/CBXpy/assets/44805883/a5b96135-1039-4303-9cb1-32a1b1393fe3)

[![status](https://joss.theoj.org/papers/008799348e8232eb9fe8180712e2dfb8/status.svg)](https://joss.theoj.org/papers/008799348e8232eb9fe8180712e2dfb8)
![tests](https://github.com/PdIPS/CBXpy/actions/workflows/Tests.yml/badge.svg) 
[![codecov](https://codecov.io/gh/PdIPS/CBXpy/graph/badge.svg?token=TU3LO8SLFP)](https://codecov.io/gh/PdIPS/CBXpy) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Doc](https://img.shields.io/badge/Documentation-latest-blue)](https://pdips.github.io/CBXpy)

A Python package for consensus-based particle dynamics, focusing on **optimization** and **sampling**. 

# How to use CBXPy?

Minimizing a function using CBXPy can be done as follows:

```python
   from cbx.dynamics import CBO        # import the CBO class

   f = lambda x: x[0]**2 + x[1]**2     # define the function to minimize
   x = CBO(f, d=2).optimize()          # run the optimization
```

A documentation together with more examples and usage instructions is available at [https://pdips.github.io/CBXpy](https://pdips.github.io/CBXpy).


# Installation

Currently ```CBXPy``` can only be installed from PyPI with pip.

```bash
   pip install cbx
```



# What is CBX?

Originally designed for optimization problems of the form

$$
   \min_{x \in \mathbb{R}^n} f(x),
$$

the scheme was introduced as CBO (Consensus-Based Optimization). Given an ensemble of points $x = (x_1, \ldots, x_N)$, the update reads

$$
x_i \gets x_i - \lambda\ dt\ (x_i - c(x)) + \sigma\ \sqrt{dt} |x_i - c(x)|\ \xi_i
$$

where $\xi_i$ are i.i.d. standard normal random vectors. The core element is the consensus point

$$
\begin{align*}
c(x) = \left(\sum_{i=1}^{N} x_i\ \exp(-\alpha\ f(x_i))\right)\bigg/\left(\sum_{i=1}^N \exp(-\alpha\ f(x_i))\right)
\end{align*}
$$

with a parameter $\alpha>0$. The scheme can be extended to sampling problems known as CBS, clustering problems and opinion dynamics, which motivates the acronym 
**CBX**, indicating the flexibility of the scheme.

## Functionality

Among others, CBXPy currently implements

* CBO (Consensus-Based Optimization) [[1]](#CBO)
* CBS (Consensus-Based Sampling) [[2]](#CBS)
* CBO with memory [[3]](#CBOMemory)
* Batching schemes [[4]](#Batching)
* Polarized CBO [[5]](#PolarizedCBO)
* Mirror CBO [[6]](#MirrorCBO)
* Adamized CBO [[7]](#AdamizedCBO) 
* Constrained CBO methods, including
  * Drift Correction [[8]](#DriftCorrection)
  * Regularization [[9]](#Regularization)
  * Hypersurface CBO [[10]](#HypersurfaceCBO)


## References

<a name="CBO">[1]</a> A consensus-based model for global optimization and its mean-field limit, Pinnau, R., Totzeck, C., Tse, O. and Martin, S., Mathematical Models and Methods in Applied Sciences 2017

<a name="CBS">[2]</a> Consensus-based sampling, Carrillo, J.A., Hoffmann, F., Stuart, A.M., and Vaes, U., Studies in Applied Mathematics 2022

<a name="CBOMemory">[3]</a> Leveraging Memory Effects and Gradient Information in Consensus-Based Optimization: On Global Convergence in Mean-Field Law, Riedl, K., 2022

<a name="Batching">[4]</a> A consensus-based global optimization method for high dimensional machine learning problems, Carrillo, J.A., Jin, S., Li, L. and Zhu, Y., ESAIM: Control, Optimisation and Calculus of Variations 2021

<a name="PolarizedCBO">[5]</a> Bungert, L., Roith, T., & Wacker, P. (2024). Polarized consensus-based dynamics for optimization and sampling. Mathematical Programming, 1-31.

<a name="MirrorCBO">[6]</a> Bungert, L., Hoffmann, F., Kim, D. Y., & Roith, T. (2025). MirrorCBO: A consensus-based optimization method in the spirit of mirror descent. arXiv preprint arXiv:2501.12189.

<a name="AdamizedCBO">[7]</a> Chen, J., Jin, S., & Lyu, L. (2020). A consensus-based global optimization method with adaptive momentum estimation. arXiv preprint arXiv:2012.04827.

<a name="DriftCorrection">[8]</a> Carrillo, J. A., Jin, S., Zhang, H., & Zhu, Y. (2024). An interacting particle consensus method for constrained global optimization. arXiv preprint arXiv:2405.00891.

<a name="Regularization">[9]</a> Borghi, G., Herty, M., & Pareschi, L. (2023). Constrained consensus-based optimization. SIAM Journal on Optimization, 33(1), 211-236.

<a name="HypersurfaceCBO">[10]</a> Fornasier, M., Huang, H., Pareschi, L., & Sünnen, P. (2020). Consensus-based optimization on hypersurfaces: Well-posedness and mean-field limit. Mathematical Models and Methods in Applied Sciences, 30(14), 2725-2751.
