# Contributing to CBXPy

Hey, great that you found your way here and want to contribute to CBXPy. And thanks a lot that you are going through these guidelines! We are always happy for your suggestions or improvements, e.g.:

* bug reports,
* bug fixes,
* feature developments,
* documentation.

## How to contribute

The best way to contribute to the repository is to create a fork. After committing your desired changes, you can create a pull request. Please make sure, that the tests are passing before submitting. The unit tests employ ``pytest`` and are implemented as a [GitHub workflow](https://github.com/PdIPS/CBXpy/blob/main/.github/workflows/Tests.yml). Therefore, the tests can be run as a [GitHub Action](https://docs.github.com/de/actions).

## Reporting a Bug

Although, the nature of CBXPy does not directly indicate possible security issues: If you find a security vulnerability, do **NOT** open an issue. Email tim.roith@desy.de instead.

Other than that, please open an issue [here](https://github.com/PdIPS/CBXpy/issues). The bug report template gives more details on how to open an issue for a bug report.

## Adding a feature

Since CBX aims to capture the growing field of consensus-based particle methods in one package, additions and implementations of new algorithm variants are very welcome. A good starting point to understand the mechanisms of CBXPy is the [documentation](https://pdips.github.io/CBXpy/). The most important aspects of the CBXPy implementation is explained there. If anything is still unclear, you can take a look at the [discussion forum](https://github.com/orgs/PdIPS/discussions).

Apart from that, we list the following aspects of adding new features and algorithms:

* **Is the feature novel?** For example, if your proposed algorithm is a special case of an existing implementation, it is preferred to employ the existing function with special parameters, instead of writing new code.

* **Are multirun-ensembles supported?** As explained in the documentation [here](https://pdips.github.io/CBXpy/userguide/dynamics.html) ensemble arrays are always of the shape $(M\times N\times d_1, \ldots, d_s)$, where $M$ denotes the number of runs and $N$ denotes the number of particles. Your feature must be able to deal with this ensemble structure.

* **Does your code avoid for-loops**? While it is very intuitive, to write operations over runs and particles as for-loops, this usually leads to performance bottlenecks due to the for-loop in Python. CBXPy does **not** aim for ultimate high-performance, but reasonable optimization with the tools provided by ``numpy`` is expected. In practice, this means, we always try to avoid for-loops by using array operations in ``numpy``.

## Code Review

Code reviews are currently only conducted by [TimRoith](https://github.com/TimRoith). A first reply on a new pull request can be expected within a week. You can also write a mail to tim.roith@desy.de if you feel that your request got lost.
