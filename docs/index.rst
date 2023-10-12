
CBXPy: Consensus Based Particle Dynamics in Python
================================================================


CBXPy is a python package implementing consensus based particle schemes. Originally designed for optimization problems

.. math::

   \min_{x \in \mathbb{R}^n} f(x),


the scheme was introduced as CBO (Consensus Based Optimization) in [1]_. Given an ensemble of points :math:`x = (x_1, \ldots, x_N)`, the update reads

.. math::

   x_i \gets x_i - \lambda\, dt\, (x_i - c(x)) + \sigma\, \sqrt{dt} |x_i - c(x)| \xi_i

where :math:`\xi_i` are i.i.d. standard normal random variables. The core element is the consensus point

.. math::

   c(x) = \frac{\sum_{i=1}^N x_i\, \exp(-\alpha\, f(x_i))}{\sum_{i=1}^N \exp(-\alpha\, f(x_i))}.

with a parameter :math:`\alpha>0`. The scheme can be extended to sampling problems [2]_ known as CBS, clustering problems and opinion dynamics [3]_, which motivates the acronym 
**CBX**, indicating the flexibility of the scheme.

Installation
------------
The package can be installed via ``pip``

.. code-block:: bash

   pip install cbx

Simple Usage Example
--------------------

The following example shows how to minimize a function using the CBO scheme

.. code-block:: python
   from cbx.dynamics import CBO

   f = lambda x: x.sum(axis=-1)
   dyn = CBO(f, d=7)
   dyn.solve()



Documentation
----------

.. toctree::
   :maxdepth: 2

   userguide/index


Usage
-----

The most important modules are documented below.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   dynamic
   objectives
   scheduler
   utils


.. button-link:: userguide
    :color: primary
    :outline:




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
