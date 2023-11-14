
CBXPy: Consensus Based Particle Dynamics in Python
================================================================


CBXPy is a python package implementing consensus based particle schemes. Originally designed for optimization problems

.. math::

   \min_{x \in \mathbb{R}^n} f(x),


the scheme was introduced as CBO (Consensus Based Optimization) in [1]_. Given an ensemble of points :math:`x = (x_1, \ldots, x_N)`, the update reads

.. math::

   x^i \gets x^i - \lambda\, dt\, (x_i - c_\alpha(x)) + \sigma\, \sqrt{dt} |x^i - c_\alpha(x)| \xi^i

where :math:`\xi_i` are i.i.d. standard normal random variables. The core element is the consensus point

.. math::

   c_\alpha(x) = \frac{\sum_{i=1}^N x^i\, \exp(-\alpha\, f(x^i))}{\sum_{i=1}^N \exp(-\alpha\, f(x^i))}.

with a parameter :math:`\alpha>0`. The scheme can be extended to sampling problems [2]_ known as CBS, clustering problems and opinion dynamics [3]_, which motivates the acronym 
**CBX**, indicating the flexibility of the scheme.

Installation
------------
The package can be installed via ``pip``

.. code-block:: bash

   pip install cbx

Simple Usage Example
--------------------

The following example shows how to minimize a function using CBXPy

::

   from cbx.dynamics import CBO

   f = lambda x: x[0]**2 + x[1]**2
   dyn = CBO(f, d=2)
   x = dyn.optimize()



Documentation
-------------

.. toctree::
   :maxdepth: 2

   userguide/index
   api/index


References
----------

.. [1] Pinnau, R., Totzeck, C., Tse, O., & Martin, S. (2017). A consensus-based model for global optimization and its mean-field limit. Mathematical Models and Methods in Applied Sciences, 27(01), 183-204.
.. [2] ??
.. [3] ??

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
