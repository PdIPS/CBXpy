dynamics
========

This module implements algorithms for optimization and sampling for consensus based
particles systems as proposed in [1]_. 

The base class ``ParticleDynamic`` implements functionality that is common 
to particle based iterative methods. The class ``CBXDynmaic`` inherits from
``ParticleDynamic`` and implements functionality that is specific to consensus 
based schemes. The following dynamics are implemented:

.. currentmodule:: cbx.dynamics


.. autosummary::
   :toctree: generated
   :nosignatures:
   :recursive:
   :template: classtemplate.rst

   ParticleDynamic
   CBXDynamic
   CBO
   CBOMemory
   CBS
   PSO
   PolarCBO




References
----------

.. [1] Pinnau, R., Totzeck, C., Tse, O., & Martin, S. (2017). A consensus-based model for global optimization and its mean-field limit. Mathematical Models and Methods in Applied Sciences, 27(01), 183-204.
