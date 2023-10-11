dynamic
==========

This module implements algorithms for optimization and sampling for consensus based
particles systems. All the dynamics inherit from the base class ``ParticleDynamic``.

.. currentmodule:: cbx.dynamics

.. autoclass:: ParticleDynamic
   :members:
   :undoc-members:
   :show-inheritance:




Standard Consensus Based Schemes
--------------------------------

The following classes implement standard consensus based schemes [1]_.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :recursive:
   :template: classtemplate.rst

   CBO




References
----------

.. [1] Pinnau, R., Totzeck, C., Tse, O., & Martin, S. (2017). A consensus-based model for global optimization and its mean-field limit. Mathematical Models and Methods in Applied Sciences, 27(01), 183-204.
