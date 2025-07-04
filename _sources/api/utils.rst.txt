.. _utils:
utils
==========

This module implements some helpful utilities.

Termination criteria
---------------------

.. currentmodule:: cbx.utils

.. autosummary::
   :toctree: generated
   :nosignatures:
   :recursive:
   :template: classtemplate.rst
   
   termination.energy_tol_term
   termination.diff_tol_term
   termination.max_eval_term
   termination.max_it_term
   termination.max_time_term

Resampling schemes
------------------

.. currentmodule:: cbx.utils

.. autosummary::
   :toctree: generated
   :nosignatures:
   :recursive:
   :template: classtemplate.rst

   resampling.resampling
   resampling.ensemble_update_resampling
   resampling.loss_update_resampling



History
-------

.. currentmodule:: cbx.utils

.. autosummary::
   :toctree: generated
   :nosignatures:
   :recursive:
   :template: classtemplate.rst

   history.track
   history.track_x
   history.track_energy
   history.track_update_norm
   history.track_consensus
   history.track_drift
   history.track_drift_mean


Particle initialization
------------------------

.. currentmodule:: cbx.utils

.. autosummary::
   :toctree: generated
   :nosignatures:
   :recursive:
   :template: classtemplate.rst

   particle_init.init_particles

Objective Handling
------------------

.. currentmodule:: cbx.utils

.. autosummary::
   :toctree: generated
   :nosignatures:
   :recursive:
   :template: classtemplate.rst

   objective_handling.cbx_objective
   objective_handling.cbx_objective_fh
   objective_handling.cbx_objective_f1D
   objective_handling.cbx_objective_f2D
