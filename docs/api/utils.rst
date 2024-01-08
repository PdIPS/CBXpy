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
   
   termination.Termination
   termination.check_energy
   termination.check_max_it
   termination.check_max_time
   termination.check_max_eval
   termination.check_diff_tol

Resampling schemes
------------------

.. currentmodule:: cbx.utils

.. autosummary::
   :toctree: generated
   :nosignatures:
   :recursive:
   :template: classtemplate.rst

   resampling.apply_resamplings
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
