User Guide
==========

CBX implements consensus based particle schemes. Below, we list the pages that explain the functionality:

.. toctree::
   :maxdepth: 1

   dynamics
   sched
   numpyvstorch
   objectives
   plotting

Basic Parameters and nomenclature
---------------------------------

We collect some basic parameters and terminology that are used in the following

==============   ==============================================
**symbol**                       **description**          
==============   ==============================================
:math:`x`        The ensemble of points               
:math:`f`        The objective function               
:math:`M`        The number of runs
:math:`N`        The number of particles
:math:`d`        The dimension
:math:`dt`       The time step parameter
:math:`t`        The current time
:math:`\alpha`   The weighting parameter for the consensus point
:math:`\sigma`   The noise scaling parameter
==============   ==============================================
