Dynamics
========

One of the core components in the CBX package are dynamics which are used to represent different variants of CBO algorithms. Each dynamic inherits from the base class :func:`CBXDynamic <cbx.dynamics.CBXDynamic>` which implements some basic functionality, that is common to all dynamics. This base class itself inherits from :class:`ParticleDynamic <cbx.dynamics.ParticleDynamic>` which implements functionality that is specific to particle based algorithms. The design choice here, was to divide between principles that are common to iterative particle based algorithms and principles that are specific to consensus based algorithms.


Modelling the ensemble
-------------------------

All particle based methods work with an ensemble of points :math:`x = (x^1, \ldots, x^N)\in \mathcal{X}^N`. Here, we choose to model the ensemble as an array, e.g., we assume that :math:`\mathcal{X} = \mathbb{R}^d` and therfore, we can represent as an :math:`N\times d` array. In most cases we assume that the array is given as a ``numpy`` array. However, often it straightforward to instead use ``torch`` tensors. See also :ref:`npvstorch`.

In many cases one wants to perform more than one run of a certain particle based scheme. In this case, a straightforward approach is to creat :math:`M\in \mathbb{N}` instances of a dynamic and run them. This could also be parallelized on the external level. However, in many cases, e.g., if all parameters stay fixed across runs, it can be efficient to directly represent the different runs on the array level. We therfore model an ensemeble as an

.. math::
    M\times N\times d

array, which allows us to employ parallelization on the array level. The value of :math:`M` defaults to 1, however it is important to always keep in mind that an ensemble has three dimensions. 
The ensemble can be accessed via the attribute ``dyn.x`` of each dynamic.

    >>> from cbx.dynamics import ParticleDynamic
    >>> dyn = ParticleDynamic(lambda x:x, d=1, N=5)
    >>> print(dyn.x.shape)
    (1, 5, 1)

If the initial ensemble is not specified, then it is initialized randomly uniform within the bounds given by ``dyn.x_min`` and ``dyn.x_max``. In this case the dimension of the problem :math:`d` can *not* be inferred and therefore have to specified:

    >>> from cbx.dynamics import ParticleDynamic
    >>> dyn = ParticleDynamic(lambda x:x)
    RuntimeError: If the inital partical system is not given, the dimension d must be specified!

However, one can specify the initial ensemble directly, in which case the dimension :math:`d` can be inferred from the shape of the array:	

    >>> import numpy as np
    >>> from cbx.dynamics import ParticleDynamic
    >>> dyn = ParticleDynamic(lambda x:x.sum(-1), x=np.ones((2,5,1)))
    >>> print(dyn.x.shape)
    (2, 5, 1)


The objective function
----------------------

A key element of each particle dynamic is the objective function :math:`f(x)`. This function has to be specified by the user. A priori one assumes that it is a map :math:`f: \mathbb{R}^d \to \mathbb{R}`. However, in many cases we need to evaluate the objective on the whole ensemble. The naive approach here, would be to loop over all indices :math:`m=1, \ldots, M, n=1, \ldots, N` and evaluate :math:`f(x^{m,n})` separately. However, this is not efficient and since the objective evaluation might happen a lot, it is better to evaluate the objective on the whole array at once. Therefore, we need to ensure that objective function ``dyn.f`` can be evaluated on an array of shape :math:`M\times N\times d` and we always think of maps

.. math::
    \mathbb{R}^{M\times N\times d} \to \mathbb{R}^{M\times N}.

I.e., in terms of dimensionality an application of ``dyn.f`` strips away the last dimension (which is the dimension of the original problem :math:`\mathcal{X}=\mathbb{R}^d`) and keeps the structure given by :math:`M\times N`.

However, there might be cases where the user specifies an objective function, which that only works within the original interpretation, i.e., :math:`f: \mathbb{R}^d \to \mathbb{R}^d`, as in the following example:

    >>> import numpy as np
    >>> def f(x):
    >>>     return abs(x[0] + x[1])
    >>> x = np.ones((3,4,2))
    >>> print(f(x).shape)
    (4, 2)

In the above example the array ``x`` yields :math:`M=3, N=4` and :math:`d=2`, therefore the output must of shape :math:`3\times 4`. However, since ``f`` as defined above only works on the single particle level, the shape of the output and therefore also the application is wrong. Let's see how the situation changes when we use the above ``f`` as an objective for a dynamic:
    
    >>> import numpy as np
    >>> from cbx.dynamics import ParticleDynamic
    >>> def f(x):
    >>>     return abs(x[0] + x[1])
    >>>
    >>> dyn = ParticleDynamic(f, x=np.ones((3,4,2)))
    >>> print(dyn.f(x).shape)
    (3, 4)

We observe that the objective function ``dyn.f`` now returns an array of shape :math:`M\times N`. This is due to the fact that an objective is promoted to the class :func:`cbx_objective <cbx.objectives.Objective>`, which handles the evaluation on the array level. By default it is assumed that the specified function, only works on the single particle level, which is expressed in the keyword argument ``f_dim=1`` of the class :class:`ParticleDynamic <cbx.dynamics.ParticleDynamic>`. If your function works on single-run ensembles of shape :math:`N\times d`, you can specify ``f_dim=2`` and respectively if it works on multi-run ensembles of shape :math:`M\times N\times d` you can specify ``f_dim=3``. 

Alternatively, one can directly specifiy the objective function as a :func:`cbx_objective <cbx.objectives.Objective>` by using the following decorator:

    >>> import numpy as np
    >>> from cbx.dynamics import ParticleDynamic
    >>> from cbx.utils.objective_handling import cbx_objective_fh
    >>> 
    >>> @cbx_objective_fh
    >>> def f(x):
    >>>     return abs(x[...,0] + x[...,1])
    >>>
    >>> dyn = ParticleDynamic(f, x=np.ones((2,5,2)))
    >>>
    >>> print(dyn.f(np.ones((3,4,2))).shape)
    >>> print(dyn.f is f)
    (3, 4)
    True

Here, we observe that the dynamic directly uses the specified objective function. For more complicated functions, one can also inherit from :class:`cbx_objective <cbx.objectives.Objective>`.

.. note::
    When inherinting from :class:`cbx_objective <cbx.objectives.Objective>`, the method :meth:`__call__ <cbx.objectives.Objective.__call__>` should not be overwritten as it is used internally to update the number of evaluation. Instead, the actual function function call should be implemented in the method ``apply(self, x)``.

    >>> import numpy as np
    >>> from cbx.dynamics import ParticleDynamic
    >>> from cbx.utils.objective_handling import cbx_objective
    >>> class objective(cbx_objective):
    >>>     def __init__(self, a=1.0):
    >>>         super().__init__()
    >>>         self.a = a
    >>>     def apply(self, x):
    >>>         return self.a * x[...,0] + x[...,1]
    >>> 
    >>> f = objective(a=2.)        
    >>> dyn = ParticleDynamic(f, x=np.ones((2,5,2)))
    >>> print(dyn.f is f)
    True

The step method
----------------

At the heart of every iterative method is the actual update that is performed. Each dynamic encodes this update in the method :meth:`inner_step <cbx.dynamics.CBXDynamic.step>`. For example, the standard CBO class :func:`CBO <cbx.dynamics.CBO>` implements the following update:

.. code-block:: python

    def inner_step(self,) -> None:
        # update, consensus point, drift and energy
        self.consensus, energy = self.compute_consensus(self.x[self.consensus_idx])
        self.drift = self.x[self.particle_idx] - self.consensus
        self.energy[self.consensus_idx] = energy
        
        # compute noise
        self.s = self.sigma * self.noise()

        # update particle positions
        self.x[self.particle_idx] = (
            self.x[self.particle_idx] -
            self.correction(self.lamda * self.dt * self.drift) +
            self.s)

In the simplest case, where we use isotropic noise and no correction, this basically implements the update

.. math::

   x^i \gets x^i - \lambda\, dt\, (x_i - c_\alpha(x)) + \sigma\, \sqrt{dt} |x^i - c_\alpha(x)| \xi^i


with an additional correction step on the drift. If you want to implement a custom update, you need to overwrite this method in an inherited class. Additionally, there might be certain procedures that should happen before or after each iteration. These can be implemented in the method :meth:`pre_step <cbx.dynamics.CBXDynamic.step>` and :meth:`post_step <cbx.dynamics.CBXDynamic.step>`. For example the base dynamic class :class:`CBO <cbx.dynamics.CBXDynmaic>`, saves the position of the old ensemble before each iteration:

.. code-block:: python

    def pre_step(self,) -> None:
        self.x_old = self.copy_particles(self.x)

After each inner step, the base class updates the best particles (both of the current ensemble and the best of the whole iteration), performs the tracking step (see :ref:`tracking`), performs an optional post processing step (e.g., clip the particles within a valid range) and most importantly, increments the iteration counter: 

.. code-block:: python

    def post_step(self) -> None:
        if hasattr(self, 'x_old'):
            self.update_diff = np.linalg.norm(self.x - self.x_old, axis=(-2,-1))/self.N
        
        self.update_best_cur_particle()
        self.update_best_particle()
        self.track()
        self.process_particles()
            
        self.it+=1

The main step method, which actually used in the iteration is the defined as

.. code-block:: python
    def step(self):
        self.pre_step()
        self.inner_step()
        self.post_step()


Noise methods and how to customize them
---------------------------------------

In the update step of consensus based methods, diffusion is modeled by the addition of noise, which is scaled by a factor dependent on the iteration. Here, it is very convenient to assume that we can compute the noise, given full information about the dynamic. Therefore, we choose to implement it as method of the dynamic class. The base class :func:`CBO <cbx.dynamics.CBXDynamic>` implements the following noise methods:

* anistropic noise (see :func:`anistropic_noise <cbx.dynamics.CBXDynamic.anistropic_noise>`),
* isotropic noise (see :func:`isotropic_noise <cbx.dynamics.CBXDynamic.isotropic_noise>`),
* covariance noise (see :func:`covariance_noise <cbx.dynamics.CBXDynamic.covariance_noise>`).

You can specify the noise as a keyword argument of the class :class:`ParticleDynamic <cbx.dynamics.ParticleDynamic>`:

    >>> from cbx.dynamics import CBXDynamic
    >>> dyn = CBXDynamic(lambda x:x, d=1, noise='isotropic')

Internally this sets the method :func:`noise <cbx.dynamics.CBXDynamic.noise>` of the dynamic class. If you want to implement a custom noise method, the best practice would be to subclass the CBO dynamic class and overwrite the method :meth:`noise <cbx.dynamics.CBXDynamic.noise>`:

    >>> from cbx.dynamics import CBXDynamic
    >>> class MyCBO(CBXDynamic):
    >>>     def noise(self, x):
    >>>         print('This is my custom noise')
    >>>         return np.zeros_like(x)
    >>> dyn = MyCBO(lambda x:x, d=1)
    >>> dyn.noise(dyn.x)
    This is my custom noise

.. note::
    It is technically possible to define a callable ``custom_noise`` and pass it as an argument by calling ``CBXDynamic(..., noise=custom_noise)``. However, this is not recommended, since this callable is not bound to the instance. Also note, that the function :func:`noise <cbx.dynamics.CBXDynamic.noise>` does not take any arguments (other than ``self``). All information about the dynamic (e.g. the drift) is taken from the dynamic class.


Correction steps
----------------

In the original CBO paper it is proposed to perform a correction step on the drift in each iteration. From a techical point of view the mechanics here are very similar to how th noise is implemented.

