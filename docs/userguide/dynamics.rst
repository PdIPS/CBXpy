Dynamics
========

One of the core components in the CBX package are dynamics which are used to represent different variants of consensus based algorithms. Each dynamic inherits from the base class :class:`CBXDynamic <cbx.dynamics.CBXDynamic>` which implements some basic functionality, that is common to all dynamics. This base class itself inherits from :class:`ParticleDynamic <cbx.dynamics.ParticleDynamic>` which implements functionality that is specific to particle based algorithms. The design choice here, was to divide between principles that are common to iterative particle based algorithms and principles that are specific to consensus based algorithms.

Optimization with dynamics
--------------------------

In the simplest case, where you choose a certain dynamic and just want to optimize a function you can use :func:`optimize <cbx.dynamics.CBXDynamic.optimize>`:

    >>> from cbx.dynamics import CBO
    >>> dyn = CBO(lambda x:x**2, d=1)
    >>> dyn.optimize()

This will run the optimization until a certain termination criterion is met. In some cases you might want to control parameters, via a scheduler, therefore the function takes the keyword argument ``sched``, see also :ref:`sched`.

Each dynamic, implements a ``step`` method that describes the update in each iteration, see :ref:`step`. When you want to run a dynamic on a low-level, controlling directly, what happens in each iteration you can simply define a custom for-loop:

    >>> from cbx.dynamics import CBXDynamic
    >>> dyn = CBXDynamic(lambda x:x**2, d=1)
    >>> for i in range(10):
    >>>     dyn.step()


Modelling the ensemble
----------------------

All particle based methods work with an ensemble of points :math:`x = (x^1, \ldots, x^N)\in \mathcal{X}^N`. Here, we choose to model the ensemble as an array, e.g., we assume that :math:`\mathcal{X} = \mathbb{R}^{d_1\times \ldots \times d_s}` and therefore, we can represent it as an :math:`N\times d_1\times \ldots \times d_s` array. In most cases we assume that the array is given as a ``numpy`` array. However, in many cases one can simply adapt the functionality to also use other array-like structures. This also allows for execution on the GPU, using e.g. ``torch``, see also :ref:`_torchwithcbx`.

In many cases one wants to perform more than one run of a certain particle based scheme. In this case, a straightforward approach is to create :math:`M\in \mathbb{N}` instances of a dynamic and run them. This could also be parallelized on the external level. However, in many cases, e.g., if all parameters stay fixed across runs, it can be efficient to directly represent the different runs on the array level. We therefore model the ensemble as an

.. math::
    M\times N\times d_1\times \ldots \times d_s

array, which allows us to employ parallelization on the array level. The value of :math:`M` defaults to 1, however it is important to always keep in mind that an ensemble has always :math:`2+s` different dimensions, where :math:`s` is given by the optimization space :math:`\mathcal{X} = \mathbb{R}^{d_1\times \ldots \times d_s}`. In the following we use the abbreviation :math:`d = (d_1,\ldots, d_s)`.

.. note::
    One might ask what the difference between a dynamic with the array structure ``(M,N,d)`` and a dynamic with the structure ``(1,M*N,d)`` is. The difference becomes visible, whenever particles interact across their ensemble. E.g., in the first case, when the consensus is computed, we compute it separately for each :math:`m\in\{1,\ldots,M\}` sub-runs,

    .. math::
        c(x)^{m} = \frac{\sum_{n=1}^N x^{m,n}\ \exp(-\alpha\ f(x^{m,n}))}{\sum_{n=1}^N \exp(-\alpha\ f(x^{m,n}))},

    whereas in the second case, we have one single ensemble with :math:`M\cdot N` particles and therefore compute a single consensus point

    .. math::
        \tilde c(x) = \frac{\sum_{m=1}^M \sum_{n=1}^N x^{m,n}\ \exp(-\alpha\ f(x^{m,n}))}{\sum_{m=1}^M \sum_{n=1}^N \exp(-\alpha\ f(x^{m,n}))}.


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

.. _step:
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

* ``noise = 'anistropic'``: anistropic noise (see :func:`anistropic_noise <cbx.dynamics.CBXDynamic.anistropic_noise>`),
* ``noise = 'isotropic'``: isotropic noise (see :func:`isotropic_noise <cbx.dynamics.CBXDynamic.isotropic_noise>`),
* ``noise = 'covariance'``: covariance noise (see :func:`covariance_noise <cbx.dynamics.CBXDynamic.covariance_noise>`).

You can specify the noise as a keyword argument of the class :class:`ParticleDynamic <cbx.dynamics.ParticleDynamic>`:

    >>> from cbx.dynamics import CBXDynamic
    >>> dyn = CBXDynamic(lambda x:x, d=1, noise='isotropic')

Internally this sets the method :func:`noise <cbx.dynamics.CBXDynamic.noise>` of the dynamic class. If you want to implement a custom noise method, you can subclass the CBO dynamic class and overwrite the method :meth:`noise <cbx.dynamics.CBXDynamic.noise>`:

    >>> from cbx.dynamics import CBXDynamic
    >>> class MyCBO(CBXDynamic):
    >>>     def noise(self,):
    >>>         print('This is my custom noise')
    >>>         return np.zeros_like(x)
    >>> dyn = MyCBO(lambda x:x, d=1)
    >>> dyn.noise(dyn.x)
    This is my custom noise


Correction steps
----------------

In the original CBO paper it is proposed to perform a correction step on the drift in each iteration. From a technical point of view the mechanics here are very similar to how the noise is implemented. The following methods are implemented in the base class :func:`CBO <cbx.dynamics.CBXDynamic>`:

* ``correction = 'none'``: no correction (see :func:`no_correction <cbx.dynamics.CBXDynamic.no_correction>`),
* ``correction = 'heavi_side'``: Heaviside correction (see :func:`heavi_side_correction <cbx.dynamics.CBXDynamic.heavi_side_correction>`),
* ``correction = 'heavi_side_reg'``: Heaviside correction with regularization (see :func:`heavi_side_correction_reg <cbx.dynamics.CBXDynamic.heavi_side_correction_reg>`).


Internally this sets the method :func:`noise <cbx.dynamics.CBXDynamic.correction>` of the dynamic class. If you want to implement a custom correction method, you can subclass the CBO dynamic class and overwrite the method :meth:`noise <cbx.dynamics.CBXDynamic.correction>` just as in the noise case:

    >>> from cbx.dynamics import CBXDynamic
    >>> class MyCBO(CBXDynamic):
    >>>     def correction(self, x):
    >>>         print('This is my custom correction')
    >>>         return np.zeros_like(x)
    >>> dyn = MyCBO(lambda x:x, d=1)
    >>> dyn.correction(dyn.x)
    This is my custom correction

If you would rather define a class such that users can specify your custom correction as keyword argument you need to edit the attribute ``correction_dict`` as follows:

    >>> from cbx.dynamics import CBXDynamic
    >>> class MyCBO(CBXDynamic):
    >>>     def custom_correction(self, x):
    >>>         print('This is my custom correction')
    >>>         return np.zeros_like(x)
    >>>     correction_dict = {**CBXDynamic.correction_dict, 'custom': 'custom_correction'}
    >>> dyn = MyCBO(lambda x:x, d=1, correction='custom')
    >>> dyn.correction(dyn.x)
    This is my custom correction


Termination criteria
--------------------

You can specify different termination criteria for your CBO algorithm, by passing a keyword argument to the CBO class. The function :func:`terminate <cbx.dynamics.CBXDynamic.terminate>` checks all the termination criteria. Since one dynamic contains multiple runs, the checks are performed per run, whenever there might be differences across each run. The list ``dyn.all_check`` saves a Boolean value for each run, that specifies if the run is terminated.

.. note::
    We check whether to terminate the run. Therefore, ``False`` means a certain check is not meant and the run should continue. ``True`` means the check is meant and the run should be stopped.

However, the function :func:`terminate <cbx.dynamics.CBXDynamic.terminate>` only returns a single Boolean value, which used to decide whether the whole dynamic should be terminated. This is due the fact, that all these sub-runs are executed by the same same step method, by one single dynamic, which needs a single termination check. If this does not fit your application, you can instead use :math:`M` different instances of a dynamic each with the number of sub-runs set to ``1``. You can decide whether to terminate, as soon as one of the sub-runs terminates, or only if all sub-runs terminate, with the keyword ``term_on_all``. 

.. note::
    If we set the option ``term_on_all=False`` (this is also the default option) the particles of sub-runs which already met a termination criterion, will be further updated. It is technically possible, to not update the particles of a sub-run after it terminated, using the values from ``dyn.all_check``, and defining a custom indexing. However, this is not implemented in the dynamics that are provided by the library. If this is a problem for your use-case, you can either specify a custom indexing or use different instances of single-sub-run dynamics.


In the following we detail the possible criteria and explain the values that are used:

``max_it``
^^^^^^^^^^

Specifies the maximum number of iteration. This is checked in the method :func:`check_max_it <cbx.dynamics.CBXDynamic.check_max_it>`. The value ``dyn.it`` is the same across all runs ``M```:

    >>> from cbx.dynamics import CBXDynamic
    >>> dyn = CBXDynamic(lambda x:x, d=1, M=5)
    >>> dyn.step()
    >>> print(dyn.it)
    1

Therefore, the check return the same value across all runs:

    >>> from cbx.dynamics import CBO
    >>> dyn = CBO(lambda x:x, d=1, M=5, max_it=2)
    >>> dyn.step()
    >>> print(dyn.check_max_it())
    >>> dyn.step()
    >>> print(dyn.check_max_it())
    False
    True


``max_eval``
^^^^^^^^^^^^

Specify a maximum number of objective evaluations. This is checked in the method :func:`check_max_eval <cbx.dynamics.CBXDynamic.check_max_eval>`. The value ``dyn.num_f_eval`` splits up into the different runs. Each ```cbx_objective`` also saves its number of iterations, which are however not split up across different runs.

    >>> from cbx.dynamics import CBO
    >>> dyn = CBO(lambda x:x, d=1, N=20, M=5, check_f_dims=False)
    >>> dyn.step()
    >>> print(dyn.num_f_eval)
    >>> print(dyn.f.num_eval)
    [20 20 20 20 20]
    100

.. note::
    In the above example we used the keyword argument ``check_f_dims=False`` to disable the check of the dimensionality of the objective function. Per default this check is enabled, in order to ensure that the objective functions returns the right dimension. However, this yields some extra evaluations. 

    We used the standard CBO algorithm, where one step requires us to compute the consensus point

    .. math::
        c_\alpha(x) = \frac{\sum_{n=1}^n x^N\ \exp(-\alpha\ f(x^n))}{\sum_{n=1}^N \exp(-\alpha\ f(x^n))}.

    For each run, we need to evaluate the objective function on the :math:`N` different particesl, which yields :math:`N` evaluations per run. In total the function is evaluated :math:`N\cdot M` times.

Since this value is evaluated per run, also the check is performed per run:

    >>> from cbx.dynamics import CBO
    >>> dyn = CBO(lambda x:x, d=1, N=20, M=5, check_f_dims=False, max_eval=40)
    >>> dyn.step()
    >>> print(dyn.check_max_eval())
    >>> dyn.step()
    >>> print(dyn.check_max_eval())
    [False False False False False]
    [ True  True  True  True  True]

``energy_tol``
^^^^^^^^^^^^^^

If this parameter is set, the termination criterion :func:`check_energy <cbx.dynamics.CBXDynamic.check_energy>` returns ``True`` whenever the **best** particle of an ensemble is below the energy tolerance.

    >>> from cbx.dynamics import CBO
    >>> from cbx.objectives import Quadratic
    >>> import numpy as np
    >>> x = np.array([[[0.], [1.]], [[1.], [1.]]])
    >>> dyn = CBO(Quadratic(), x=x, energy_tol=0.5)
    >>> dyn.eval_energy()
    >>> dyn.post_step()
    >>> print(dyn.check_energy())
    >>> print(dyn.terminate())
    [ True False]
    False

.. note::
    In the above example we choose the initial configuration ``x`` with shape (2, 2, 1), i.e., we have ``M=2`` runs, ``N=2`` particles per run and ``d=1``. The particles are chosen as 

    .. math::
        x^{1,:} = \begin{bmatrix} [0]\\ [1] \end{bmatrix},\quad
        x^{2,:} = \begin{bmatrix} [1]\\ [1] \end{bmatrix},

    and the objective function is defined as

    .. math::
        f(x) = x^2

    Therfore, the first particle in the first run, is already the optimum, :math:`x^{1,1} = 0`, with an energy of :math:`f(x^{1,1}) = 0`. On the other hand the second run has two particles with the sam energy :math:`f(x^{2,1}) = f(x^{2,2}) = 1`.


    The energy is computed in the method :func:`eval_energy <cbx.dynamics.CBXDynamic.eval_energy>` and is stored in the attribute ``dyn.energy``. We use the method :func:`post_step <cbx.dynamics.CBXDynamic.post_step>` to update the best found energy in each run, which is stored in the attribute ``dyn.best_energy``. This is then used to in the check :func:`check_energy <cbx.dynamics.CBXDynamic.check_energy>`. As expected the first run returns ``True`` since it already found the optimum. For the second one, all particles have an energy above the energy tolerance and therefore the check returns ``False``. 
    
    By default the Boolean ```term_on_all`` is set to ``True``, therefore ``dyn.terminate`` returns ``False``, since not all runs are terminated.

``diff_tol``
^^^^^^^^^^^^

If this parameter is set, the termination criterion :func:`check_update_diff <cbx.dynamics.CBXDynamic.check_update_diff>` returns ``True`` whenever the difference between the previous ensemble and the current one is below the difference tolerance.


    >>> from cbx.dynamics import CBO
    >>> from cbx.objectives import Quadratic
    >>> import numpy as np
    >>> dyn = CBO(Quadratic(), d=1, sigma=0, dt=0., diff_tol=0.5)
    >>> dyn.step()
    >>> print(dyn.check_update_diff())
    >>> print(dyn.terminate())
    [ True  True]
    Run 0 returning on checks: 
    check_update_diff
    Run 1 returning on checks: 
    check_update_diff
    True

.. note::
    In the above example we set ``dt=sigma=0``, therfore, particles can not move from one iteration to another. The difference between ``dyn.x_old`` and ``dyn.x`` is zero, after one step and therefore the check returns ``True``.

.. _tracking:
Tracking and history
--------------------

Dynamics that inherit from :class:`ParticleDynamic <cbx.dynamics.ParticleDynamic>` allow to track different quantities during the iteration. The function :func:`track <cbx.dynmaics.ParticleDynamic.track>` updates the dictionary ``dyn.history``, which stores the values, as specified by the user. In order to decide, which values to track, you can you use the keyword argument ``track_list``, which expects a list of strings that specifiy the name of the tracked object. In the following we detail, which strings are possible for :class:`ParticleDynamic <cbx.dynamics.ParticleDynamic>`:

* ``'x'```: Specifies, that the particles ``dyn.x`` should be tracked after each step. Note, that in that case the entry in the history ``dyn.history['x']`` is initialized as a ``(max_it, M, N, d)`` array. It might happen that this is to large for your memory. Furthermore, the array is **not** reshaped after the termination of the iteration. If the dynamic allows for ``max_it`` many iterations, but terminates already after less iterations, the entry ``dyn.history['x']`` will still have the shape ``(max_it, M, N, d)``.

* ``'update_norm'``: Specifies, that the norm of the difference between the old and the new ensemble should be tracked. The values are save in ``dyn.history['update_norm']`` which is a ``(max_it, M)`` array.

* ``'energy'``: Specifies, that the **best** energy in each iteration should be tracked. The values are saved in ``dyn.history['energy']`` which is a ``(max_it, M)`` array.

The subclass :class:`CBXDynamic cbx.dynamics.CBXDynamic` additionally allows to specify the following:

* ``'consensus'``: Specifies, that the consensus points should be tracked. They are saved in ``dyn.history['consensus']`` which is a ``(max_it, M, d)`` array.

* ``'drift'``: Specifies, that the drift vectors should be tracked. They are saved in ``dyn.history['drift']``. Since the dimensionality of the drift is not clear a priori, this is stored as a list, where in each step the new drift is appended.

* ``'drift_mean'``: Specifies that the mean of the drift vectors should be tracked. It is saved in ``dyn.history['drift_mean']`` which is a ``(max_it, M, d)`` array.

Specifying a custom track value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to use a custom tracking value you need to subclass :class:`CBXDynamic cbx.dynamics.CBXDynamic` and define the following functions (the concrete names are not important):

* ``track_custom_init``: Here you initialize the value in ``dyn.history``, e.g., you can initialize an array to store the values in.
* ``track_custom``: This performs the actual tracking after each update.

In order to make this available via the ``track_list`` functionality, we need to alter the dictionary ``track_dict`` as follows:

    >>> from cbx.dynamics import CBXDynamic
    >>> class custom_CBO(CBXDynamic):
    >>>     ...
    >>>     def track_custom_init(self,):
    >>>         ...
    >>>
    >>>     def track_custom(self,):
    >>>         ...
    >>>
    >>>     track_dict = {'custom': ('track_custom_init', 'track_custom'), **CBXDynamic.track_dict}

This allows us to specify the value 'custom' in the above class, via ``custom_CBO(..., track_list=[...,'custom'])``.


Batching
--------

As proposed in [1]_ it is common to perform only batch updates across the ensemble. In order to specify batching in a cbx class you can use the keyword argument ``CBXDynamic(...,batch_args=batch_args)``, where ``batch_args`` is a dictionary with the following keys:

* ``'batch_partial'``: If ``True`` the consensus and particle indices are the same. If ``False`` the particle indices are an ``Ellipsis``.

* ``'batch_size'``: The size of the batch.

* ``'seed'``: The seed for the random number generator.

* ``'var'``: The resampling variant. 

We explain the mechanism and the behavior of these arguments below.

.. note::
    Here, and in the following this batching should not be confused with the batching of a objective function. If your objective function is given as a sum over many functions, it might make sense to batch the evaluation of this function. However, the batching over the ensemble is conceptually different.


The base class :class:`CBXDynamic <cbx.dynamics.CBXDynamic>` implements the function :func:`set_batch_idx <cbx.dynamics.CBXDynamic.set_batch_idx>`. If it is called it sets the following attributes 

* ``dyn.consensus_idx``: the indices used to computed the consensus point,
* ``dyn.particle_idx``: the indices updated in each step.

The keyword argument ``batch_partial``decides how consensus and particle indices relate to each other:


* ``batch_partial=True``: the consensus and particle indices are the same.
* ``batch_partial=False``: each particle is updated from the partially computed consensus and therefore, the particle indices are an ``Ellipsis``.

The attribute ``dyn.consenus_idx`` is a tuple of array indices such that we can directly use it for array indexing:

    >>> import numpy as np
    >>> from cbx.dynamics import CBXDynamic
    >>> dyn = CBXDynamic(lambda x:x, M=4, N=5, d=1, batch_args={'size':2})
    >>> dyn.set_batch_idx()
    >>> print(dyn.consensus_idx)
    >>> print(dyn.x[dyn.consensus_idx].shape)
    (array([[0, 0],
            [1, 1],
            [2, 2],
            [3, 3]]),
     array([[1, 3],
            [1, 3],
            [0, 1],
            [2, 0]]),
     Ellipsis)
     (4, 2, 1)

The first entry, allows for convenient broadcasting in the run dimension, this array :math:`M\in\N_0^{M\times\text{batch_size}}`is deterministic and defined as

.. math::
    M_{m, n} := n.

The second entry stores the indices of the particles that belong to the current batch. This array has the same shape as the previous one and randomly selects indices in the range ``0`` to ``N-1``, independently across each run. In the best the indices are unique within a single sub-run.


Performance evaluation
----------------------


References
----------

.. [1] Carrillo, J. A., Jin, S., Li, L., & Zhu, Y. (2021). A consensus-based global optimization method for high dimensional machine learning problems. ESAIM: Control, Optimisation and Calculus of Variations, 27, S5.

