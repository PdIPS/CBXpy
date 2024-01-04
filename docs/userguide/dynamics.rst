Dynamics
========

One of the core components in the CBX package is dynamics, which is used to represent different variants of consensus-based algorithms. Each dynamic inherits from the base class :class:`CBXDynamic <cbx.dynamics.CBXDynamic>`, which implements some basic functionality common to all dynamics. This base class itself inherits from :class:`ParticleDynamic <cbx.dynamics.ParticleDynamic>`, which implements functionality specific to particle-based algorithms. The design choice here was to divide between principles common to iterative particle-based algorithms and principles specific to consensus-based algorithms.

Optimization with dynamics
--------------------------

In the simplest case, when selecting a specific dynamic and aiming to optimize a function, you can utilize the function :func:`optimize <cbx.dynamics.CBXDynamic.optimize>`:

    >>> from cbx.dynamics import CBXDynamic
    >>> dyn = CBXDynamic(lambda x:x**2, d=1)
    >>> dyn.optimize()

This function executes the optimization process until a specified termination criterion is met. In scenarios where parameter control via a scheduler is necessary, the function includes the keyword argument sched—for more details, refer to :ref:`sched`.

Each dynamic implements a step method that delineates the update in each iteration (see :ref:`step`). If you wish to run a dynamic at a lower level, directly managing each iteration's operations, you can define a custom for-loop:

    >>> from cbx.dynamics import CBXDynamic
    >>> dyn = CBXDynamic(lambda x:x**2, d=1)
    >>> for i in range(10):
    >>>     dyn.step()


Modelling the ensemble
----------------------

All particle-based methods operate on an ensemble of points :math:`x = (x^1, \ldots, x^N) \in \mathcal{X}^N`. Here, we opt to model the ensemble as an array, assuming :math:`\mathcal{X} = \mathbb{R}^d`, hence representing it as an :math:`N \times d` array. In most cases, we assume this array is provided as a ``numpy`` array. However, it's often straightforward to use torch tensors instead, see also :ref:`npvstorch`.

For multiple runs of a particle-based scheme, creating :math:`M \in \mathbb{N}` instances of a dynamic and running them is a typical approach. This process could also be parallelized at an external level. However, in scenarios where, for instance, all parameters remain fixed across runs, it can be efficient to represent the different runs directly on the array level. We therefore model an ensemble as an

.. math::
    M\times N\times d

array, which allows us to employ parallelization on the array level. The value of :math:`M` defaults to 1, however it is important to always keep in mind that an ensemble has three dimensions. Here, we introduce the term sub-run to denote a distinct run within the aforementioned interpretation, or simply run when the context is evident.

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

However, there might be cases where the user specifies an objective function that only works within the original interpretation, i.e., :math:`f: \mathbb{R}^d \to \mathbb{R}`, as in the following example:

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

We observe that the objective function ``dyn.f`` now returns an array of shape :math:`M\times N`. This is due to the fact that an objective is promoted to the class :func:`cbx_objective <cbx.objectives.Objective>`, which handles the evaluation on the array level. By default it is assumed that the specified function, only works on the single particle level, which is expressed in the keyword argument ``f_dim=1D`` of the class :class:`ParticleDynamic <cbx.dynamics.ParticleDynamic>`. If your function works on single-run ensembles of shape :math:`N\times d`, you can specify ``f_dim=2D`` and respectively if it works on multi-run ensembles of shape :math:`M\times N\times d` you can specify ``f_dim=3D``. If you specify the latter, the objective function is **not** modfied or wrapped, but is directly used for the dynamic:

    >>> import numpy as np
    >>> from cbx.dynamics import ParticleDynamic
    >>>
    >>> def f(x):
    >>>     return abs(x[...,0] + x[...,1])
    >>>
    >>> dyn0 = ParticleDynamic(f, x=np.ones((2,5,2)))
    >>> dyn1 = ParticleDynamic(f, x=np.ones((2,5,2)), f_dim='3D')
    >>>
    >>> print(dyn0.f(np.ones((3,4,2))).shape)
    >>> print(dyn1.f(np.ones((3,4,2))).shape)
    >>> print(dyn0.f is f)
    >>> print(dyn1.f is f)
    (3, 4)
    (3, 4)
    False
    True

Here, we observe that the dynamic directly uses the specified objective function for ``f_dim='3D'``. For more complicated functions, one can also inherit from :class:`cbx_objective <cbx.objectives.Objective>`.

.. note::
    When inheriting from :class:`cbx_objective <cbx.objectives.Objective>`, the method :meth:`__call__ <cbx.objectives.Objective.__call__>` should not be overwritten as it is used internally to update the number of evaluation. Instead, the actual function call should be implemented in the method ``apply(self, x)``.

    >>> import numpy as np
    >>> from cbx.dynamics import ParticleDynamic
    >>> from cbx.utils.objective_handling import cbx_objective
    >>> class objective(cbx_objective):
    >>>     def __init__(self, a=1.0):
    >>>         super().__init__()
    >>>         self.a = a
    >>>     def apply(self, x):
    >>>         return self.a * x[...,0] + x[...,1]

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

In the update step of consensus based methods, diffusion is modeled by the addition of noise, which is scaled by a factor dependent on the iteration. Here, it is very convenient to assume that we can compute the noise, given full information about the dynamic. Therefore, the callable that implements the specific noise needs to accept the dynamic as an argument. This function is then saved in the attribute :attr:`noise_callable <cbx.dynamics.CBXDynamic.noise_callable>`. The function that is called during the iteration :func:`noise <cbx.dynamics.CBXDynamic.noise>` is defined as follows:

.. code-block:: python

    def noise(self):
        return self.noise_callable(self)

You can specify the noise as keyword argument of the class :class:`CBXDynamic <cbx.dynamics.CBXDynamic>`. This can be a string from the following list:

* ``noise = 'anistropic'``: anistropic noise (see :class:`anistropic_noise <cbx.noise.anistropic_noise>`),
* ``noise = 'isotropic'``: isotropic noise (see :class:`isotropic_noise <cbx.noise.isotropic_noise>`),
* ``noise = 'covariance'``: covariance noise (see :class:`covariance_noise <cbx.noise.covariance_noise>`).

You can specify the noise as a keyword argument of the class :class:`ParticleDynamic <cbx.dynamics.ParticleDynamic>`:

    >>> from cbx.dynamics import CBXDynamic
    >>> dyn = CBXDynamic(lambda x:x, d=1, noise='isotropic')

Alternatively, you can define a custom callable and specify it to be used as the ``noise_callable``:

    >>> from cbx.dynamics import CBXDynamic
    >>> def my_noise(dyn):
    >>>     print('This is my custom noise')
    >>> dyn = CBXDynamic(lambda x:x, d=1, noise=my_noise)
    >>> dyn.noise()
    >>> print(dyn.noise_callable is my_noise)
    This is my custom noise
    True



.. note::
    The function :func:`noise <cbx.dynamics.CBXDynamic.noise>` does not take any arguments, other than ``self``.


Correction steps
----------------

In the original CBO paper it is proposed to perform a correction step on the drift in each iteration. From a technical point of view the mechanics here are very similar to how the noise is implemented. The following methods can be specified as keyword argument of the class :class:`CBXDynamic <cbx.dynamics.CBXDynamic>`:

* ``correction = 'none'``: no correction (see :class:`no_correction <cbx.correction.no_correction>`),
* ``correction = 'heavi_side'``: Heaviside correction (see :func:`heavi_side_correction <cbx.correction.heavi_side_correction>`),
* ``correction = 'heavi_side_reg'``: Heaviside correction with regularization (see :func:`heavi_side_correction_reg <cbx.correction.heavi_side_correction_reg>`).

As in the case for the noise, this first sets the function :func:`correction_callable <cbx.dynamics.CBXDynamic.correction_callable>` of the dynamic class. The actual correction is then defined as follows:

.. code-block:: python

    def correction(self, x):
        return self.correction_callable(self, x)

.. note::

    The function :func:`correction <cbx.dynamics.CBXDynamic.correction>` additionally takes ``x`` as an argument.

You can also use a custom callable and specify it to be used as the ``correction_callable``:

    >>> from cbx.dynamics import CBXDynamic
    >>> def my_correction(dyn, x):
    >>>     print('This is my custom correction')
    >>> dyn = CBXDynamic(lambda x:x, d=1, correction=my_correction)
    >>> dyn.correction(dyn.x)
    >>> print(dyn.correction_callable is my_correction)
    This is my custom correction

Termination criteria
--------------------

You can specify different termination criteria for your CBO algorithm, by passing the dictionary ``term_args`` to the class :class:`CBXDynamic <cbx.dynamics.CBXDynamic>`. The function :func:`terminate <cbx.dynamics.CBXDynamic.terminate>` checks all the termination criteria. Since one dynamic contains multiple runs, the checks are performed per run, whenever there might be differences across each run. The list ``dyn.all_check`` saves a Boolean value for each run, that specifies if the run is terminated.

.. note::
    We check whether to terminate the run. Therefore, ``False`` means a certain check is not meant and the run should continue. ``True`` means the check is meant and the run should be stopped.

However, the function :func:`terminate <cbx.dynamics.CBXDynamic.terminate>` only returns a single Boolean value, which is used to decide whether the whole dynamic should be terminated. This is because all these sub-runs are executed by the same step method, by one single dynamic, which needs a single termination check. If this does not fit your application, you can instead use :math:`M` different instances of a dynamic each with the number of sub-runs set to ``1``. You can decide whether to terminate, as soon as one of the sub-runs terminates, or only if all sub-runs terminate, with the keyword ``term_on_all``, i.e., ``term_args = {..., 'term_on_all':True}``.

.. note::
    If we set the option ``term_on_all=False`` (this is also the default option) the particles of sub-runs which already met a termination criterion, will be further updated. It is technically possible, to not update the particles of a sub-run after it terminated, using the values from ``dyn.all_check``, and defining a custom indexing. However, this is not implemented in the dynamics that are provided by the library. If this is a problem for your use-case, you can either specify a custom indexing or use different instances of single-sub-run dynamics.

Internally, an instance of the class :class:`Terminate <cbx.utils.termination.Terminate>` is created, which handles all the checks.

In the following we detail the possible criteria and explain the values that are used:

``term_args = {..., 'max_it': <int>}``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specifies the maximum number of iteration. This is checked with the method :func:`check_max_it <cbx.utils.termination.check_max_it>`. The value ``dyn.it`` is the same across all runs ``M```:

    >>> from cbx.dynamics import CBXDynamic
    >>> dyn = CBXDynamic(lambda x:x, d=1, M=5)
    >>> dyn.step()
    >>> print(dyn.it)
    1

Therefore, the check return the same value across all runs:

    >>> from cbx.dynamics import CBO
    >>> from cbx.utils.termination import check_max_it
    >>> dyn = CBO(lambda x:x, d=1, M=5, term_args={'max_it':2})
    >>> dyn.step()
    >>> print(check_max_it(dyn))
    >>> dyn.step()
    >>> print(check_max_it(dyn))
    False
    True


``term_args = {..., 'max_eval': <int>}``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specify a maximum number of objective evaluations. This is checked in the method :func:`check_max_eval <cbx.utils.termination.check_max_eval>`. The value ``dyn.num_f_eval`` splits up into the different runs. Each ```cbx_objective`` also saves its number of iterations, which are however not split up across different runs.

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
    >>> from cbx.utils.termination import check_max_eval
    >>> dyn = CBO(lambda x:x, d=1, N=20, M=5, check_f_dims=False, term_args={'max_eval':40})
    >>> dyn.step()
    >>> print(check_max_eval(dyn))
    >>> dyn.step()
    >>> print(check_max_eval(dyn))
    [False False False False False]
    [ True  True  True  True  True]

``term_args = {..., 'energy_tol': <float>}``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If this parameter is set, the termination criterion :func:`check_energy <cbx.utils.termination.check_energy>` returns ``True`` whenever the **best** particle of an ensemble is below the energy tolerance.

    >>> from cbx.dynamics import CBO
    >>> from cbx.objectives import Quadratic
    >>> from cbx.utils.termination import check_energy
    >>> import numpy as np
    >>> x = np.array([[[0.], [1.]], [[1.], [1.]]])
    >>> dyn = CBO(Quadratic(), x=x, term_args={'energy_tol':0.5})
    >>> dyn.eval_energy()
    >>> dyn.post_step()
    >>> print(check_energy(dyn))
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


    The energy is computed in the method :func:`eval_energy <cbx.dynamics.CBXDynamic.eval_energy>` and is stored in the attribute ``dyn.energy``. We use the method :func:`post_step <cbx.dynamics.CBXDynamic.post_step>` to update the best found energy in each run, which is stored in the attribute ``dyn.best_energy``. This is then used to in the check :func:`check_energy <cbx.utils.termination.check_energy>`. As expected the first run returns ``True`` since it already found the optimum. For the second one, all particles have an energy above the energy tolerance and therefore the check returns ``False``.

    By default the Boolean ```term_on_all`` is set to ``True``, therefore ``dyn.terminate`` returns ``False``, since not all runs are terminated.

``term_args = {..., 'diff_tol': <float>}``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If this parameter is set, the termination criterion :func:`check_update_diff <cbx.utils.termination.check_update_diff>` returns ``True`` whenever the difference between the previous ensemble and the current one is below the difference tolerance.

    >>> from cbx.dynamics import CBO
    >>> from cbx.objectives import Quadratic
    >>> from cbx.utils.termination import check_diff_tol
    >>> import numpy as np
    >>> dyn = CBO(Quadratic(), d=1, M=2, sigma=0, dt=0., term_args={'diff_tol':0.5})
    >>> dyn.step()
    >>> print(check_diff_tol(dyn))
    >>> print(dyn.terminate())
    [ True  True]
    Run 0 returning on checks:
    check_update_diff
    Run 1 returning on checks:
    check_update_diff
    True

.. note::
    In the above example we set ``dt=sigma=0``, therfore, particles can not move from one iteration to another. The difference between ``dyn.x_old`` and ``dyn.x`` is zero, after one step and therefore the check returns ``True``.

``term_args = {..., 'extra_checks':[<callable>]}``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the keyword ``'extra_checks'`` you can specify a list of callable that perform additional checks. Each callable should take a dynamic object as its single argument and retrun a boolean.

    >>> from cbx.dynamics import CBXDynamic
    >>> def custom_check_0(dyn):
    >>>     print('This is custom check 0')
    >>>     return False
    >>> def custom_check_1(dyn):
    >>>     print('This is custom check 1')
    >>>     return True
    >>> dyn = CBXDynamic(lambda x:x, d=1, term_args={'extra_checks':[custom_check_0, custom_check_1]})
    >>> dyn.terminate()
    This is custom check 0
    This is custom check 1
    Run 0 returning on checks:
    custom_check_1

.. _tracking:
Tracking and history
--------------------

During the iteration, we can save different values in the dictionary ``dyn.history``. You can specify, which values to track, with the dictonary ``track_args``. In the follwing we specify possible keys:

``track_args={...,'save_int': <int>}``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The value ``save_int`` specifiy the interval in which the values should be tracked.

``track_args={...,names=[....,'x']}``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specifies, that the particles ``dyn.x`` should be tracked after each step. In that case the entry in the history ``dyn.history['x']`` is a basic list of arrays of shape ``(M, N, d)``.

``track_args={...,names=[....,'update_norm']}``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specifies, that the norm of the difference between the old and the new ensemble should be tracked. The values are saved in ``dyn.history['update_norm']`` which is a list of arrays of shape ``(M,)``.

``track_args={...,names=[....,'energy']}``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specifies, that the **best** energy in each iteration should be tracked. The values are saved in ``dyn.history['energy']`` which is a list of arrays of shape ``(M,)``.

``track_args={...,names=[....,'consensus']}``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specifies, that the consensus points should be tracked. They are saved in ``dyn.history['consensus']`` which is a list of arrays of shape ``(M, d)``. This is only available in the subclass :class:`CBXDynamic cbx.dynamics.CBXDynamic`.

``track_args={...,names=[....,'drift']}``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specifies, that the drift vectors should be tracked. They are saved in ``dyn.history['drift']`` which is a list of arrays of shape ``(M, d)``. This is only available in the subclass :class:`CBXDynamic cbx.dynamics.CBXDynamic`.

``track_args={...,names=[....,'drift_mean']}``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specifies that the mean of the drift vectors should be tracked. It is saved in ``dyn.history['drift_mean']`` which is a list of arrays of shape ``(M, d)``.

``track_args={...,extra_tracks=[<track>]}``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The list ``extra_tracks`` allows you to specify additional functions that perform custom tracking routines. The instances should be a class that implement the following functions

* ``init_history``: Here you initialize the value in ``dyn.history``, e.g., you can initialize an array or list to store the values in.
* ``update``: This performs the tracking after each update.


    >>> from cbx.dynamics import CBXDynamic
    >>> class MyCustomTrack:
    >>>     def init_history(dyn):
    >>>         print('Initializing my custom track')
    >>>         dyn.history['my_custom_track'] = []
    >>>     def update(dyn):
    >>>         print('Updating my custom track')
    >>>         dyn.history['my_custom_track'].append(dyn.x.min(axis=-1))
    >>> dyn = CBXDynamic(lambda x:x, d=1, track_args={'extra_tracks':[MyCustomTrack]})
    >>> dyn.step()
    >>> print(dyn.history['my_custom_track'])
    Initializing my custom track
    Updating my custom track
    [0]

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

