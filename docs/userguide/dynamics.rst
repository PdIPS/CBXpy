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

```python
def step(self):
    self.inner_step()
    self.update()
```




In the following we explain the different building blocks the method :meth:`step <cbx.dynamics.CBXDynamic.step>`:, which captures the update of the ensemble. It consists of three parts, the pre-, inner- and post-step and usually is then defined as

>>> class SomeDynamic(ParticleDynamic):
>>>     ...
>>>     def step(self):
>>>         self.pre_step()
>>>         self.inner_step()
>>>         self.post_step()

