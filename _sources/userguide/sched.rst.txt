.. _sched:
Schedulers
==========

Schedulers allow to update parameters of the dynamic. Most commonly, one may want to adjust the heat parameter :math:`\alpha` during the iteration. 
The base class is :class:`cbx.scheduler.param_update` and all schedulers are derived from it. 

The function that performs the update is :func:`update <cbx.scheduler.param_update.update>`. The scheduler accesses the parameters of the dynamic from the outside, therefore the dynamic needs to be passed to the update function.  


The easiest scheduler :class:`multiply <cbx.scheduler.multiply>` simply multiplies the specified value by a constant.

    >>> from cbx.dynamics import CBXDynamic
    >>> from cbx.scheduler import multiply
    >>> dyn = CBXDynamic(lambda x:x**2, d=1, alpha=1.0)
    >>> sched = multiply(name='alpha', factor=0.5)
    >>> sched.update(dyn)
    >>> print(dyn.alpha)
    0.5


Specifying multiple updates
---------------------------

If you want to specify multiple parameter updates, you can use the class :class:`scheduler <cbx.scheduler.scheduler>` which allows you to pass multiple scheduler. It also implements an update function, which uses the ``update`` function of the individual schedulers.

    >>> from cbx.dynamics import CBXDynamic
    >>> from cbx.scheduler import scheduler, multiply
    >>> dyn = CBXDynamic(lambda x:x**2, d=1, alpha=1.0, dt=0.5)
    >>> sched = schedulr([multiply(name='dt', factor=0.5), multiply(name='alpha', factor=0.5)])
    >>> sched.update(dyn)
    >>> print(dyn.alpha)
    >>> print(dyn.dt)
    0.5
    0.25

