r"""
Scheduler
==========

This module implements the  schedulers employed in conensus based schemes.

"""

import numpy as np
from scipy.special import logsumexp
import warnings

class param_update():
    r"""Base class for parameter updates

    This class implements the base class for parameter updates.

    Parameters
    ----------
    name : str
        The name of the parameter that should be updated. The default is 'alpha'.
    maximum : float
        The maximum value of the parameter. The default is 1e5.
    """
    def __init__(
        self, 
        name: str ='alpha', 
        maximum: float = 1e5,
        minimum: float = 1e-5
    ):
        self.name = name
        self.maximum = maximum
        self.minimum = minimum

    def update(self, dyn) -> None:
        """
        Updates the object with the given `dyn` parameter.

        Parameters
        ----------
        dyn
            The dynamic of which the parameter should be updated.

        Returns
        -------
        None
        """
        pass

    def ensure_max(self, dyn):
        r"""Ensures that the parameter does not exceed its maximum value."""
        setattr(dyn, self.name, np.minimum(self.maximum, getattr(dyn, self.name)))


class scheduler():
    r"""scheduler class
    
    This class allows to update multiple parmeters with one update call.

    Parameters
    ----------
    var_params : list
        A list of parameter updates, that implement an ``update`` function.    

    """

    def __init__(self, var_params):
        self.var_params = var_params

    def update(self, dyn) -> None:
        """
        Updates the dynamic variables in the object.

        Parameters
        ----------
            dyn: The dynamic variables to update.

        Returns
        -------
            None
        """
        for var_param in self.var_params:
            var_param.update(dyn)

class multiply(param_update):
    def __init__(self, 
                 factor = 1.0,
                 **kwargs):
        """
        This scheduler updates the parameter as specified by ``'name'``, by multiplying it by a given ``'factor'``.

        Parameters
        ----------
        factor : float
            The factor by which the parameter should be multiplied.

        """
        super(multiply, self).__init__(**kwargs)
        self.factor = factor
    
    def update(self, dyn) -> None:
        r"""Update the parameter as specified by ``'name'``, by multiplying it by a given ``'factor'``."""
        old_val = getattr(dyn, self.name)
        new_val = self.factor * old_val
        setattr(dyn, self.name, new_val)
        self.ensure_max(dyn)
    

# class for alpha_eff scheduler
class effective_sample_size(param_update):
    r"""effective sample size scheduler class
    
    This class implements a scheduler for the :math:`\alpha`-parameter based on the effective sample size as inroduced 
    in [1]_. In every step we try to find :math:`\alpha` such that
    The :math:`\alpha`-parameter is updated according to the rule
    
    .. math::
        
        J_{eff}(\alpha) = \frac{\left(\sum_{i=1}^N w_i(\alpha)\right)^2}{\sum_{i=1}^N w_i(\alpha)^2} = \eta N
        
    where :math:`\eta` is a parameter, :math:`N` is the number of particles and :math:`w_i := \exp(-\alpha f(x_i))`. The above equation is solved via bisection.


    
    Parameters
    ----------
    eta : float, optional
        The parameter :math:`\eta` of the scheduler. The default is 0.5.
    alpha_max : float, optional
        The maximum value of the :math:`\alpha`-parameter. The default is 100000.0.
    factor : float, optional
        The parameter :math:`r` of the scheduler. The default is 1.05. 

    References
    ----------
    .. [1] Carrillo, J. A., Hoffmann, F., Stuart, A. M., & Vaes, U. (2022). Consensus‐based sampling. Studies in Applied Mathematics, 148(3), 1069-1140. 


    """
    def __init__(self, name = 'alpha', eta=.5, maximum=1e5, solve_max_it = 15):
        super().__init__(name = name, maximum=maximum)
        if self.name != 'alpha':
            warnings.warn('effective_number scheduler only works for alpha parameter! You specified name = {}!'.format(self.name), stacklevel=2)
        self.eta = eta
        self.J_eff = 1.0
        self.solve_max_it = solve_max_it
        
    def update(self, dyn):
        val = getattr(dyn, self.name)
        val = bisection_solve(
            eff_sample_size_gap(dyn.energy, self.eta), 
            self.minimum * np.ones((dyn.M,)), self.maximum * np.ones((dyn.M,)), 
            max_it = self.solve_max_it, thresh=1e-2
        )
        setattr(dyn, self.name, val[:, None])
        self.ensure_max(dyn)
        
        
class eff_sample_size_gap:
    r"""effective sample size gap
    
    This class is used for the effective sample size scheduler. Its call is defined as
    
    .. math::
        
        \alpha \mapsto J_{eff}(\alpha) - \eta N.
        
    Therefore, the root of this non-increasing function solve the effective sampling size equation for :math:`\alpha`.
    """
    
    
    def __init__(self, energy, eta):
        self.eta = eta
        self.energy = energy
        self.N = energy.shape[-1]
    
    def __call__(self, alpha):
        nom   = logsumexp(-alpha[:, None] * self.energy, axis=-1)
        denom = logsumexp(-2 * alpha[:, None] * self.energy, axis=-1)
        return np.exp(2 * nom - denom) - self.eta * self.N
        
def bisection_solve(f, low, high, max_it = 100, thresh = 1e-2, verbosity=0):
    r"""simple bisection optimization to solve for roots
    
    Parameters
    ----------
    f : Callable
        A non-increasing function of which we want to find roots, it expects inputs of the shape (M,) where M denotes
        the number of different runs
    low: Array
        The low initial value for the bisection, should be an array of size (M,)
    high: Array
        The high initial value for the bisection, should be an array of size (M,)
    
    
    Returns
    -------
    roots of the function f
    """
    it = 0
    x = high
    term = False
    idx = np.arange(len(low))
    while not term:
        x = (low + high)/2
        fx = f(x)
        gtzero = np.where(fx[idx] > 0)[0]
        ltzero  = np.where(fx[idx] < 0)[0]
        # update low and high
        low[idx[gtzero]] = x[idx[gtzero]]
        high[idx[ltzero]]  = x[idx[ltzero]]
        # update running idx and iteration
        idx = np.where(np.abs(fx) > thresh)[0]
        it += 1
        term = (it > max_it) | (len(idx) == 0)
    if verbosity > 0:
        print('Finishing after ' + str(it) + ' Iterations')
    return x