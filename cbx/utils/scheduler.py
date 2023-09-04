r"""
Scheduler
==========

This module implements the :math:`\alpha`-schedulers employed in the conensuse schemes.

"""

import numpy as np
from scipy.special import logsumexp
from abc import ABC, abstractmethod
import warnings

class scheduler(object):
    r"""scheduler class
    
    This class implements the base scheduler class. It is used to implement the :math:`\alpha`-schedulers
    employed in the consensus schemes.
    
    Parameters
    ----------
    opt : object
        The optimizer for which the :math:`\alpha`-parameter should be updated

    alpha : float, optional
        The initial value of the :math:`\alpha`-parameter. The default is 1.0.

    alpha_max : float, optional
        The maximum value of the :math:`\alpha`-parameter. The default is 100000.0.

    """

    def __init__(self, dyn, var_params):
        self.dyn = dyn
        self.var_params = var_params

    def update(self):
        for var_param in self.var_params:
            var_param.update(self.dyn)


class param_update(ABC):
    r"""Abstract class for parameter updates

    This class implements the base class for parameter updates.
    """
    def __init__(self, name ='alpha', maximum = 1e5):
        self.name = name
        self.maximum = maximum

    @abstractmethod
    def update(self, dyn):
        pass

    def ensure_max(self, dyn):
        r"""Ensure that the :math:`\alpha`-parameter does not exceed its maximum value."""
        setattr(dyn, self.name, min(self.maximum, getattr(dyn, self.name)))

class multiply(param_update):
    def __init__(self, name = 'alpha',
                 maximum = 1e5, factor = 1.0):
        super(multiply, self).__init__(name=name, maximum=maximum)
        self.factor = factor
    
    def update(self, dyn):
        r"""Update the :math:`\alpha`-parameter in opt according to the exponential scheduler."""
        old_val = getattr(dyn, self.name)
        new_val = min(self.factor * old_val, self.maximum)
        setattr(dyn, self.name, new_val)
        self.ensure_max(dyn)
    
    


# class for alpha_eff scheduler
class effective_number(param_update):
    r"""effective_number scheduler class
    
    This class implements a scheduler for the :math:`\alpha`-parameter based on the effective number of particles.
    The :math:`\alpha`-parameter is updated according to the rule
    
    .. math::
        
        \alpha_{k+1} = \begin{cases}
        \alpha_k \cdot r & \text{if } J_{eff} \geq \eta \cdot J \\ 
        \alpha_k / r & \text{otherwise}
        \end{cases} 
        
    where :math:`r`, :math:`\eta` are parameters and :math:`J` is the number of particles. The effictive number of
    particles is defined as

    .. math::

        J_{eff} = \frac{1}{\sum_{i=1}^J w_i^2}
    
    where :math:`w_i` are the weights of the particles. This was, e.g., employed in [1]_.


    
    Parameters
    ----------
    opt : object
        The optimizer for which the :math:`\alpha`-parameter should be updated
    eta : float, optional
        The parameter :math:`\eta` of the scheduler. The default is 1.0.
    alpha_max : float, optional
        The maximum value of the :math:`\alpha`-parameter. The default is 100000.0.
    factor : float, optional
        The parameter :math:`r` of the scheduler. The default is 1.05. 

    References
    ----------
    .. [1] Carrillo, J. A., Hoffmann, F., Stuart, A. M., & Vaes, U. (2022). Consensus‐based sampling. Studies in Applied Mathematics, 148(3), 1069-1140. 


    """
    def __init__(self, name = 'alpha', eta=1.0, maximum=1e5, factor=1.05):
        super(effective_number, self).__init__(name = name, maximum=maximum)
        if self.name != 'alpha':
            warnings.warn('effective_number scheduler only works for alpha parameter! You specified name = {}!'.format(self.name))
        self.eta = eta
        self.J_eff = 1.0
        self.factor=factor
    
    def update(self, dyn):
        val = getattr(dyn, self.name)
        
        energy = dyn.f(dyn.x)
        dyn.num_f_eval += np.ones(dyn.M) * dyn.batch_size

        term1 = logsumexp(-val * energy)
        term2 = logsumexp(-2 * val * energy)
        self.J_eff = np.exp(2*term1 - term2)
        
        if self.J_eff >= self.eta * dyn.N:
            setattr(dyn, self.name, val * self.factor)
        else:
            setattr(dyn, self.name, val/self.factor)

        self.ensure_max(dyn)