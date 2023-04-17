#%%
from typing import Callable, Union

import numpy as np


class ParticleDynamic():
    r"""Particle dynamic class

    This class implements the base particle dynamic class. It is used to update the particles
    in the consensus schemes.   

    Parameters
    ----------
    x : array_like, shape (J, d)
        The initial positions of the particles. For a system of :math:`J` particles, the i-th row of this array ``x[i,:]``
        represents the position :math:`x_i` of the i-th particle.
    f : obejective
        The objective function :math:`f(x)` of the system.
    alpha : float, optional
        The heat parameter :math:`\alpha` of the system. The default is 1.0.
    
    """

    def __init__(self, x: np.ndarray, f: Callable, batch_eval: bool = False, 
                 correction: Union[None, str, Callable] = None, correction_eps: float = 1e-3):
        self.N = x.shape[0]
        self.d = x.shape[1]
        self.x = x.copy()

        self.f = self._promote_objective(f, batch_eval)
        self.energy = None
        self.update_diff = float('inf')
        self.f_min = float('inf')

        if correction is None:
            self.correction = no_correction()
        elif correction == 'heavi_side':
            self.correction = heavi_side()
        elif correction == 'heavi_side_reg':
            self.correction = heavi_side_reg(eps = correction_eps)
        else:
            self.correction = correction


    def _promote_objective(self, f, batch_eval):
        if not callable(f):
            raise TypeError("Objective function must be callable.")
        if batch_eval:
            return f
        else:
            return batched_objective(f)
        
class batched_objective:
    def __init__(self, f):
        self.f = f
    
    def __call__(self, x):
        return np.apply_along_axis(self.f, 1, np.atleast_2d(x))
    

class no_correction:
    def __call__(self, x):
        return np.ones(x.shape)

class heavi_side:
    def __call__(self, x):
        return np.where(x > 0, 1,0)

class heavi_side_reg:
    def __init__(self, eps=1e-3):
        self.eps = eps
    
    def __call__(self, x):
        return 0.5 + 0.5 * np.tanh(x/self.eps)