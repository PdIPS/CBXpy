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
                 batch_size: Union[None, int] = None,
                 energy_tol: float = 0., diff_tol: float = 0.,
                 max_eval: int = float('inf'),
                 correction: Union[None, str, Callable] = None, correction_eps: float = 1e-3,
                 T: float = 100.):
        self.N = x.shape[0]
        self.d = x.shape[1]
        self.x = x.copy()

        self.f = self._promote_objective(f, batch_eval)
        self.f_min = float('inf')
        self.num_f_eval = 0
        self.energy = np.ones((self.N, 1)) * float('inf')
        self.update_diff = float('inf')

        # termination parameters
        self.energy_tol = energy_tol
        self.diff_tol = diff_tol
        self.max_eval = max_eval
        self.T = T
        
        self.m_alpha = np.zeros((1, self.d))
        self.t = 0.
        

        # termination checks
        self.checks = [self.check_max_time, self.check_energy, self.check_update_diff, self.check_max_eval]

        if correction is None:
            self.correction = no_correction()
        elif correction == 'heavi_side':
            self.correction = heavi_side()
        elif correction == 'heavi_side_reg':
            self.correction = heavi_side_reg(eps = correction_eps)
        else:
            self.correction = correction

        # batching
        if batch_size is None:
            self.batch_size = self.N
        else:
            self.batch_size = min(batch_size, self.N)
        self.indices = np.random.permutation(self.N)


    def _promote_objective(self, f, batch_eval):
        if not callable(f):
            raise TypeError("Objective function must be callable.")
        if batch_eval:
            return f
        else:
            return batched_objective(f)

    def set_batch_idx(self,):
        r"""Set batch indices

        This method sets the batch indices for the next iteration of the algorithm.
        """
            
        if len(self.indices) < self.batch_size: # if indices are exhausted
            indices = np.random.permutation(self.N)
            self.indices = np.concatenate((self.indices, indices))

        self.batch_idx = self.indices[:self.batch_size] # get batch indices
        self.indices = self.indices[self.batch_size:] # remove batch indices from indices

    def check_max_time(self):
        return self.t > self.T
            
    def check_energy(self):
        return self.f_min < self.energy_tol
    
    def check_update_diff(self):
        return self.update_diff < self.diff_tol
    
    def check_max_eval(self):
        return self.num_f_eval > self.max_eval
    
    def terminate(self):
        for check in self.checks:
            if check():
                print('Returning on check: ' + check.__name__)
                return True
    

class batched_objective:
    def __init__(self, f):
        self.f = f
    
    def __call__(self, x):
        return np.apply_along_axis(self.f, 1, np.atleast_2d(x))
    

class no_correction:
    def __call__(self, dyn):
        return np.ones((dyn.N,1))

class heavi_side:
    def __call__(self, dyn):
        x = dyn.energy - dyn.f(dyn.m_alpha)
        dyn.num_f_eval += dyn.m_alpha.shape[0] # update number of function evaluations

        return np.where(x > 0, 1,0)

class heavi_side_reg:
    def __init__(self, eps=1e-3):
        self.eps = eps
    
    def __call__(self, dyn):
        x = dyn.energy - dyn.f(dyn.m_alpha)
        dyn.num_f_eval += dyn.m_alpha.shape[0] # update number of function evaluations

        return 0.5 + 0.5 * np.tanh(x/dyn.eps)
    