#%%
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
    beta : float, optional
        The heat parameter :math:`\beta` of the system. The default is 1.0.
    
    """

    def __init__(self, x, f, batch_eval: bool = False):
        self.x = x.copy()
        self.N = x.shape[0]
        self.f = self._promote_objective(f, batch_eval)
        self.energy = None
        self.update_diff = float('inf')
        self.f_min = float('inf')


    def _promote_objective(self, f, batch_eval):
        if not callable(f):
            raise TypeError("Objective function must be callable.")
        if batch_eval:
            return f
        else:
            def batch_f(x):
                return np.apply_along_axis(f, 1, x)
            return batch_f