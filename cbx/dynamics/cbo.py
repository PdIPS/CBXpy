import numpy as np
from scipy.special import logsumexp

from .pdyn import ParticleDynamic

#%% CBO
class CBO(ParticleDynamic):
    r"""Consensus-based optimization (CBO) class

    This class implements the CBO algorithm as described in [1]_. The algorithm
    is a particle dynamic algorithm that is used to minimize the objective function :math:`f(x)`.

    Parameters
    ----------
    x : array_like, shape (N, d)
        The initial positions of the particles. For a system of :math:`N` particles, the i-th row of this array ``x[i,:]``
        represents the position :math:`x_i` of the i-th particle.
    f : objective
        The objective function :math:`f(x)` of the system.
    dt : float, optional
        The parameter :math:`dt` of the system. The default is 0.1.
    lamda : float, optional
        The decay parameter :math:`\lambda` of the system. The default is 1.0.
    alpha : float, optional
        The heat parameter :math:`\alpha` of the system. The default is 1.0.
    noise : noise_model, optional
        The noise model that is used to compute the noise vector. The default is ``normal_noise(dt=0.1)``.
    sigma : float, optional
        The parameter :math:`\sigma` of the noise model. The default is 1.0.
    
    References
    ----------
    .. [1] Pinnau, R., Totzeck, C., Tse, O., & Martin, S. (2017). A consensus-based model for global optimization and its mean-field limit. 
        Mathematical Models and Methods in Applied Sciences, 27(01), 183-204.

    """

    def __init__(self, f, **kwargs) -> None:
        super(CBO, self).__init__(f, **kwargs)
        
    
    def step(self,) -> None:
        r"""Performs one step of the CBO algorithm.

        Parameters
        ----------
        None

        Returns
        -------
        None
        
        """
        self.pre_step()
        
        # update, consensus point, drift and energy
        self.consensus, energy = self.compute_consensus(self.x[self.consensus_idx])        
        self.drift = self.x[self.particle_idx] - self.consensus
        self.energy[self.consensus_idx] = energy
        
        # compute noise
        self.s = self.sigma * self.noise()

        #  update particle positions
        self.x[self.particle_idx] = (
            self.x[self.particle_idx] -
            self.correction(self.lamda * self.dt * self.drift) +
            self.s)
        
        self.post_step()
        
        
    def compute_consensus(self, x_batch) -> None:
        r"""Updates the weighted mean of the particles.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # evaluation of objective function on batch
        energy = self.f(x_batch) # update energy
        self.num_f_eval += np.ones(self.M,dtype=int) * x_batch.shape[-2] # update number of function evaluations
        
        weights = - self.alpha * energy
        coeffs = np.exp(weights - logsumexp(weights, axis=(-1,), keepdims=True))[...,None]
        
        problem_idx = np.where(np.abs(coeffs.sum(axis=-2)-1) > 0.1)[0]
        if len(problem_idx) > 0:
            raise RuntimeError('Problematic consensus computation!')
        
        return (x_batch * coeffs).sum(axis=-2, keepdims=True), energy