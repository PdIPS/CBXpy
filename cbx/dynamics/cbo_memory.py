import numpy as np
from typing import Union
#from scipy.special import logsumexp

from .cbo import CBO, cbo_update

#%% CBO_Memory
class CBOMemory(CBO):
    r"""Consensus-based optimization with memory effects (CBOMemory) class

    This class implements the CBO algorithm with memory effects as described in [1]_ and [2]_. The algorithm
    is a particle dynamic algorithm that is used to minimize the objective function :math:`f(x)`.

    Parameters
    ----------
    f : objective
        The objective function :math:`f(x)` of the system.
    x : array_like, shape (N, d)
        The initial positions of the particles. For a system of :math:`N` particles, the i-th row of this array ``x[i,:]``
        represents the position :math:`x_i` of the i-th particle.
    y : array_like, shape (N, d)
        The initial positions of the particles. For a system of :math:`N` particles, the i-th row of this array ``y[i,:]``
        represents the or an approximation of the historical best position :math:`y_i` of the i-th particle.
    dt : float, optional
        The parameter :math:`dt` of the system. The default is 0.1.
    alpha : float, optional
        The heat parameter :math:`\alpha` of the system. The default is 1.0.
    lamda : float, optional
        The decay parameter :math:`\lambda` of the system. The default is 1.0.
    noise : noise_model, optional
        The noise model that is used to compute the noise vector. The default is ``normal_noise(dt=0.1)``.
    sigma : float, optional
        The parameter :math:`\sigma` of the noise model. The default is 1.0.
    lamda_memory : float, optional
        The decay parameter :math:`\lambda_{\text{memory}}` of the system. The default is 1.0.
    sigma_memory : float, optional
        The parameter :math:`\sigma_{\text{memory}}` of the noise model. The default is 1.0.
    
    References
    ----------
    .. [1] Grassi, S. & Pareschi, L. (2021). From particle swarm optimization to consensus based optimization: stochastic modeling and mean-field limit.
        Math. Models Methods Appl. Sci., 31(8):1625–1657.
    .. [2] Riedl, K. (2023). Leveraging memory effects and gradient information in consensus-based optimization: On global convergence in mean-field law.
        Eur. J. Appl. Math., XX (X), XX-XX.

    """

    def __init__(self,
                 f,
                 lamda_memory: float = 0.4,
                 sigma_memory: Union[float, None] = None,
                 **kwargs) -> None:
        
        super(CBOMemory, self).__init__(f, **kwargs)
        
        self.lamda_memory = lamda_memory
        
        # init historical best positions of particles
        self.y = self.copy(self.x)
        
        if sigma_memory is None:
            self.sigma_memory = self.lamda_memory * self.sigma
        else:
            self.sigma_memory = sigma_memory
        
        self.energy = self.f(self.x)
        self.num_f_eval += np.ones(self.M, dtype=int) * self.N # update number of function evaluations
        self.ergexp = tuple([Ellipsis] + [None for _ in range(self.x.ndim-2)])
        
    def pre_step(self,):
        # save old positions
        self.x_old = self.copy(self.x) # save old positions
        self.y_old = self.copy(self.y) # save old positions
        
        # set new batch indices
        self.set_batch_idx()
        
    def memory_step(self,):
        # add memory step, first define new drift
        self.drift = self.x[self.particle_idx] - self.y[self.particle_idx]
        self.x[self.particle_idx] += cbo_update(
            self.drift, self.lamda_memory, self.dt, 
            self.sigma_memory, self.noise()
        )
        
    def inner_step(self,) -> None:
        r"""Performs one step of the CBOMemory algorithm.

        Parameters
        ----------
        None

        Returns
        -------
        None
        
        """
        
        # first perform regular cbo step
        self.cbo_step()
        self.memory_step()
        
        # evaluation of objective function on all particles
        energy_new = self.eval_f(self.x[self.particle_idx])  
        
        # historical best positions of particles 
        self.y[self.particle_idx] += (
            ((self.energy>energy_new)[self.ergexp]) * 
            (self.x[self.particle_idx] - self.y[self.particle_idx])
        )
        self.energy = np.minimum(self.energy, energy_new)

        
    def compute_consensus(self,) -> None:
        r"""Updates the weighted mean of the particles.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.consensus, _ = self._compute_consensus(
            self.energy, self.y[self.consensus_idx], 
            self.alpha[self.active_runs_idx, :]
        )
    
    def update_best_cur_particle(self,) -> None:
        self.f_min = self.energy.min(axis=-1)
        self.f_min_idx = self.energy.argmin(axis=-1)
        
        self.best_cur_particle = self.x[np.arange(self.M), self.f_min_idx, :]
        self.best_cur_energy = self.energy[np.arange(self.M), self.f_min_idx]
        