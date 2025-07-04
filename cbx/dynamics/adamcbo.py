import numpy as np
from typing import Tuple
#from scipy.special import logsumexp

from .cbo import CBO

class AdamCBO(CBO):
    """
    AdamCBO is a variant of the CBO algorithm that uses Adam-like updates for the consensus-based optimization.
    This was introduced in the paper:
    "A consensus-based global optimization method with adaptive momentum estimation" by Jingrun Chen, Shi Jin, Liyao Lyu

    Parameters
    ----------
    f : callable
        The objective function to be minimized.
    betas : Tuple[float, float], optional
        Coefficients for the first and second moment estimates, by default (0.9, 0.99).
    eps : float, optional
        A small constant for numerical stability, by default 1e-8.
    **kwargs : additional keyword arguments
        Additional parameters for the CBO algorithm, such as `lamda`, `sigma`, etc.
    """

    def __init__(
        self,
        f,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        **kwargs
    ) -> None:
        super().__init__(f, **kwargs)
        self.betas = betas
        self.eps = eps
        self.m = np.zeros_like(self.x)
        self.v = np.zeros_like(self.x)

    def inner_step(self):
        # compute consensus, sets self.energy and self.consensus
        self.compute_consensus()
        # set drift
        self.drift = self.x[self.particle_idx] - self.consensus
        # update moments
        self.update_moments()

        # perform cbo update step
        self.x[self.particle_idx] -= (
            self.lamda * 
            (
                (self.m[self.particle_idx]  * self.m_factor) / 
                ((self.v[self.particle_idx] * self.v_factor)**0.5 + self.eps)
            ) 
            - self.sigma * self.noise()
        )

    def update_moments(self):
        # update moments
        self.m[self.particle_idx] = self.betas[0] * self.m[self.particle_idx] + (1 - self.betas[0]) * self.drift
        self.v[self.particle_idx] = self.betas[1] * self.v[self.particle_idx] + (1 - self.betas[1]) * self.drift**2

        # update factors
        self.m_factor = 1/(1 - self.betas[0]**(self.it+1))
        self.v_factor = 1/(1 - self.betas[1]**(self.it+1))