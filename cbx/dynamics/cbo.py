from .pdyn import CBXDynamic

def cbo_update(drift, lamda, dt, sigma, noise):
    return -lamda * dt * drift + sigma * noise
#%% CBO
class CBO(CBXDynamic):
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
        super().__init__(f, **kwargs)
        
    def cbo_step(self,):
        # compute consensus, sets self.energy and self.consensus
        self.compute_consensus()
        # update drift and apply drift correction
        self.drift = self.correction(self.x[self.particle_idx] - self.consensus)
        # perform cbo update step
        self.x[self.particle_idx] += cbo_update(
            self.drift, self.lamda, self.dt, 
            self.sigma, self.noise()
        )
        
    inner_step = cbo_step

        