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
    x : array_like, shape (J, d)
        The initial positions of the particles. For a system of :math:`J` particles, the i-th row of this array ``x[i,:]``
        represents the position :math:`x_i` of the i-th particle.
    f : obejective
        The objective function :math:`f(x)` of the system.
    alpha : float, optional
        The heat parameter :math:`\alpha` of the system. The default is 1.0.
    noise : noise_model, optional
        The noise model that is used to compute the noise vector. The default is ``normal_noise(dt=0.1)``.
    dt : float, optional
        The parameter :math:`dt` of the noise model. The default is 0.1.
    sigma : float, optional
        The parameter :math:`\sigma` of the noise model. The default is 1.0.
    lamda : float, optional
        The decay parameter :math:`\lambda` of the noise model. The default is 1.0.
    
    References
    ----------
    .. [1] Pinnau, R., Totzeck, C., Tse, O., & Martin, S. (2017). A consensus-based model for global optimization and its mean-field limit. 
        Mathematical Models and Methods in Applied Sciences, 27(01), 183-204.

    """

    def __init__(self, x, f, noise,
                 batch_eval: bool = False,
                 alpha: float = 1.0, dt: float = 0.1, sigma: float =1.0, 
                 lamda: float = 1.0,
                 batch_size: int = None,
                 correction: str = None,
                 correction_eps: float = 1e-3,
                 max_eval: int = float('inf'),
                 T: float = 100.) -> None:
        
        super(CBO, self).__init__(
                    x, f, batch_eval=batch_eval,
                    batch_size=batch_size,
                    correction=correction, correction_eps = correction_eps,
                    T = T, max_eval=max_eval)
        
        # additional parameters
        self.dt = dt
        self.alpha = alpha
        self.sigma = sigma
        self.lamda = lamda

        # set noise model
        self.noise = noise
        
    
    def step(self,) -> None:
        r"""Performs one step of the CBO algorithm.

        Parameters
        ----------
        None

        Returns
        -------
        None
        
        """
        x_old = self.x.copy() # save old positions
        self.set_batch_idx() # set batch indices
        self.update_mean() # update mean
        
        self.x[self.batch_idx, :] = self.x[self.batch_idx, :] -\
            self.lamda * self.dt * self.m_diff * self.correction(self) +\
            self.sigma * self.noise(self.m_diff)

        self.update_diff = np.linalg.norm(self.x - x_old)
        self.f_min = np.min(self.energy)
        self.t += self.dt
        
        
    def update_mean(self) -> None:
        r"""Updates the weighted mean of the particles.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        x_batch = self.x[self.batch_idx, :] # get batch
        self.energy = self.f(x_batch) # update energy
        self.num_f_eval += self.batch_size # update number of function evaluations
        
        weights = - self.alpha * self.energy
        coeffs = np.expand_dims(np.exp(weights - logsumexp(weights)), axis=1)
        self.m_alpha = np.sum(x_batch * coeffs, axis=0)
        self.m_diff = x_batch - self.m_alpha
