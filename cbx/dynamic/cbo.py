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
        self.set_batch_idx()
        self.x_old = self.copy_particles(self.x) # save old positions
        x_batch = self.x[self.M_idx, self.batch_idx, :] # get batch

        mind = self.get_mean_ind()
        ind = self.get_ind()#
        # first update
        self.m_alpha = self.compute_mean(self.x[mind])        
        self.m_diff = self.x[ind] - self.m_alpha
        
        # inter step
        self.s = self.sigma * self.noise(self.m_diff)

        self.x[ind] = (
            self.x[ind] -
            self.lamda * self.dt * self.m_diff * self.correction(self)[ind] +
            self.s)
        
        self.post_step()
        
        
    def compute_mean(self, x_batch) -> None:
        r"""Updates the weighted mean of the particles.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        e_ind = self.get_mean_ind()[:2]
        self.energy = self.f(x_batch) # update energy
        self.num_f_eval += np.ones(self.M) * self.batch_size # update number of function evaluations
        
        weights = - self.alpha * self.energy#[e_ind]
        coeffs = np.exp(weights - logsumexp(weights, axis=(-1,), keepdims=True))[...,None]
        return (x_batch * coeffs).sum(axis=-2, keepdims=True)