"""
This module implements the noise methods for cbx dynamics
"""
from typing import Callable
from numpy.typing import ArrayLike
from numpy.random import normal
import numpy as np

def get_noise(name: str, dyn):
    if name == 'isotropic':
        return isotropic_noise(norm=dyn.norm, sampler=dyn.sampler)
    elif name == 'anisotropic':
        return anisotropic_noise(norm=dyn.norm, sampler=dyn.sampler)
    elif name == 'covariance' or name == 'sampling':
        return covariance_noise(norm=dyn.norm, sampler=dyn.sampler)
    else:
        raise NotImplementedError('Noise model {} not implemented'.format(name))


class noise:
    def __init__(self, 
                 norm: Callable = None, 
                 sampler: Callable = None):
        self.norm = norm if norm is not None else np.linalg.norm
        self.sampler = sampler if sampler is not None else normal
    def __call__(self, dyn):
        """
        This function returns the noise vector for a given dynamic object. This is the function that is called in the dynamic object.
        Therfore, it has to have a fixed signature, accepting a dynamic object as input. The concrete implementation of this function
        is provided by each specific noise model.

        Parameters
        ----------
        dyn
            The dynamic object
        
        Returns
        -------
        ArrayLike
            The noise vector
        """
        raise NotImplementedError('Base class does not implement __call__')
    
    def sample(self, ):
        """
        This function performs the sampling of the noise vector. Each specific noise model must implement this function.
        """
        raise NotImplementedError('Base class does not implement sample')

class isotropic_noise(noise):
    r"""

    This class implements the isotropic noise model. From the drift :math:`d = x - c(x)`,
    the noise vector is computed as

    .. math::

        n_{m,n} = \sqrt{dt}\cdot \|d_{m,n}\|_2\cdot \xi.


    Here, :math:`\xi` is a random vector of size :math:`(d)` distributed according to :math:`\mathcal{N}(0,1)`.
    
    Parameters
    ----------
    None
    
    Note
    ----
    Only the norm of the drift is used for the noise. Therefore, the noise vector is scaled with the same factor in each dimension, 
    which motivates the name **isotropic**. 
    """
    def __init__(self, 
                 norm: Callable = None, 
                 sampler: Callable = None):
         super().__init__(norm = norm, sampler = sampler)

    def __call__(self, dyn) -> ArrayLike:
        return np.sqrt(dyn.dt) * self.sample(dyn.drift)

    def sample(self, drift) -> ArrayLike:
        r'''

        This function implements the isotropic noise model. From the drift :math:`d = x - c(x)`,
        the noise vector is computed as

        .. math::

            n_{m,n} = \sqrt{dt}\cdot \|d_{m,n}\|_2\cdot \xi.


        Here, :math:`\xi` is a random vector of size :math:`(d)` distributed according to :math:`\mathcal{N}(0,1)`.
        
        Parameters
        ----------
        None
        
        Note
        ----
        Only the norm of the drift is used for the noise. Therefore, the noise vector is scaled with the same factor in each dimension, 
        which motivates the name **isotropic**. 
        '''
        z = self.sampler(size = drift.shape)
        return z * self.norm(drift.reshape(*drift.shape[:2],-1), axis=-1).reshape(drift.shape[:2] + (drift.ndim-2)*(1,)) 
    


class anisotropic_noise(noise):
        r"""
        This class implements the anisotropic noise model. From the drift :math:`d = x - c(x)`,
        the noise vector is computed as

        .. math::

            n_{m,n} = \sqrt{dt}\cdot d_{m,n} \cdot \xi.

        Here, :math:`\xi` is a random vector of size :math:`(d)` distributed according to :math:`\mathcal{N}(0,1)`.

        Returns:
            numpy.ndarray: The generated noise.

        Note
        ----
        The plain drift is used for the noise. Therefore, the noise vector is scaled with a different factor in each dimension, 
        which motivates the name **anisotropic**.
        """

        def __init__(self, 
                     norm: Callable = None,
                     sampler: Callable = None):
            super().__init__(norm = norm, sampler = sampler)

        def __call__(self, dyn) -> ArrayLike:
            return np.sqrt(dyn.dt) * self.sample(dyn.drift)

        def sample(self, drift: ArrayLike) -> ArrayLike:
            r"""

            This function implements the anisotropic noise model. From the drift :math:`d = x - c(x)`,
            the noise vector is computed as

            .. math::

                n_{m,n} = \sqrt{dt}\cdot d_{m,n} \cdot \xi.

            Here, :math:`\xi` is a random vector of size :math:`(d)` distributed according to :math:`\mathcal{N}(0,1)`.

            Returns:
                numpy.ndarray: The generated noise.

            Note
            ----
            The plain drift is used for the noise. Therefore, the noise vector is scaled with a different factor in each dimension, 
            which motivates the name **anisotropic**.
            """

            return self.sampler(size = drift.shape) * drift
        
class covariance_noise(noise):
        r"""

        This class implements the covariance noise model. Given the covariance matrix :math:`\text{Cov}(x)\in\mathbb{R}^{M\times d\times d}` of the ensemble,
        the noise vector is computed as

        .. math::

            n_{m,n} = \sqrt{(1/\lambda)\cdot (1-\exp(-dt))^2} \cdot \sqrt{\text{Cov}(x)}\xi.

        Here, :math:`\xi` is a random vector of size :math:`(d)` distributed according to :math:`\mathcal{N}(0,1)`.
        """
        
        def __init__(self, 
                     norm: Callable = None,
                     sampler: Callable = None,
                     mode = 'sampling'):
            super().__init__(norm = norm, sampler = sampler)
            self.mode = mode

        def __call__(self, dyn) -> ArrayLike:
             dyn.update_covariance()
             #factor = np.sqrt((1/dyn.lamda) * (1 - np.exp(-dyn.dt)**2))[(...,) + (None,) * (dyn.x.ndim - 2)]
             factor = np.sqrt((2*dyn.dt)/self.lamda(dyn))[(...,) + (None,) * (dyn.x.ndim - 2)]
             return factor * self.sample(dyn.drift, dyn.Cov_sqrt)
         
        def lamda(self, dyn):
            if self.mode == 'sampling':
                return 1/(1 + dyn.alpha)
            else:
                return 1
        
        def sample(self, drift:ArrayLike, Cov_sqrt:ArrayLike) -> ArrayLike:
            r"""
            This function implements the covariance noise model. 
            Given the covariance matrix :math:`\text{Cov}(x)\in\mathbb{R}^{M\times d\times d}` of the ensemble,
            the noise vector is computed as

            .. math::

                n_{m,n} = \sqrt{(1/\lambda)\cdot (1-\exp(-dt))^2} \cdot \sqrt{\text{Cov}(x)}\xi.

            Here, :math:`\xi` is a random vector of size :math:`(d)` distributed according to :math:`\mathcal{N}(0,1)`.

            Parameters
            ----------
                drift (ArrayLike): The drift of the ensemble.
                Cov_sqrt (ArrayLike): The square root of the covariance matrix of the ensemble.

            Returns
            -------
                ArrayLike: The covariance noise.
            
            """

            z = self.sampler(size = drift.shape) 
            return self.apply_cov_sqrt(Cov_sqrt, z)
        
        def apply_cov_sqrt(self, Cov_sqrt: ArrayLike, z:ArrayLike) -> ArrayLike:
            """
            Applies the square root of the covariance matrix to the input tensor.

            Parameters
            ----------
                Cov_sqrt (ArrayLike): The square root of the covariance matrix.
                z (ArrayLike): The input tensor of shape (batch_size, num_features, seq_length).

            Returns:
                ArrayLike: The output of the matrix-vector product.
            """
            if Cov_sqrt.ndim == z.ndim:
                return np.einsum('kij,klj->kli', Cov_sqrt, z)
            elif Cov_sqrt.ndim == z.ndim + 1:
                return np.einsum('klij,klj->kli', Cov_sqrt, z)
            else:
                raise RuntimeError('Shape mismatch between Cov_sqrt and sampled vector!')
    
def exponential_sampler():
    """The function returns the exponential sampler."""
    def _exponential_sampler(size=None):
        x = np.random.exponential(1.0, size)
        sign = np.random.choice([-1, 1], size)
        x *= sign
        return x
    return _exponential_sampler
