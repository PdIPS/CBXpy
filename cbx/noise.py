"""
Noise
======

This module implements some noise models commonly used in consensus 
based methods. 
  
"""

import numpy as np
from abc import ABC, abstractmethod

#%%
class noise_model(ABC):
    """Abstract noise model class
    """

    @abstractmethod
    def __call__(self, m_diff):
        """Call method for classes that inherit from ``noise_model``

        Parameters
        ----------
        m_diff : array_like, shape (J, d) 
            For a system of :math:`J` particles, the i-th row of this array ``m_diff[i,:]`` 
            represents the vector :math:`x_i - \mathsf{m}(x_i)` where :math:`x\in\R^d` denotes 
            the position of the i-th particle and :math:`\mathsf{m}(x_i)` its weighted mean.

        Returns
        -------
        n : array_like, shape (J,d)
            The random vector that is computed by the repspective noise model.
        """

class normal_noise(noise_model):
    r"""Model for normal distributed noise

    This class implements a normal noise model with zero mean and covariance matrix :math:`\dt I_d`
    where :math:`\dt` is a parameter of the class. Given the vector :math:`x_i - \mathsf{m}(x_i)`,
    the noise vector is computed as

    .. math::

        n_i = \dt \sqrt{\|x_i - \mathsf{m}(x_i)\|_2}\ \mathcal{N}(0,1)


    Parameters
    ----------
    dt : float, optional
        The parameter :math:`\dt` of the noise model. The default is 0.1.
    
    Examples
    --------
    >>> import numpy as np
    >>> from polarcbo.noise import normal_noise
    >>> m_diff = np.array([[2,3], [4,5], [1,4.]])
    >>> noise = normal_noise(dt=0.1)
    >>> noise(m_diff)
    array([[-2.4309445 ,  1.34997294],
           [-1.08502177,  0.24030935],
           [ 0.1794014 , -1.09228077]])


    """
    def __init__(self, dt = 0.1):
        self.dt = dt

    def __call__(self, m_diff):
        z = np.sqrt(self.dt) * np.random.normal(0, 1, size=m_diff.shape)
        return z * np.linalg.norm(m_diff, axis=-1,keepdims=True)
    
class comp_noise(noise_model):   
    """Model for componentwise normal distributed noise


    """    
    def __init__(self, dt = 0.1):
        self.dt = dt

    def __call__(self, m_diff):
        """call
        """
        z = np.random.normal(0, np.sqrt(self.dt), size=m_diff.shape) * m_diff
        return z

# %%