import numpy as np
from typing import Callable
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from .cbo import CBO

#%% Kernel for PolarCBO
class kernel:
    """Base class for kernels
    
    This class implements kernels for PolarCBO. Every sub-class must implement the ``neg_log`` function
    which is required for the PolarCBO algorithm.
    """

    def __init__(self, kappa=1.):
        self.kappa = kappa

    def __call__(self, x: ArrayLike, y: ArrayLike):
        """Evaluates the kernel
        
        Parameters
        ----------
            x : ArrrayLike
            y : ArrrayLike

        Returns
        -------
            ArrayLike
        """
        raise NotImplementedError('The class ' + self.__class__.__name__ + ' does not implement the call function')

    def neg_log(self, x: ArrayLike, y: ArrayLike):
        """Evaluates the negative logarithm of the kernel
        
        Parameters
        ----------
            x : ArrrayLike
            y : ArrrayLike
        
        Returns
        -------
            ArrayLike
        """
        raise NotImplementedError('The class ' + self.__class__.__name__ + ' does not implement the neg_log function')


#%%
class Gaussian_kernel(kernel):
    r"""Gaussian Kernel

    This class implements a Gaussian kernel, that can be used for PolarCBO.
    ----------
    Arguments:
        kappa (float, optional): The communication radius of the kernel. 
            Using kappa=np.inf yields a constant kernel. Default: 1.0.
    """
    def __init__(self, kappa = 1.0):
        super().__init__(kappa=kappa)
    
    def __call__(self, x: ArrayLike, y: ArrayLike):
        r"""Evaluates the Gaussian Kernel
        
        ..math::

            k(x,y) = \exp(-\frac{1}{2\kappa^2}\|x-y\|_2^2)

        Parameters
        ----------
            x : ArrrayLike
            y : ArrrayLike

        Returns
        -------
            ArrayLike
        """
        dists = ((x-y)**2).sum(tuple(i for i in range(3, x.ndim)))
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.exp(-np.true_divide(1, 2*self.kappa**2) * dists)
    
    def neg_log(self, x, y):        
        return np.true_divide(1, 2*self.kappa**2) * ((x-y)**2).sum(tuple(i for i in range(3, x.ndim)))
    
class Laplace_kernel(kernel):
    """Laplace Kernel

    This class implements a Laplace kernel, that can be used for PolarCBO.
    
    """
    def __init__(self, kappa = 1.0):
        super().__init__(kappa=kappa)
    
    def __call__(self, x,y):
        dists = np.linalg.norm(x-y, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.exp(-np.true_divide(1, self.kappa) * dists)
    
    def neg_log(self, x,y):
        dists = np.linalg.norm(x-y, axis=-1,ord=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.true_divide(1, self.kappa) * dists
        
class Constant_kernel(kernel):
    def __init__(self, kappa = 1.0):
        super().__init__(kappa=kappa)
    
    def __call__(self, x,y):
        dists = np.linalg.norm(x-y, axis=-1)
        dists = dists / self.kappa
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.exp(-dists**np.inf)
    
    def neg_log(self, x,y):
        dists = np.linalg.norm(x-y, axis=-1)
        dists = dists / self.kappa
        with np.errstate(divide='ignore', invalid='ignore'):
            return dists**np.inf
    
class InverseQuadratic_kernel(kernel):
    def __init__(self, kappa = 1.0):
        super().__init__(kappa=kappa)
    
    def __call__(self, x,y):
        dists = np.true_divide(1, self.kappa) * np.linalg.norm(x-y, axis=-1,ord=2)
        return 1/(1+dists**2)
    
    def neg_log(self, x,y):
        dists = np.true_divide(1, self.kappa) * np.linalg.norm(x-y, axis=-1,ord=2)
        return -np.log(1/(1+dists**2))
    
class Taz_kernel(kernel):
    def __init__(self, kappa = 1.0):
        super().__init__(kappa=kappa)

    def __call__(self, x,y):
        return np.exp(-self.neg_log(x,y))
    
    def neg_log(self, x,y):
        dists = np.linalg.norm(x-y, axis=-1,ord=2)
        dists =  np.true_divide(1, self.kappa) *  dists/np.max(dists)
        return  dists**2

#%% PolarCBO
def compute_polar_consensus(energy, x, neg_log_eval, alpha = 1., kernel_factor = 1.):
    weights = -kernel_factor * neg_log_eval - alpha * energy[:,None,:]
    coeffs = np.exp(weights - logsumexp(weights, axis=-1, keepdims=True))
    coeff_expan = tuple([Ellipsis] + [None for i in range(x.ndim-2)])
    c = np.sum(x[:,None,...] * coeffs[coeff_expan], axis=2)
    return c, energy


class PolarCBO(CBO):
    r"""PolarCBO

    This class implements the PolarCBO algorithm as proposed in [1]_.
    
    Parameters
    ----------
    f: objective
        The objective function :math:`f` of the system.
    kernel : optional
        The kernel function :math:`k(\cdot, \cdot)` that is used to compute the particle dependent consensus :math:`c(x_i)`. You can choose from the following options:
            * 'Gaussian': The Gaussian kernel :math:`k(x_i, x_j) = e^{-\frac{1}{2\kappa^2} ||x_i - x_j||^2}`.
            * 'Laplace': The Laplace kernel :math:`k(x_i, x_j) = e^{-\frac{1}{\kappa} ||x_i - x_j||}`
            * 'Constant': The constant kernel :math:`k(x_i, x_j) = \begin{cases} 1, & ||x_i - x_j|| \leq \kappa \\ \infty, & \text{else} \end{cases}` .
            * 'InverseQuadratic': The inverse quadratic kernel :math:`k(x_i, x_j) = \frac{1}{1 + \kappa^{-1} \cdot ||x_i - x_j||^2}`
            * 'Taz': The Taz kernel

        You can also specify a custom class implementing a ``neg_log`` function, i.e., the negative logarithm of the kernel.

    kappa : float, optional
        The kernel parameter :math:`\kappa`.
    kernel_factor_mode : str, optional
        Decides how to scale the kernel, additionally to the factor :math:`\kappa`.
        - 'alpha': the kernel is addittionally multiplied by :math:`\kappa`. Default.
        - 'const': the kernel is not scaled addtionally.
    
    References
    ----------
    .. [1] Bungert, L., Wacker, P., & Roith, T. (2022). Polarized consensus-based 
           dynamics for optimization and sampling. arXiv preprint arXiv:2211.05238.

    """

    def __init__(self,f,
                 kernel = 'Gaussian',
                 kappa: float = 1.0,
                 kernel_factor_mode: str = 'alpha',
                 compute_consensus: Callable = None,
                 **kwargs) -> None:
        super().__init__(f, compute_consensus = compute_consensus if compute_consensus is not None else compute_polar_consensus, **kwargs)
        
        self.kappa = kappa
        self.set_kernel(kernel)
        self.kernel_factor_mode = kernel_factor_mode
    
    kernel_dict = {'Gaussian': Gaussian_kernel,
                   'Laplace': Laplace_kernel,
                   'Constant': Constant_kernel,
                   'InverseQuadratic': InverseQuadratic_kernel,
                   'Taz': Taz_kernel}
    
    def set_kernel(self, kernel):
        """
        Sets the kernel for the model.

        Parameters
        ----------
        kernel (str or object): The kernel to be set. 
            If a string, it must be a key in the kernel_dict attribute of the class. 
            If an object, it will be directly assigned as the kernel.
   
        Returns
        -------
            None

        Raises
        -------
            ValueError: If the provided kernel name is not found in the kernel_dict attribute.
        """
        if isinstance(kernel,str):
            if kernel in self.kernel_dict:
                self.kernel = self.kernel_dict[kernel](kappa=self.kappa)
            else: 
                raise ValueError('Unknown kernel name: ' + 
                                 kernel + '. Choose from: ' + str(self.kernel_dict.keys()))
        else:
            self.kernel = kernel
                
    def kernel_factor(self, ):
        if self.kernel_factor_mode == 'const':
            return 1
        elif self.kernel_factor_mode == 'alpha':
            return self.alpha[self.active_runs_idx, :, None]
        else:
            raise NotImplementedError('Unknown mode: ' + self.kernel_factor_mode)
        
    def compute_consensus(self,):
        x = self.x[self.consensus_idx]
        energy = self.eval_f(x)
        neg_log_eval = self.kernel.neg_log(x[:,None,...], x[:,:,None,...])
        return self._compute_consensus(energy, x, neg_log_eval, alpha = self.alpha[self.active_runs_idx, :, None], kernel_factor = self.kernel_factor())
        