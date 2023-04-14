"""
Objectives
==========

This module implements obejective functions to test the performance of consesus 
algorithms.

"""

import numpy as np
from abc import ABC, abstractmethod
#%% Objective function

class objective(ABC):
    """Abstract objective function class"""

    @abstractmethod
    def __call__(self, x):
        """Call method for classes that inherit from ``objective``

        Parameters
        ----------
        x : array_like, shape (J, d)
            For a system of :math:`J` particles, the i-th row of this array ``x[i,:]`` 
            represents the position of the i-th particle.
            
        Returns
        -------
        y : array_like, shape (J,)
            The value of the objective function at the positions ``x``.

        """
    

class three_hump_camel(objective):
    """Three-hump camel function

    Three-hump camel function is a multimodal function with 2 global minima at
    :math:`(0,0)` and :math:`(0,0)`. The function is defined as

    .. math::

        f(x,y) = 2x^2 - 1.05x^4 + \\frac{1}{6}x^6 + xy + y^2

    Examples
    --------
    >>> import numpy as np
    >>> from polarcbo.objectives import three_hump_camel
    >>> x = np.array([[1,2], [3,4], [5,6.]])
    >>> obj = three_hump_camel()
    >>> obj(x)
    array([   7.11666667,   82.45      , 2063.91666667])
    """

    def __call__(self, x):
        return 2*x[:,0]**2 - 1.05 * x[:,0]**4 + (1/6) * x[:,0]**6 + x[:,0]*x[:,1] + x[:,1]**2



class McCormick(objective):
    """McCormick's function
    
    McCormick's function is a multimodal function with two global minima at
    :math:`(-0.54719,-1.54719)` and :math:`(1.54719,0.54719)`. The function is defined as
    

    .. math::

        f(x,y) = \sin(x+y) + (x-y)^2 - 1.5x + 2.5y + 1

    See `McCormick's function <https://en.wikipedia.org/wiki/Test_functions_for_optimization>`_.


    Examples
    --------
    >>> import numpy as np
    >>> from polarcbo.objectives import McCormick
    >>> x = np.array([[1,2], [3,4], [5,6]])
    >>> f = McCormick()
    >>> f(x)
    array([5.64112001, 8.1569866 , 8.50000979])

    """

    def __call__(self, x):
        return np.sin(x[:,0] + x[:,1]) + (x[:,0] - x[:,1])**2 - 1.5 * x[:,0] + 2.5*x[:,1] + 1


class Rosenbrock(objective):
    """Rosenbrock's function

    Rosenbrock's function is a multimodal function with 4 global minima at
    :math:`(1,1)`, :math:`(1,1)`, :math:`(1,1)`, and
    :math:`(1,1)`. The function is defined as

    .. math::

        f(x,y) = (a-x)^2 + b(y-x^2)^2

    See `Rosenbrock's function <https://en.wikipedia.org/wiki/Rosenbrock_function>`_.

    Parameters
    ----------
    a : float, optional
        The first parameter of the function. The default is 1.0.
    b : float, optional
        The second parameter of the function. The default is 100.0.

    Examples
    --------
    >>> import numpy as np
    >>> from polarcbo.objectives import Rosenbrock
    >>> x = np.array([[1,2], [3,4], [5,6]])
    >>> f = Rosenbrock()
    >>> f(x)
    array([  0.,  76.,  76.])

    """

    def __init__(self, a=1., b=100.):
        self.a = a
        self.b = b

    def __call__(self, x):
        return (self.a - x[:,0])**2 + self.b* (x[:,1] - x[:,0]**2)**2



class Himmelblau(objective):
    """Himmelblau's function

    Himmelblau's function is a multimodal function with 4 global minima at
    :math:`(3,2)`, :math:`(-2.805118,3.131312)`, :math:`(-3.779310,-3.283186)`, and
    :math:`(3.584428,-1.848126)`. The function is defined as 

    .. math::

        f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

    See `Himmelblau's function <https://en.wikipedia.org/wiki/Himmelblau%27s_function>`_.

    Parameters
    ----------
    factor : float, optional    
        The factor by which the input is multiplied. The default is 1.0.

    Examples
    --------
    >>> import numpy as np
    >>> from polarcbo.objectives import Himmelblau
    >>> x = np.array([[1,2], [3,4], [5,6]])
    >>> f = Himmelblau()
    >>> f(x)
    array([  68.,  148., 1556.])

    """

    def __init__(self, factor=1.0):
        self.factor = factor
        
    def __call__(self, x):
        x = self.factor*x
        return (x[...,0]**2 + x[...,1] - 11)**2 + (x[...,0] + x[...,1]**2 - 7)**2

class Rastrigin(objective):
    r"""Rastrigin's function

    Rastrigin's function is a multimodal function with a global minima at
    :math:`(0,0)`. The function is originally defined on :math:`\mathbb{R}^2` as
    
    .. math::

        f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2.

    See `Rastrigin's function <https://en.wikipedia.org/wiki/Rastrigin_function>`_. 
    For our case we employ a shifted version on :math:`\mathbb{R}^d`, where the global minimum is at
    :math:`(b)` and we additonally employ a offset :math:`c`,

    .. math::

        \tilde{f}(x,y)  = \frac{1}{n} \sum_{i=1}^n (x_i - b)^2 - 10 \cos(2 \pi (x_i - b)) + 10 + c.

    Parameters
    ----------
    b : float, optional
        The first parameter of the function. The default is 0.0.
    c : float, optional
        The second parameter of the function. The default is 0.0.
    
    Examples
    --------
    >>> import numpy as np
    >>> from polarcbo.objectives import Rastrigin
    >>> x = np.array([[1,2], [3,4], [5,6]])
    >>> f = Rastrigin()
    >>> f(x)
    array([  68.,  148., 1556.])

    """

    def __init__(self, b=0., c=0.):
        self.b = b
        self.c = c
        
    def __call__(self, x):
        return (1/x.shape[1]) * np.sum((x - self.b)**2 - \
                10*np.cos(2*np.pi*(x - self.b)) + 10, axis=-1) + self.c
            
            
class Rastrigin_multimodal():
    """Multimodal Rastrigin's function
    
    Let :math:`V` be the Rastrigin's function. Then the multimodal Rastrigin's function is defined as

    .. math::

        f(x) = \prod_{i=1}^n V(\alpha_i (x - z_i))

    Parameters
    ----------
    alpha : list of floats, optional
        The factor for each multiplicative term. The default is [1.0].
    z : numpy array, optional
        The shift vectors in each term. The default is np.array([[0]]).

    Examples
    --------
    >>> import numpy as np
    >>> from polarcbo.objectives import Rastrigin_multimodal
    >>> x = np.array([[1,2], [3,4], [5,6]])
    >>> alpha = [2., 3.]
    >>> z = np.array([[2,3], [4,5]])
    >>> f = Rastrigin_multimodal(alpha = alpha, z = z)
    >>> f(x)
    array([324.,  36., 324.])

    See Also
    --------
    Rastrigin : The Rastrigin's function
    Ackley_multimodal : The multimodal Ackley's function

    """

    def __init__(self, alpha = [1.], z = np.array([[0]])):
        self.V = Rastrigin()
        self.alpha = alpha
        self.z = z
        self.minima = z
        self.num_terms = len(alpha)
        
    def __call__(self, x):
        y = np.ones(x.shape[0:-1]   )
        for i in range(self.num_terms):
            y *= self.V(self.alpha[i] * (x - self.z[i,:]))
        return y            

class Ackley():
    r"""Ackley's function

    Ackley's function is a multimodal function with a global minima at
    :math:`(0,0)`. The function is originally defined on :math:`\mathbb{R}^2` as
    
    .. math::

        f(x,y) = -20 \exp \left( -b \sqrt{\frac{1}{2} (x^2 + y^2)} \right) - \exp \left( \frac{1}{2} \cos(c x) + \cos(c y) \right) + a + e

    See `Ackley's function <https://en.wikipedia.org/wiki/Ackley_function>`_.

    Parameters
    ----------
    a : float, optional
        The default is 20.0.
    b : float, optional
        The default is 0.2.
    c : float, optional
        The default is 2*np.pi.
    
    Examples
    --------
    >>> import numpy as np
    >>> from polarcbo.objectives import Ackley
    >>> x = np.array([[1,2], [3,4], [5,6]])
    >>> f = Ackley()
    >>> f(x)
    array([  68.,  148., 1556.])
    """

    def __init__(self, a=20., b=0.2, c=2*np.pi):
        self.a=a
        self.b=b
        self.c=c
    
    def __call__(self, x):
        d = x.shape[-1]
        
        arg1 = -self.b * np.sqrt(1/d) * np.linalg.norm(x,axis=-1)
        arg2 = (1/d) * np.sum(np.cos(self.c * x), axis=-1)
        
        return -self.a * np.exp(arg1) - np.exp(arg2) + self.a + np.e

class Ackley_multimodal():
    """Multimodal Ackley's function

    Let :math:`V` be the Ackley's function. Then the multimodal Ackley's function is defined as

    .. math::

        f(x) = \prod_{i=1}^n V(\alpha_i (x - z_i))

    Parameters
    ----------
    alpha : list of floats, optional
        The factor for each multiplicative term. The default is [1.0].
    z : numpy array, optional
        The shift vectors in each term. The default is np.array([[0]]).

    Examples
    --------
    >>> import numpy as np
    >>> from polarcbo.objectives import Ackley_multimodal
    >>> x = np.array([[1,2], [3,4], [5,6]])
    >>> alpha = [2., 3.]
    >>> z = np.array([[2,3], [4,5]])
    >>> f = Ackley_multimodal(alpha = alpha, z = z)
    >>> f(x)
    array([110.07368964,  59.49910362, 126.11721609])

    See Also
    --------
    Ackley
    Rasrigin_multimodal

    """

    def __init__(self, alpha = [1.], z = np.array([[0]])):
        self.V = Ackley()
        self.alpha = alpha
        self.z = z
        self.minima = z
        self.num_terms = len(alpha)
        
    def __call__(self, x):
        y = np.ones(x.shape[0:-1]   )
        for i in range(self.num_terms):
            y *= self.V(self.alpha[i] * (x - self.z[i,:]))
        return y
        
class test2d():
    def __init__(self):
        return
    
    def __call__(self, x):
        return np.cos(x.T[0])+np.sin(x.T[1])


        
class accelerated_sinus():
    def __init__(self, a=1.0):
        self.a = a

    def __call__(self, x):
        return np.sin((self.a * x)/(1+x*x)).squeeze() + 1
    
class nd_sinus():
    def __init__(self, a=1.0):
        self.a = a

    def __call__(self, x):
        
        x = 0.3*x
        z = 1/x.shape[-1] * np.linalg.norm(x,axis=-1)**2
        
        
        res = (np.sin(z) + 1) * (x[...,0]**4 - x[...,0]**2 + 1)
        return res.squeeze() 
    
class p_4th_order():
    def __init__(self,):
        pass

    def __call__(self, x):
        #n = np.sqrt(1/x.shape[-1]) * np.linalg.norm(x, axis=-1)
        #n = 1/x.shape[-1] *np.sum(x, axis = -1)
        n =  x
        
        #res = (n**4 - n**2 + 1)
        res = (np.sum(n**4,axis=-1) - np.sum(n**2,axis=-1) + 1)
        return res.squeeze() 
    
class Quadratic():
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x):
        return np.linalg.norm(self.alpha*x, axis=-1)**2
    
class Banana():
    def __init__(self, m=0, sigma=0.5, sigma_prior=2):
        self.m = m
        self.sigma = sigma
        self.sigma_prior = sigma_prior
    
    def __call__(self, x):
        G = ((x[...,1]-1)**2-(x[...,0]-2.5) -1)
        Phi = 0.5/(self.sigma**2)*(G - self.m)**2
        I = Phi + 0.5/(self.sigma_prior**2)*np.linalg.norm(x,axis=-1)**2
        
        return I

class Bimodal():
    def __init__(self, a=[1., 1.5], b=[-1.2, -0.7]):
        self.a = a
        self.b = b
    
    def __call__(self, x):
        a = self.a
        b = self.b         
        ret = -np.log(np.exp(-((x[...,0]-a[0])**2 + (x[...,1]-a[1])**2/0.2)) \
                      + 0.5*np.exp( -(x[...,0]-b[0])**2/8 - (x[...,1]-b[1])**2/0.5 ))
        return ret
        

class Unimodal():
    def __init__(self, a=[-1.2, -0.7]):
        self.a = a
    
    def __call__(self, x):
        a = self.a
        ret = -np.log(0.5*np.exp( -(x[...,0]-a[0])**2/8 - (x[...,1]-a[1])**2/0.5 ))
        
        return ret
                