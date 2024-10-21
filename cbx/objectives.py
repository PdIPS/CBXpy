"""
objectives
==========

This module implements obejective functions to test the performance of consesus 
algorithms.

"""

import numpy as np
from scipy.stats import multivariate_normal
from .utils.objective_handling import cbx_objective
    
#%%

class three_hump_camel(cbx_objective):
    """Three-hump camel function


    Three-hump camel function is a multimodal function with a global minimum at
    :math:`(0,0)`. The function is defined as

    .. math::

        f(x,y) = 2x^2 - 1.05x^4 + \\frac{1}{6}x^6 + xy + y^2

    Examples
    --------
    >>> import numpy as np
    >>> from cbx.objectives import three_hump_camel
    >>> x = np.array([[1,2], [3,4], [5,6.]])
    >>> obj = three_hump_camel()
    >>> obj(x)
    array([   7.11666667,   82.45      , 2063.91666667])
    
    Visualization
    -------------

    .. plot::

        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        from cbx.objectives import three_hump_camel
        fig = plt.figure(figsize=(15,5))
        x_min = -2.
        x_max =  2.
        y_min = -2.
        y_max =  2.
        f = three_hump_camel()

        num_pts_landscape = 100
        xx = np.linspace(x_min, x_max, num_pts_landscape)
        yy = np.linspace(y_min, y_max, num_pts_landscape)
        XX, YY = np.meshgrid(xx,yy)
        XXYY = np.stack((XX.T,YY.T)).T
        Z = np.zeros((num_pts_landscape,num_pts_landscape, 2))
        Z[:,:,0:2] = XXYY
        ZZ = f(Z)

        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection='3d')	
        cs = ax0.contourf(XX,YY,ZZ, 20, cmap=cm.jet)
        ax0.contour(cs, colors='orange', alpha=0.2)
        ax0.plot(f.minima[:, 0], f.minima[:, 1], color='orange', marker='x', markersize=5)
        ax1.plot_surface(XX,YY,ZZ, cmap=cm.jet)
        ax0.set_title('Contour plot')
        ax1.set_title('Surface plot')
    """

    def __init__(self):
        super().__init__()
        self.minima = np.array([[0,0]])

    def apply(self, x):
        return 2*x[..., 0]**2 - 1.05 * x[..., 0]**4 + (1/6) * x[..., 0]**6 + x[..., 0]*x[..., 1] + x[..., 1]**2



class McCormick(cbx_objective):
    r"""McCormick's function
    
    McCormick's function is a multimodal function with two global minima at
    :math:`(-0.54719,-1.54719)` and :math:`(1.54719,0.54719)`. The function is defined as
    

    .. math::

        f(x,y) = \sin(x+y) + (x-y)^2 - 1.5x + 2.5y + 1

    See `McCormick's function <https://en.wikipedia.org/wiki/Test_functions_for_optimization>`_.


    Examples
    --------
    >>> import numpy as np
    >>> from cbx.objectives import McCormick
    >>> x = np.array([[1,2], [3,4], [5,6]])
    >>> f = McCormick()
    >>> f(x)
    array([5.64112001, 8.1569866 , 8.50000979])

    Visualization
    -------------

    .. plot::

        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        from cbx.objectives import McCormick
        fig = plt.figure(figsize=(15,5))
        x_min = -2.
        x_max = 3.
        y_min = -3
        y_max = 4
        f = McCormick()

        num_pts_landscape = 100
        xx = np.linspace(x_min, x_max, num_pts_landscape)
        yy = np.linspace(y_min, y_max, num_pts_landscape)
        XX, YY = np.meshgrid(xx,yy)
        XXYY = np.stack((XX.T,YY.T)).T
        Z = np.zeros((num_pts_landscape,num_pts_landscape, 2))
        Z[:,:,0:2] = XXYY
        ZZ = f(Z)

        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection='3d')	
        cs = ax0.contourf(XX,YY,ZZ, 20, cmap=cm.jet)
        ax0.contour(cs, colors='orange', alpha=0.2)
        ax0.plot(-0.54719,-1.54719, color='orange', marker='x', markersize=10)
        ax0.plot(1.54719, 0.54719, color='orange', marker='x', markersize=10)
        ax1.plot_surface(XX,YY,ZZ, cmap=cm.jet)
        ax0.set_title('Contour plot')
        ax1.set_title('Surface plot')
    """

    def apply(self, x):
        return np.sin(x[..., 0] + x[...,1]) + (x[...,0] - x[...,1])**2 - 1.5 * x[...,0] + 2.5*x[...,1] + 1


class Rosenbrock(cbx_objective):
    """Rosenbrock's function

    Rosenbrock's function is a multimodal function with a global minimum at
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
    >>> from cbx.objectives import Rosenbrock
    >>> x = np.array([[1,2], [3,4], [5,6]])
    >>> f = Rosenbrock()
    >>> f(x)
    array([  0.,  76.,  76.])

    Visualization
    -------------

    .. plot::

        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        from cbx.objectives import Rosenbrock
        fig = plt.figure(figsize=(15,5))
        x_min = -2.
        x_max = 2.
        y_min = -1.
        y_max = 3.
        f = Rosenbrock()

        num_pts_landscape = 150
        xx = np.linspace(x_min, x_max, num_pts_landscape)
        yy = np.linspace(y_min, y_max, num_pts_landscape)
        XX, YY = np.meshgrid(xx,yy)
        XXYY = np.stack((XX.T,YY.T)).T
        Z = np.zeros((num_pts_landscape,num_pts_landscape, 2))
        Z[:,:,0:2] = XXYY
        ZZ = f(Z)

        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection='3d')	
        cs = ax0.contourf(XX,YY,ZZ, 20, cmap=cm.jet)
        ax0.contour(cs, colors='orange', alpha=0.2)
        ax0.plot(1,1, color='orange', marker='x', markersize=10)
        ax1.plot_surface(XX,YY,ZZ, cmap=cm.jet)
        ax0.set_title('Contour plot')
        ax1.set_title('Surface plot')

    """

    def __init__(self, a=1., b=100.):
        super().__init__()
        self.a = a
        self.b = b

    def apply(self, x):
        return (self.a - x[..., 0])**2 + self.b* (x[..., 1] - x[..., 0]**2)**2



class Himmelblau(cbx_objective):
    """Himmelblau's function

    Himmelblau's function is a multimodal function with. The function is defined as 

    .. math::

        f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

    See `Himmelblau's function <https://en.wikipedia.org/wiki/Himmelblau%27s_function>`_.

    Parameters
    ----------
    factor : float, optional    
        The factor by which the input is multiplied. The default is 1.0.

        
    Global minima
    -------------
    - :math:`f(x,y) = 0` at :math:`(x,y) = (3,2)`
    - :math:`f(x,y) = 0` at :math:`(x,y) = (-2.805118,3.131312)`
    - :math:`f(x,y) = 0` at :math:`(x,y) = (-3.779310,-3.283186)`
    - :math:`f(x,y) = 0` at :math:`(x,y) = (3.584428,-1.848126)`


    Examples
    --------
    >>> import numpy as np
    >>> from cbx.objectives import Himmelblau
    >>> x = np.array([[1,2], [3,4], [5,6]])
    >>> f = Himmelblau()
    >>> f(x)
    array([  68.,  148., 1556.])

    Visualization
    -------------

    .. plot::

        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        from cbx.objectives import Himmelblau
        fig = plt.figure(figsize=(15,5))
        x_min = -5.
        x_max = 5.
        y_min = -5.
        y_max = 5.
        f = Himmelblau()

        num_pts_landscape = 250
        xx = np.linspace(x_min, x_max, num_pts_landscape)
        yy = np.linspace(y_min, y_max, num_pts_landscape)
        XX, YY = np.meshgrid(xx,yy)
        XXYY = np.stack((XX.T,YY.T)).T
        Z = np.zeros((num_pts_landscape,num_pts_landscape, 2))
        Z[:,:,0:2] = XXYY
        ZZ = f(Z)

        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection='3d')	
        cs = ax0.contourf(XX,YY,ZZ, 20, cmap=cm.jet)
        ax0.contour(cs, colors='orange', alpha=0.2)
        ax0.scatter(f.minima[:, 0], f.minima[:, 1], color='orange', marker='x', s=15)
        ax1.plot_surface(XX,YY,ZZ, cmap=cm.jet)
        ax0.set_title('Contour plot')
        ax1.set_title('Surface plot')

    """

    def __init__(self, factor=1.0):
        super().__init__()
        self.factor = factor
        self.minima = np.array([[3,2], [-2.805118,3.131312], [-3.779310,-3.283186], [3.584428,-1.848126]])
        
    def apply(self, x):
        x = self.factor*x
        return (x[...,0]**2 + x[...,1] - 11)**2 + (x[...,0] + x[...,1]**2 - 7)**2


class Rastrigin(cbx_objective):
    r"""Rastrigin's function

    Rastrigin's function is a multimodal function with a global minima at
    :math:`(0,0)`. The function is originally defined on :math:`\mathbb{R}^2` as
    
    .. math::

        f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2.

    See `Rastrigin's function <https://en.wikipedia.org/wiki/Rastrigin_function>`_. 
    For our case we employ a shifted version on :math:`\mathbb{R}^d`, where the global minimum is at
    :math:`(b)` and we additonally employ a offset :math:`c`,

    .. math::

        \tilde{f}(x,y)  = \frac{1}{n} \sum_{i=1}^n \left[ (x_i - b)^2 - 10 \cos(2 \pi (x_i - b)) + 10 \right] + c.

    Parameters
    ----------
    b : float, optional
        The first parameter of the function. The default is 0.0.
    c : float, optional
        The second parameter of the function. The default is 0.0.
    
    Examples
    --------
    >>> import numpy as np
    >>> from cbx.objectives import Rastrigin
    >>> x = np.array([[1,2], [3,4], [5,6]])
    >>> f = Rastrigin()
    >>> f(x)
    array([  68.,  148., 1556.])

    Visualization
    -------------

    .. plot::

        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        from cbx.objectives import Rastrigin
        fig = plt.figure(figsize=(15,5))
        x_min = -2.
        x_max =  2.
        y_min = -2.
        y_max =  2.
        f = Rastrigin()

        num_pts_landscape = 100
        xx = np.linspace(x_min, x_max, num_pts_landscape)
        yy = np.linspace(y_min, y_max, num_pts_landscape)
        XX, YY = np.meshgrid(xx,yy)
        XXYY = np.stack((XX.T,YY.T)).T
        Z = np.zeros((num_pts_landscape,num_pts_landscape, 2))
        Z[:,:,0:2] = XXYY
        ZZ = f(Z)

        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection='3d')	
        cs = ax0.contourf(XX,YY,ZZ, 20, cmap=cm.jet)
        ax0.contour(cs, colors='orange', alpha=0.2)
        ax0.plot(f.minima[:, 0], f.minima[:, 1], color='orange', marker='x', markersize=5)
        ax1.plot_surface(XX,YY,ZZ, cmap=cm.jet)
        ax0.set_title('Contour plot')
        ax1.set_title('Surface plot')

    """

    def __init__(self, b=0., c=0.):
        super().__init__()
        self.b = b
        self.c = c
        self.minima = np.array([[self.b, self.b]])
        
    def apply(self, x):
        return (1/x.shape[-1]) * ((x - self.b)**2 - 10*np.cos(2*np.pi*(x - self.b)) + 10).sum(-1) + self.c
            
            
class Rastrigin_multimodal(cbx_objective):
    r"""Multimodal Rastrigin's function
    
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
    >>> from cbx.objectives import Rastrigin_multimodal
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

    def __init__(self, alpha = None, z = None):
        super().__init__()
        self.alpha = alpha if alpha else [1.]
        self.z = z if z else np.zeros((len(self.alpha), 1))

        self.V = Rastrigin()
        self.minima = self.z
        self.num_terms = len(self.alpha)
        
    def apply(self, x):
        y = np.ones(x.shape[0:-1]   )
        for i in range(self.num_terms):
            y *= self.V(self.alpha[i] * (x - self.z[i,:]))
        return y            


class Ackley(cbx_objective):
    r"""Ackley's function

    Ackley's function is a multimodal function with a global minima at
    :math:`(0,0)`. The function is originally defined on :math:`\mathbb{R}^2` as
    
    .. math::

        f(x,y) = -20 \exp \left( -b \sqrt{\frac{1}{2} (x^2 + y^2)} \right) - \exp \left( \frac{1}{2} (\cos(c x) + \cos(c y)) \right) + a + e

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
    >>> from cbx.objectives import Ackley
    >>> x = np.array([[1,2], [3,4], [5,6]])
    >>> f = Ackley()
    >>> f(x)
    array([  68.,  148., 1556.])

    Visualization
    -------------

    .. plot::

        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        from cbx.objectives import Ackley
        fig = plt.figure(figsize=(15,5))
        x_min = -2.
        x_max =  2.
        y_min = -2.
        y_max =  2.
        f = Ackley()

        num_pts_landscape = 100
        xx = np.linspace(x_min, x_max, num_pts_landscape)
        yy = np.linspace(y_min, y_max, num_pts_landscape)
        XX, YY = np.meshgrid(xx,yy)
        XXYY = np.stack((XX.T,YY.T)).T
        Z = np.zeros((num_pts_landscape,num_pts_landscape, 2))
        Z[:,:,0:2] = XXYY
        ZZ = f(Z)

        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection='3d')	
        cs = ax0.contourf(XX,YY,ZZ, 20, cmap=cm.jet)
        ax0.contour(cs, colors='orange', alpha=0.2)
        ax0.plot(f.minima[:, 0], f.minima[:, 1], color='orange', marker='x', markersize=5)
        ax1.plot_surface(XX,YY,ZZ, cmap=cm.jet)
        ax0.set_title('Contour plot')
        ax1.set_title('Surface plot')
    """

    def __init__(self, A=20., b=0.2, c=2*np.pi, minimum = None):
        super().__init__()
        self.A=A
        self.b=b
        self.c=c
        self.minima = 0 if minimum is None else minimum
    
    def apply(self, x):
        d = x.shape[-1]
        x = x - self.minima
        
        arg1 = -self.b * np.sqrt(1/d) * np.linalg.norm(x,axis=-1)
        arg2 = (1/d) * np.sum(np.cos(self.c * x), axis=-1)
        
        return -self.A * np.exp(arg1) - np.exp(arg2) + self.A + np.e


class Ackley_multimodal(cbx_objective):
    r"""Multimodal Ackley's function

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
    >>> from cbx.objectives import Ackley_multimodal
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

    def __init__(self, alpha = None, z = None):
        super().__init__()
        self.alpha = alpha if alpha else [1.]
        self.z = z if z else np.zeros((len(self.alpha), 1))
        self.V = Ackley()
        self.minima = self.z
        self.num_terms = len(self.alpha)
        
    def apply(self, x):
        y = np.ones(x.shape[0:-1]   )
        for i in range(self.num_terms):
            y *= self.V(self.alpha[i] * (x - self.z[i,:]))
        return y
        

# def test2d(x):
#     return np.cos(x.T[0])+np.sin(x.T[1])


class accelerated_sinus(cbx_objective):
    def __init__(self, a=1.0):
        super().__init__()
        self.a = a

    def apply(self, x):
        x = np.linalg.norm(x, axis=-1)
        return np.sin((self.a * x)/(1+x*x)).squeeze() + 1


class nd_sinus(cbx_objective):
    def __init__(self, a=1.0):
        super().__init__()
        self.a = a

    def apply(self, x):
        
        x = 0.3*x
        z = 1/x.shape[-1] * np.linalg.norm(x,axis=-1)**2
        
        
        res = (np.sin(z) + 1) * (x[...,0]**4 - x[...,0]**2 + 1)
        return res.squeeze() 

  
class p_4th_order(cbx_objective):
    def __init__(self,):
        super().__init__()
        
    def apply(self, x):
        n =  x
        
        res = (np.sum(n**4,axis=-1) - np.sum(n**2,axis=-1) + 1)
        return res.squeeze() 


class Quadratic(cbx_objective):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def apply(self, x):
        return np.linalg.norm(self.alpha*x, axis=-1)**2

    
class Banana(cbx_objective):
    def __init__(self, m=0, sigma=0.5, sigma_prior=2):
        super().__init__()
        self.m = m
        self.sigma = sigma
        self.sigma_prior = sigma_prior
    
    def apply(self, x):
        G = ((x[...,1]-1)**2-(x[...,0]-2.5) -1)
        Phi = 0.5/(self.sigma**2)*(G - self.m)**2
        return Phi + 0.5/(self.sigma_prior**2)*np.linalg.norm(x,axis=-1)**2


class Bimodal(cbx_objective):
    def __init__(self, a=None, b=None):
        super().__init__()
        self.a = a if a else [1., 1.5]
        self.b = b if b else [-1.2, -0.7]
    
    def apply(self, x):
        a = self.a
        b = self.b         
        ret = -np.log(np.exp(-((x[...,0]-a[0])**2 + (x[...,1]-a[1])**2/0.2)) \
                      + 0.5*np.exp( -(x[...,0]-b[0])**2/8 - (x[...,1]-b[1])**2/0.5 ))
        return ret
        

class Unimodal(cbx_objective):
    def __init__(self, a = None):
        super().__init__()
        self.a = a if a else [-1.2, -0.7]
    
    def apply(self, x):
        a = self.a
        ret = -np.log(0.5*np.exp( -(x[...,0]-a[0])**2/8 - (x[...,1]-a[1])**2/0.5 ))
        
        return ret
    
class Multimodal(cbx_objective):
    def __init__(self, means=None, covs=None):
        super().__init__()
        self.means = [np.zeros((2,))] if  means is None else means
        self.covs  = [np.eye(2)]     if  covs  is None else covs
        self.mns   = [multivariate_normal(mean=m, cov=c) for m,c in zip(self.means, self.covs)]
        
    def apply(self, x):
        res = 0
        for mn in self.mns:
            res += mn.pdf(x)
        return -np.log(res)
            
    

class Bukin6(cbx_objective):
    r"""Bukin's function 6

    Bunkin's sixth function is a function with many local minima and one global minimum. It is defined as
    
    .. math::

        f(x,y) = 100\sqrt{|y - 0.01x^2|} + 0.01|x + 10|,

    see, e.g., [1]_.

    Parameters
    ----------
    None

    
    Global minima
    -------------
    - :math:`f(x,y) = 0` at :math:`(x,y) = (0,0)`
    
    Examples
    --------
    >>> import numpy as np
    >>> from cbx.objectives import Bukin6
    >>> x = np.array([[1,2], [3,4], [5,6]])
    >>> f = Bukin6()
    >>> f(x)
    array([  68.,  148., 1556.])

    Visualization
    -------------

    .. plot::

        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        from cbx.objectives import Bukin6
        fig = plt.figure(figsize=(15,5))
        x_min = -2.
        x_max =  2.
        y_min = -2.
        y_max =  2.
        f = Bukin6()

        num_pts_landscape = 100
        xx = np.linspace(x_min, x_max, num_pts_landscape)
        yy = np.linspace(y_min, y_max, num_pts_landscape)
        XX, YY = np.meshgrid(xx,yy)
        XXYY = np.stack((XX.T,YY.T)).T
        Z = np.zeros((num_pts_landscape,num_pts_landscape, 2))
        Z[:,:,0:2] = XXYY
        ZZ = f(Z)

        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection='3d')	
        cs = ax0.contourf(XX,YY,ZZ, 20, cmap=cm.jet)
        ax0.contour(cs, colors='orange', alpha=0.2)
        ax0.plot(f.minima[:, 0], f.minima[:, 1], color='orange', marker='x', markersize=5)
        ax1.plot_surface(XX,YY,ZZ, cmap=cm.jet)
        ax0.set_title('Contour plot')
        ax1.set_title('Surface plot')

    References
    ----------
    .. [1] https://www.sfu.ca/~ssurjano/bukin6.html

    """
    def __init__(self,):
        super().__init__()
        self.minima = np.array([[0, 0]])
    
    def apply(self, x):
        return 100 * np.sqrt(np.abs(x[...,1] - 0.01 * x[...,0]**2)) + 0.01 * np.abs(x[...,0] + 10)
    

class cross_in_tray(cbx_objective):
    r"""Cross-In-Tray function

    The Cross-In-Tray function is a function with many local minima and one global minimum [1]_. It is defined as
    
    .. math::

        f(x,y) = -0.0001 \left( \left| \sin(x) \sin(y) \exp \left( \left| 100 - \frac{\sqrt{x^2 + y^2}}{\pi} \right| \right) + 1 \right| + 1 \right)^0.1,

    see [1]_.

    Parameters
    ----------
    None

    
    Global minima
    -------------
    - :math:`f(x,y) = -2.06261` at :math:`(x,y) = (1.34941, 1.34941)`
    - :math:`f(x,y) = -2.06261` at :math:`(x,y) = (-1.34941, -1.34941)`
    - :math:`f(x,y) = -2.06261` at :math:`(x,y) = (1.34941, -1.34941)`
    - :math:`f(x,y) = -2.06261` at :math:`(x,y) = (-1.34941, 1.34941)`
    
    Examples
    --------
    >>> import numpy as np
    >>> from cbx.objectives import cross_in_tray
    >>> x = np.array([[1,2], [3,4], [5,6]])
    >>> f = cross_in_tray()
    >>> f(x)


    Visualization
    -------------

    .. plot::

        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        from cbx.objectives import cross_in_tray
        fig = plt.figure(figsize=(15,5))
        x_min = -2.
        x_max =  2.
        y_min = -2.
        y_max =  2.
        f = cross_in_tray()

        num_pts_landscape = 100
        xx = np.linspace(x_min, x_max, num_pts_landscape)
        yy = np.linspace(y_min, y_max, num_pts_landscape)
        XX, YY = np.meshgrid(xx,yy)
        XXYY = np.stack((XX.T,YY.T)).T
        Z = np.zeros((num_pts_landscape,num_pts_landscape, 2))
        Z[:,:,0:2] = XXYY
        ZZ = f(Z)

        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection='3d')	
        cs = ax0.contourf(XX,YY,ZZ, 20, cmap=cm.jet)
        ax0.contour(cs, colors='orange', alpha=0.2)
        ax0.scatter(f.minima[:, 0], f.minima[:, 1], color='orange', marker='x', s=20)
        ax1.plot_surface(XX,YY,ZZ, cmap=cm.jet)
        ax0.set_title('Contour plot')
        ax1.set_title('Surface plot')

        
    References
    ----------
    .. [1] https://www.sfu.ca/~ssurjano/crossit.html
    """

    def __init__(self):
        super().__init__()
        self.minima = np.array([[1.34941, 1.34941], [-1.34941, 1.34941], [1.34941, -1.34941], [-1.34941, -1.34941]])

    def apply(self, x):
        return -0.0001 * (np.abs(np.sin(x[...,0]) * np.sin(x[...,1]) * np.exp(np.abs(100 - np.sqrt(x[...,0]**2 + x[...,1]**2)/np.pi))) + 1)**0.1
    

class Easom(cbx_objective):
    r"""Easom

    The Easom function is a function with many local minima and one global minimum [1]_ . It is defined as
    
    .. math::

        f(x,y) = -\cos(x) \cos(y) \exp \left( -\left( x - \pi \right)^2 - \left( y - \pi \right)^2 \right).


    Parameters
    ----------
    None

    
    Global minima
    -------------
    - :math:`f(x,y) = -1` at :math:`(x,y) = (\pi, \pi)`
    
    Examples
    --------
    >>> import numpy as np
    >>> from cbx.objectives import Easom
    >>> x = np.array([[1,2], [3,4], [5,6]])
    >>> f = Easom()
    >>> f(x)


    Visualization
    -------------

    .. plot::

        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        from cbx.objectives import Easom
        fig = plt.figure(figsize=(15,5))
        x_min = 0
        x_max =  2. * np.pi
        y_min = 0.
        y_max =  2. * np.pi
        f = Easom()

        num_pts_landscape = 100
        xx = np.linspace(x_min, x_max, num_pts_landscape)
        yy = np.linspace(y_min, y_max, num_pts_landscape)
        XX, YY = np.meshgrid(xx,yy)
        XXYY = np.stack((XX.T,YY.T)).T
        Z = np.zeros((num_pts_landscape,num_pts_landscape, 2))
        Z[:,:,0:2] = XXYY
        ZZ = f(Z)

        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection='3d')	
        cs = ax0.contourf(XX,YY,ZZ, 20, cmap=cm.jet)
        ax0.contour(cs, colors='orange', alpha=0.2)
        ax0.plot(f.minima[:, 0], f.minima[:, 1], color='orange', marker='x', markersize=5)
        ax1.plot_surface(XX,YY,ZZ, cmap=cm.jet)
        ax0.set_title('Contour plot')
        ax1.set_title('Surface plot')

        
    References
    ----------
    .. [1] https://www.sfu.ca/~ssurjano/easom.html
    """

    def __init__(self):
        super().__init__()
        self.minima = np.array([[np.pi, np.pi]])

    def apply(self, x):
        return -np.cos(x[...,0]) * np.cos(x[...,1]) * np.exp(-((x[...,0] - np.pi)**2 + (x[...,1] - np.pi)**2))
    


class drop_wave(cbx_objective):
    r"""Drop Wave

    The Drop Wave function is a function with many local minima and one global minimum [1]_. It is defined as
    
    .. math::

        f(x,y) = -\left( 1 + \cos(12 \sqrt{x^2 + y^2}) \right) \exp \left( -\frac{x^2 + y^2}{2(1 + 0.001(x^2 + y^2))} \right),

    see [1]_.

    Parameters
    ----------
    None

    
    Global minima
    -------------
    - :math:`f(x,y) = -1` at :math:`(x,y) = (0, 0)`
    
    Examples
    --------
    >>> import numpy as np
    >>> from cbx.objectives import drop_wave
    >>> x = np.array([[1,2], [3,4], [5,6]])
    >>> f = drop_wave()
    >>> f(x)


    Visualization
    -------------

    .. plot::

        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        from cbx.objectives import drop_wave
        fig = plt.figure(figsize=(15,5))
        x_min = -2.
        x_max =  2.
        y_min = -2.
        y_max =  2.
        f = drop_wave()

        num_pts_landscape = 100
        xx = np.linspace(x_min, x_max, num_pts_landscape)
        yy = np.linspace(y_min, y_max, num_pts_landscape)
        XX, YY = np.meshgrid(xx,yy)
        XXYY = np.stack((XX.T,YY.T)).T
        Z = np.zeros((num_pts_landscape,num_pts_landscape, 2))
        Z[:,:,0:2] = XXYY
        ZZ = f(Z)

        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection='3d')	
        cs = ax0.contourf(XX,YY,ZZ, 20, cmap=cm.jet)
        ax0.contour(cs, colors='orange', alpha=0.2)
        ax0.plot(f.minima[:, 0], f.minima[:, 1], color='orange', marker='x', markersize=5)
        ax1.plot_surface(XX,YY,ZZ, cmap=cm.jet) 
        ax0.set_title('Contour plot')
        ax1.set_title('Surface plot')

    
    References
    ----------
    .. [1] https://www.sfu.ca/~ssurjano/drop.html
    """
    def __init__(self):
        super().__init__()
        self.minima = np.array([[0, 0]])

    def apply(self, x):
        return -(1 + np.cos(12 * np.sqrt(x[...,0]**2 + x[...,1]**2))) * np.exp(-0.5 * (x[...,0]**2 + x[...,1]**2) / (1 + 0.001 * (x[...,0]**2 + x[...,1]**2)))
    

class Holder_table(cbx_objective):
    r"""Holder table

    The Holder table function is a function with many local minima and four global minima [1]_. It is defined as

    .. math::

        f(x,y) = -\left| \sin(x) \cos(y) \exp \left( \left| 1 - \frac{\sqrt{x^2 + y^2}}{\pi} \right| \right) \right|,

    and its domain is :math:`[-10,10]^2`. Note, that this function can decrease further if the domain is enlarged.

    Parameters
    ----------
    None


    Global minima
    -------------
    - :math:`f(x,y) = -19.2085` at :math:`(x,y) = (8.05502, 9.66459)`
    - :math:`f(x,y) = -19.2085` at :math:`(x,y) = (-8.05502, 9.66459)`
    - :math:`f(x,y) = -19.2085` at :math:`(x,y) = (8.05502, -9.66459)`
    - :math:`f(x,y) = -19.2085` at :math:`(x,y) = (-8.05502, -9.66459)`

    Examples
    --------
    >>> import numpy as np
    >>> from cbx.objectives import Holder_table
    >>> x = np.array([[1,2], [3,4], [5,6]])
    >>> f = Holder_table()
    >>> f(x)


    Visualization
    -------------
    
    .. plot::

        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        from cbx.objectives import Holder_table
        fig = plt.figure(figsize=(15,5))
        x_min = -10.
        x_max =  10.
        y_min = -10.
        y_max =  10.
        f = Holder_table()

        num_pts_landscape = 100
        xx = np.linspace(x_min, x_max, num_pts_landscape)
        yy = np.linspace(y_min, y_max, num_pts_landscape)
        XX, YY = np.meshgrid(xx,yy)
        XXYY = np.stack((XX.T,YY.T)).T
        Z = np.zeros((num_pts_landscape,num_pts_landscape, 2))
        Z[:,:,0:2] = XXYY
        ZZ = f(Z)

        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection='3d')	
        cs = ax0.contourf(XX,YY,ZZ, 20, cmap=cm.jet)
        ax0.contour(cs, colors='orange', alpha=0.2)
        ax0.scatter(f.minima[:, 0], f.minima[:, 1], color='orange', marker='x', s=20)
        ax1.plot_surface(XX,YY,ZZ, cmap=cm.jet) 
        ax0.set_title('Contour plot')
        ax1.set_title('Surface plot')

    
    References
    ----------
    .. [1] https://www.sfu.ca/~ssurjano/holdertable.html

    """

    def __init__(self):
        super().__init__()
        self.minima = np.array([[8.05502, 9.66459], [-8.05502, 9.66459], [8.05502, -9.66459], [-8.05502, -9.66459]])

    def apply(self, x):
        return -np.abs(np.sin(x[...,0]) * np.cos(x[...,1]) * np.exp(np.abs(1 - np.sqrt(x[...,0]**2 + x[...,1]**2) / np.pi)))


class snowflake(cbx_objective):
    r"""Snowflake

    The snowflake function is a function with many local minima and six global minima [1]_. Using polar coordinates, it is as



    .. math::
    
        f(r, \phi) = \min\{f_0(r,\phi), f_1(r,\phi), f_2(r,\phi), 0.8\},


    where for :math:`i\in\{0,1,2\}` we define

    .. math::

        f_i(r,\phi) = r^8 - r^4 + \sqrt{\left|\cos\left(\phi + i\cdot \frac{\pi}{3}\right)\right|} \cdot r^{0.3}.

        
    This function was introduced to showcase the performance of the PolarCBO algorithm [2]_.
    
    Parameters
    ----------
    alpha : float
        Scales the input. Default is .5


    Visualization
    -------------
    
    .. plot::

        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        from cbx.objectives import snowflake
        fig = plt.figure(figsize=(15,5))
        x_min = -2.5
        x_max =  2.5
        y_min = -2.5
        y_max =  2.5
        f = snowflake()

        num_pts_landscape = 100
        xx = np.linspace(x_min, x_max, num_pts_landscape)
        yy = np.linspace(y_min, y_max, num_pts_landscape)
        XX, YY = np.meshgrid(xx,yy)
        XXYY = np.stack((XX.T,YY.T)).T
        Z = np.zeros((num_pts_landscape,num_pts_landscape, 2))
        Z[:,:,0:2] = XXYY
        ZZ = f(Z)

        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection='3d')	
        cs = ax0.contourf(XX,YY,ZZ, 20, cmap=cm.get_cmap('Blues'))
        ax0.contour(cs, colors='white', alpha=0.2)
        ax0.scatter(f.minima[:, 0], f.minima[:, 1], color='blue', marker='x', s=20)
        ax1.plot_surface(XX,YY,ZZ, cmap=cm.get_cmap('Blues')) 
        ax0.set_title('Contour plot')
        ax1.set_title('Surface plot')

    
    References
    ----------
    .. [1] https://github.com/TimRoith/polarcbo
    .. [2] Bungert, L., Roith, T., Wacker, P. (2022): Polarized consensus-based dynamics for optimization and sampling. arXiv:2211.05238

    """


    def __init__(self, alpha=.5):
        super().__init__()
        self.alpha = alpha
        self.minima_polar = np.array([[ 1/self.alpha * 0.5**(1/4), np.pi/2],
                                      [-1/self.alpha * 0.5**(1/4), np.pi/2], 
                                      [ 1/self.alpha * 0.5**(1/4), np.pi/2 - np.pi/3], 
                                      [-1/self.alpha * 0.5**(1/4), np.pi/2 - np.pi/3],
                                      [ 1/self.alpha * 0.5**(1/4), np.pi/2 - 2*np.pi/3],
                                      [-1/self.alpha * 0.5**(1/4), np.pi/2 - 2*np.pi/3]])
        
        self.minima = np.zeros((self.minima_polar.shape))
        self.minima[:, 0] = self.minima_polar[:, 0] * np.cos(self.minima_polar[:, 1])
        self.minima[:, 1] = self.minima_polar[:, 0] * np.sin(self.minima_polar[:, 1])

    def apply(self, x):
        x = self.alpha * x 
        r = np.linalg.norm(x,axis=-1)
        phi = np.arctan2(x[...,1], x[...,0])
        
        res = np.ones((x.shape[:-1]))
        for psi in [0, np.pi/3, np.pi*2/3]:
            g = r**8 - r**4 + np.abs(np.cos(phi+psi))**0.5*r**0.3
            res = np.minimum(res, g)
        
        res = np.minimum(res, .8)
        return res
                

class eggholder(cbx_objective):
    r"""Eggholder

    The Eggholder function is a function with many local minima and one global minimum [1]_. It is defined as

    .. math::
    
        f(x,y) = -(y+47)\cdot \sin\left(\sqrt{\left|y+x/2+47\right|}\right) - x\cdot \sin\left(\sqrt{\left|x-y-47\right|}\right).

        
    
    Parameters
    ----------
    None



    Visualization
    -------------
    
    .. plot::

        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        from cbx.objectives import eggholder
        fig = plt.figure(figsize=(15,5))
        x_min = -600
        x_max =  600
        y_min = x_min
        y_max = x_max
        f = eggholder()

        num_pts_landscape = 200
        xx = np.linspace(x_min, x_max, num_pts_landscape)
        yy = np.linspace(y_min, y_max, num_pts_landscape)
        XX, YY = np.meshgrid(xx,yy)
        XXYY = np.stack((XX.T,YY.T)).T
        Z = np.zeros((num_pts_landscape,num_pts_landscape, 2))
        Z[:,:,0:2] = XXYY
        ZZ = f(Z)

        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection='3d')	
        cs = ax0.contourf(XX,YY,ZZ,30, cmap=cm.jet)
        ax0.contour(cs, colors='orange', alpha=0.2)
        ax0.scatter(f.minima[:, 0], f.minima[:, 1], color='orange', marker='x', s=30)
        ax1.plot_surface(XX,YY,ZZ, cmap=cm.jet) 
        ax0.set_title('Contour plot')
        ax1.set_title('Surface plot')

    
    References
    ----------
    .. [1] https://www.sfu.ca/~ssurjano/egg.html

    """

    def __init__(self):
        super().__init__()
        self.minima = np.array([[512, 404.2319]])

    def apply(self, x):
        return -(x[...,1] + 47) * np.sin(np.sqrt(np.abs(x[...,1] + x[...,0]/2 + 47))) - x[...,0] * np.sin(np.sqrt(np.abs(x[...,0] - (x[...,1] + 47))))
    

class Michalewicz(cbx_objective):
    r"""Michalewicz

    Michalewicz function is a function with many local minima and one global minimum [1]_. It is defined as

    .. math::
    
        f(x,y) = -\sum_{i=1}^d \sin(x_i)\cdot \left(\sin\left(\frac{i x_i^2}{\pi}\right)\right)^{2m},

    where :math:`d` denotes the dimension and the parameter :math:`m` is ususally chosen as :math:`m=10`.

        
    
    Parameters
    ----------
    None



    Visualization
    -------------
    
    .. plot::

        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        from cbx.objectives import Michalewicz
        fig = plt.figure(figsize=(15,5))
        x_min = 0.
        x_max =  4.
        y_min = x_min
        y_max = x_max
        f = Michalewicz()

        num_pts_landscape = 200
        xx = np.linspace(x_min, x_max, num_pts_landscape)
        yy = np.linspace(y_min, y_max, num_pts_landscape)
        XX, YY = np.meshgrid(xx,yy)
        XXYY = np.stack((XX.T,YY.T)).T
        Z = np.zeros((num_pts_landscape,num_pts_landscape, 2))
        Z[:,:,0:2] = XXYY
        ZZ = f(Z)

        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection='3d')	
        cs = ax0.contourf(XX,YY,ZZ,30, cmap=cm.jet)
        ax0.contour(cs, colors='orange', alpha=0.2)
        ax0.scatter(f.minima[:, 0], f.minima[:, 1], color='orange', marker='x', s=30)
        ax1.plot_surface(XX,YY,ZZ, cmap=cm.jet) 
        ax0.set_title('Contour plot')
        ax1.set_title('Surface plot')

    
    References
    ----------
    .. [1] https://www.sfu.ca/~ssurjano/michal.html

    """
        
    def __init__(self, d=2, m=10):
        super().__init__()
        self.d = d
        self.m = m
        
        if d == 2:
            self.minima = np.array([[2.2029, 1.5708]])
        else:
            self.minima = None

    def apply(self, x):
        arr_shape = np.ones(x.ndim, dtype=int)
        arr_shape[-1] = x.shape[-1]
        arr = np.arange(x.shape[-1]).reshape(arr_shape) + 1
        return -np.sum(np.sin(x) * np.sin(arr * (x**2)/np.pi)**(2*self.m), axis=-1)

