import matplotlib.pyplot as plt
import numpy as np


def contour_2D(f, ax = None, num_pts = 50,
                    x_min = -1., x_max = 1.,
                    y_min = None, y_max = None,
                    **kwargs):
    if ax is None:
        ax = plt.gca()
    y_min = x_min if (not y_min) else y_min
    y_max = x_max if (not y_max) else y_max
    d = 2
    x = np.linspace(x_min, x_max, num_pts)
    y = np.linspace(y_min, y_max, num_pts)
    X, Y = np.meshgrid(x,y)
    XY = np.stack((X.T,Y.T)).T
    Z = np.zeros((num_pts, num_pts, d))
    Z[:,:,0:2] = XY
    Z = f(Z)
    cf = ax.contourf(X,Y,Z, **kwargs)
    return cf

