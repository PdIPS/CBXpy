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


def plot_evolution(dyn, num_run = 0, dims = None,
                   wait = 0.5, freq=5,
                   cf_args={}, scx_args={}, scc_args={}):
    dims = dims if not dims is None else [0,1]
    fig, ax = plt.subplots(1,)
    _ = contour_2D(dyn.f, ax=ax, **cf_args)
    
    if not 'x' in dyn.history:
        raise RuntimeError('The dynamic has no particle history!')
    if 'consensus' in dyn.history:
        c = dyn.history['consensus']
        plot_c = True
    if 'drift' in dyn.history:
        d = dyn.history['drift']
        pidx = dyn.history['particle_idx']
        plot_d = True

    x = dyn.history['x']
    scx = ax.scatter(x[0, num_run, :, dims[0]], x[0, num_run, :, dims[1]], **scx_args)
    if plot_c:
        scc = ax.scatter(c[0, num_run, :, dims[0]], c[0, num_run, :, dims[1]], **scc_args)
        
    if plot_d:
        quiver = ax.quiver(x[0, :][pidx[0]][..., dims[0]][num_run,:], x[0, :][pidx[0]][..., dims[1]][num_run,:], 
                           -d[0][num_run, :,dims[0]], -d[0][num_run, :,dims[1]],
                           scale=1.,scale_units='xy', angles='xy', width=0.001,
                           color='orange')
    for i in range(x.shape[0]):
        if i%freq == 0:
            scx.set_offsets(x[i, num_run, ...][:, dims])
            # plot consensus
            if plot_c:
                scc.set_offsets(c[i, num_run, ...][:, dims])
            # plot drift  
            if plot_d:
                quiver.set_offsets(np.array([x[i,...][pidx[i]][...,dims[0]][num_run,:], 
                                             x[i,...][pidx[i]][...,dims[1]][num_run,:]]).T)
                quiver.set_UVC(-d[i][num_run, :, dims[0]], -d[i][num_run, :, dims[1]])
            ax.set_title('Iteration: ' + str(i))
            plt.pause(wait)
        

