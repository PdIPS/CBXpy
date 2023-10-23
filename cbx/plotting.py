import matplotlib.pyplot as plt
import numpy as np
import warnings


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


class plot_dynamic:
    def __init__(self, dyn, 
                 num_run = 0, dims = None,
                 ax = None,
                 plot_consensus = False,
                 plot_drift = False,
                 contour_args = None,
                 particle_args = None,
                 cosensus_args = None):
        self.dyn = dyn
        self.dims = dims if dims is not None else [0,1]
        self.num_run = num_run
        
        
        if 'x' not in dyn.history:
            raise RuntimeError('The dynamic has no particle history!')
        self.x = self.dyn.history['x']
        self.max_it = self.x.shape[0] - 1
        
        if plot_consensus:
            if 'consensus' not in dyn.history:
                warnings.warn('The dynamic does not save the consensus points.' +
                              'Ignoring plot_consensus=True!', stacklevel=2)
                plot_consensus = False
            else:
                self.c = dyn.history['consensus']
            
        if plot_drift:
            if 'drift' not in dyn.history:
                warnings.warn('The dynamic does not save the drift.' +
                              'Ignoring plot_drift=True!', stacklevel=2)
                plot_drift = False
            else:
                self.d = dyn.history['drift']
                self.pidx = dyn.history['particle_idx']
            
        self.plot_consensus = plot_consensus
        self.plot_drift = plot_drift
        
        if ax is None:
            fig, ax = plt.subplots(1,)
            
        self.contour_args = contour_args if contour_args is not None else {}
        self.particle_args = particle_args if particle_args is not None else {}
        self.cosensus_args = cosensus_args if cosensus_args is not None else {}
        
        self.ax = ax
        self.init_plot()
        
        
    def init_plot(self,):
        _ = contour_2D(self.dyn.f, ax=self.ax, **self.contour_args)
        self.scx = self.ax.scatter(self.x[0, self.num_run, :, self.dims[0]], 
                              self.x[0, self.num_run, :, self.dims[1]], 
                              **self.particle_args)
        if self.plot_consensus:
            self.scc = self.ax.scatter(self.c[0, self.num_run, :, self.dims[0]], 
                                       self.c[0, self.num_run, :, self.dims[1]], 
                                       **self.cosensus_args)
        if self.plot_drift:
            self.quiver = self.ax.quiver(
                self.x[0, :][self.pidx[0]][..., self.dims[0]][self.num_run,:], 
                self.x[0, :][self.pidx[0]][..., self.dims[1]][self.num_run,:], 
                -self.d[0][self.num_run, :,self.dims[0]], 
                -self.d[0][self.num_run, :,self.dims[1]],
                scale=1., scale_units='xy', angles='xy', 
                width=0.001,color='orange')
    
    def run_plots(self, freq=5, wait=0.1):
        for i in range(self.max_it):
            if i%freq == 0:
                self.plot_at_ind(i)
                self.decorate_at_ind(i)
                plt.pause(wait)
            
    def decorate_at_ind(self,i):
        self.ax.set_title('Iteration: ' + str(i))
        
        
    def plot_at_ind(self, i):
        self.plot_particles_at_ind(i)
        self.plot_consensus_at_ind(i)
        self.plot_drift_at_ind(i)
            
    def plot_particles_at_ind(self, i):
        self.scx.set_offsets(self.x[i, self.num_run, ...][:, self.dims])
        
    def plot_consensus_at_ind(self, i):
        if self.plot_consensus:
            self.scc.set_offsets(self.c[i, self.num_run, ...][:, self.dims])
        
    def plot_drift_at_ind(self, i):
        if self.plot_drift:
            self.quiver.set_offsets(
                np.array([self.x[i,...][self.pidx[i]][..., self.dims[0]][self.num_run,:], 
                          self.x[i,...][self.pidx[i]][..., self.dims[1]][self.num_run,:]]).T)
            self.quiver.set_UVC(
                -self.d[i][self.num_run, :, self.dims[0]], 
                -self.d[i][self.num_run, :, self.dims[1]])

def plot_evolution(dyn, num_run = 0, dims = None,
                   wait = 0.5, freq=5,
                   cf_args=None, scx_args=None, scc_args=None):
    cf_args = cf_args if cf_args is not None else {}
    scx_args = scx_args if scx_args is not None else {}
    scc_args = scc_args if scc_args is not None else {}
    
    dims = dims if dims is not None else [0,1]
    fig, ax = plt.subplots(1,)
    _ = contour_2D(dyn.f, ax=ax, **cf_args)
    plot_c = False
    plot_d = False
    if 'x' not in dyn.history:
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
        

