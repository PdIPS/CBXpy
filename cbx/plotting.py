import matplotlib.pyplot as plt
import numpy as np
import warnings

def plot_1D(f, ax = None, num_pts = 200,
            x_min = -1., x_max = 1.,
            **kwargs):
    if ax is None:
        ax = plt.gca()
    x = np.linspace(x_min, x_max, num_pts)[None,:,None]
    y = f(x)[0,...]
    ax.plot(x[0,:,0], y, **kwargs)


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
    r"""plot_dynamic

    Plots particles, consensus, and drift of the specified dynamic.

    Parameters
    ----------
    dyn
        The dynamic object to plot.
    num_run: int
        The index of the run to plot. Defaults to 0.
    dims
        The dimensions to plot. Defaults to [0,1].
    ax
        The axis to plot on. If None is specified, a new axis is created.
    plot_consensus: bool
        Whether to plot the consensus points. Defaults to False.
    plot_drift: bool
        Whether to plot the drift. Defaults to False.
    eval_energy_1d: bool
        Whether to evaluate the energy of the dynamic in 1D. Defaults to True.
    objective_args
        Additional arguments to pass to contour_2D, which creates a contour plot of ``dyn.f``.
    particle_args
        Additional arguments to pass to ax.scatter, which plots the particles.
    cosensus_args
        Additional arguments to pass to ax.scatter, which plots the consensus points.

    """
    def __init__(self,
                 dyn,
                 num_run = 0, dims = None,
                 ax = None,
                 plot_consensus = False,
                 plot_drift = False,
                 eval_energy_1d = True,
                 objective_args = None,
                 particle_args = None,
                 cosensus_args = None):
        
        self.dyn = dyn
        self.d = dyn.d
        self.dims = dims if dims is not None else [0,1]
        self.num_run = num_run
            
        self.plot_consensus = plot_consensus
        self.plot_drift = plot_drift
        self.eval_energy_1d = eval_energy_1d
        
        if ax is None:
            fig, ax = plt.subplots(1,)
            
        self.objective_args = objective_args if objective_args is not None else {}
        self.particle_args = particle_args if particle_args is not None else {}
        self.cosensus_args = cosensus_args if cosensus_args is not None else {}

        xmin = self.objective_args.get('x_min', -1.)
        xmax = self.objective_args.get('x_max', 1.)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([xmin, xmax])
        
        self.ax = ax
        
    def init_plot(self,) -> None:
        """
        Initializes the plot for visualizing the data.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.plot_objective()
        self.init_x(self.dyn.x)

        if self.plot_consensus:
            if self.dyn.consensus is None:
                c = self.dyn.x
            else:
                c = self.dyn.consensus
            self.init_consensus(c)

        if self.plot_drift:
            if not hasattr(self.dyn, 'drift') or self.dyn.drift is None:
                dr = self.dyn.x
                pidx = Ellipsis
            else:
                dr = self.dyn.drift
                pidx = self.dyn.particle_idx

            self.init_drift(self.dyn.x, dr, pidx)

    def plot_objective(self):
        """
        Plots the objective function.

        """
        
        if self.d == 1:
            _ = plot_1D(self.dyn.f, ax=self.ax, **self.objective_args)
        else:
             _ = contour_2D(self.dyn.f, ax=self.ax, **self.objective_args)
    def init_x(self, x):
        """
        Initializes the particle plot.

        Parameters
        ----------
        x
            The particles to plot.

        Returns
        -------
        None
        """

        if self.d == 1:
            if self.eval_energy_1d:
                y = self.dyn.f(x[self.num_run, :, self.dims[0]][None,:,None])[...,None]
            else:
                y = np.zeros_like(x[self.num_run, :, self.dims[0]])
        else:
            y = x[self.num_run, :, self.dims[1]]

        self.scx = self.ax.scatter(
        x[self.num_run, :, self.dims[0]], 
        y, 
        **self.particle_args)
        
    def init_consensus(self, c):
        """
        Initializes the consensus plot.

        Parameters
        ----------
        c
            The consensus points to plot.
        
        Returns
        -------
        None
        """

        if self.d == 1:
            y = np.zeros_like(c[self.num_run, :, self.dims[0]])
        else:
            y = c[self.num_run, :, self.dims[1]]

        self.scc = self.ax.scatter(
            c[self.num_run, :, self.dims[0]], 
            y,
            **self.cosensus_args)
        
    def init_drift(self, x, dr, pidx):
        """
        Initializes the drift plot.
        
        Parameters
        ----------
        x
            The particles to plot.
        dr
            The drift to plot.
        pidx
            The indices of the particles to plot.
            
        Returns
        -------
        None
        """
        
        if self.d == 1:
            warnings.warn('Drift plot not supported for 1D data.', stacklevel=2)
            self.plot_drift = False
        else:
            self.quiver = self.ax.quiver(
                x[pidx][..., self.dims[0]][self.num_run,:], 
                x[pidx][..., self.dims[1]][self.num_run,:], 
                -dr[self.num_run, :, self.dims[0]], 
                -dr[self.num_run, :, self.dims[1]],
                scale=1., scale_units='xy', angles='xy', 
                width=0.001,color='orange')
        
    def update(self, wait=0.1):
        """
        Updates the plot.

        Parameters
        ----------
        wait
            The time to wait between frames. Defaults to 0.1.
        
        Returns
        -------
        None
        """

        self.plot_particles(self.dyn.x_old)
        if self.plot_consensus:
            self.plot_c(self.dyn.consensus)
        if self.plot_drift:
            self.plot_d(self.dyn.x_old, self.dyn.drift, self.dyn.particle_idx)

        plt.pause(wait)
        
    def plot_particles(self, x):
        """
        Plots particles.
        
        Parameters
        ----------
        x
            The particles to plot.
        
        Returns
        -------
        None
        """
        if self.d == 1:
            if self.eval_energy_1d:
                y = self.dyn.f(x[self.num_run, :, self.dims[0]][None,:,None])[0,...]
            else:
                y = np.zeros_like(x[self.num_run, ...][:, self.dims[0]])
            z = np.stack([x[self.num_run, ...][:, 0], y], axis=-1)
        else:
            z = x[self.num_run, ...][:, self.dims]

        self.scx.set_offsets(z)
        
    def plot_c(self, c):
        """
        Plots consensus.

        Parameters
        ----------
        c
            The consensus to plot.

        Returns
        -------
        None
        """

        if self.plot_consensus and c is not None:
            if self.d == 1:
                z = np.stack([c[self.num_run, ...][:, 0],
                              np.zeros_like(c[self.num_run, ...][:, 0])], axis=-1)
            else:
                z = c[self.num_run, ...][:, self.dims]

            self.scc.set_offsets(z)
        
    def plot_d(self, x, dr, pidx):
        """
        Plots drift.

        Parameters
        ----------
        x
            The particles to plot.
        dr
            The drift to plot.
        pidx
            The indices of the particles to plot.
        
        Returns
        -------
        None
        """
        
        if pidx is None:
            pidx = Ellipsis

        if self.plot_drift:
            self.quiver.set_offsets(
                np.array([x[pidx][..., self.dims[0]][self.num_run, :], 
                          x[pidx][..., self.dims[1]][self.num_run, :]]).T)
            self.quiver.set_UVC(
                -dr[self.num_run, :, self.dims[0]], 
                -dr[self.num_run, :, self.dims[1]])
            
            

class plot_dynamic_history(plot_dynamic):
    """plot_dynamic_history
    
    Visualize a dynamic from its history.

    Parameters
    ----------
    dyn (object): The dynamic object.
    **kwargs: Additional keyword arguments. See `plot_dynamic` for details.

    Raises
    ------
        RuntimeError: If the dynamic object has no particle history.

    Warnings
    --------
        If plot_consensus is True and the dynamic object does not save the consensus points.
        If plot_drift is True and the dynamic object does not save the drift.
    """

    def __init__(self, dyn, **kwargs):
        super().__init__(dyn, **kwargs)

        if 'x' not in dyn.history:
            raise RuntimeError('The dynamic has no particle history!')
        self.x = self.dyn.history['x']
        self.max_it = len(self.x) - 1
        
        if self.plot_consensus:
            if 'consensus' not in dyn.history:
                warnings.warn('The dynamic does not save the consensus points.' +
                              'Ignoring plot_consensus=True!', stacklevel=2)
                self.plot_consensus = False
            else:
                self.c = dyn.history['consensus']
            
        if self.plot_drift:
            if 'drift' not in dyn.history:
                warnings.warn('The dynamic does not save the drift.' +
                              'Ignoring plot_drift=True!', stacklevel=2)
                self.plot_drift = False
            else:
                self.dr = dyn.history['drift']
                self.pidx = dyn.history['particle_idx']
        
        self.init_plot()

    
    def init_plot(self,):
        """
        Initializes the plot for visualizing the dynamic.

        """

        self.plot_objective()
        self.init_x(self.x[0])
        if self.plot_consensus:
            self.init_consensus(self.c[0])

        if self.plot_drift:
            self.init_drift(self.x[0], self.dr[0], self.pidx[0])

    def plot_at_ind(self, i):
        """
        Plots the dynamic at iteration i.

        Parameters
        ----------
        i (int): The iteration to plot.

        Returns
        -------
        None
        """

        self.plot_particles(self.x[i])

        if self.plot_consensus:
            self.plot_c(self.c[i])
        if self.plot_drift:
            self.plot_d(self.x[i], self.dr[i], self.pidx[i])

    def decorate_at_ind(self,i):
        """
        Decorates the plot at iteration i.

        Parameters
        ----------
        i (int): The iteration at which to decorate.

        Returns
        -------
        None

        """
        self.ax.set_title('Iteration: ' + str(i))

    def run_plots(self, freq=5, wait=0.1):
        """
        Visualizes the evolution of the dynamic over time, using the history

        Parameters:
            freq (int): The frequency at which plots should be generated. Default is 5.
            wait (float): The amount of time to pause between plots. Default is 0.1.

        Returns:
            None
        """
        for i in range(self.max_it):
            if i%freq == 0:
                self.plot_at_ind(i)
                self.decorate_at_ind(i)
                plt.pause(wait)
        
        

