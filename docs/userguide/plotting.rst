Plotting
========

The module :mod:`cbx.plotting` provides some plotting functionality, that help to visualize the dynamic. One has the option to directly plot each iteration as the dynamic runs, or alternatively, save the results in the history of the dynamic and visualize it after it terminated.   

Visualizing during the run
--------------------------

If you want to visualize the dynamic while it runs, you can use the class :class:`plot_dynamic <cbx.plotting.plot_dynamic>`:

    >>> from cbx.dynamics import CBXDynamic
    >>> dyn = CBXDynamic(lambda x:x**2, d=1)
    >>> plotter = plot_dynamic(dyn)
    >>> plotter.init_plot()

Here, it is important to call the method :meth:`init_plot <cbx.plotting.plot_dynamic.init_plot>`, which initializes the necessary parameters for the plotting. After each step of the dynamic, the method :func:`update <cbx.plotting.plot_dynamic.update>` updates the plot, according to changes in the dynamic. The following objects can be plotted:

* the ensemble: a '1D' or '2D' scatter plot of the ensemble. If the array has more than two dimensions, the dimensions specified in ``dims`` are used. In order to be consistent with the drift and the consensus, we always use ``dyn.x_old`` as the ensemble.
* the consensus: a '1D' or '2D' scatter plot of the consensus point. If the array has more than two dimensions, the dimensions specified in ``dims`` are used. You can specify to plot the consensus by setting ``plot_consensus=True``.
* the drift: a quiver plot of the drift. This only works if the dimension is at least two. If the array has more than two dimensions, the dimensions specified in ``dims`` are used. You can specify to plot the drift by setting ``plot_drift=True``.

Visualizing after the run
------------------------

You can also visualize the content of the history, by using the class :class:`plot_dynamic_history <cbx.plotting.plot_history>`. This only works, if you specified to track the ensemble.

    >>> from cbx.dynamics import CBXDynamic
    >>> dyn = CBXDynamic(lambda x:x**2, d=1, track=['x'], max_it = 10)
    >>> dyn.optimize()
    >>> plotter = plot_dynamic(dyn)
    >>> plotter.run_plots()

The class :class:`plot_dynamic_history <cbx.plotting.plot_history>` uses the function :meth:`plot_at_ind <cbx.plotting.plot_history.plot_at_ind>` to plot the content of the history at a certain index. This function can also be used for sliders.