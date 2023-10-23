from cbx.dynamics import CBS
from cbx.objectives import Quadratic
from cbx.plotting import plot_dynamic
import matplotlib.pyplot as plt
#%%
f = Quadratic()
dyn = CBS(f, d=5, M=3, max_it=1000, track_list=['x'],)

dyn.run()
#%%
plt.close('all')
plotter = plot_dynamic(dyn,
                       contour_args={'x_min':-3, 'x_max':3},)
plotter.run_plots(wait=0.05, freq=1)