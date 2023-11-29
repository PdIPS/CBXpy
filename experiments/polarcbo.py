from cbx.objectives import snowflake
from cbx.dynamics import PolarCBO
from cbx.plotting import plot_dynamic_history
from cbx.scheduler import multiply
import matplotlib.pyplot as plt
import numpy as np
#%%
np.random.seed(42)
f = snowflake()
dyn = PolarCBO(f, d=2,
          M=2,
          alpha=1.,
          N=100,
          noise='anisotropic',
          sigma=12.,
          kappa=1,
          track_args={'names':[
            'x', 
            'consensus', 
            'drift']},
          batch_args={'size':50})
dyn.optimize(sched = multiply(factor=1.02, maximum=1e12))

#%%
plt.close('all')
plotter = plot_dynamic_history(
    dyn, 
    objective_args={'x_min':-2, 'x_max':2},
    plot_consensus=True,
    plot_drift=True)
plotter.run_plots(freq=1, wait=0.1)