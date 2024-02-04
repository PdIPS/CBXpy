import numpy as np
import cbx as cbx
from cbx.dynamics import CBO
from cbx.objectives import Rastrigin
from cbx.utils.objective_handling import cbx_objective_fh
from cbx.scheduler import scheduler, multiply
from cbx.plotting import plot_dynamic, plot_dynamic_history
import matplotlib.pyplot as plt

np.random.seed(420)
#%%
conf = {'alpha': 40.0,
        'dt': 0.1,
        'sigma': 1.,#8,#5.1,#8.0,
        'lamda': 1.0,
        'batch_args':{
        'batch_size':200,
        'batch_partial': False},
        'd': 2,
        'term_args':{'max_it': 50},
        'N': 50,
        'M': 3,
        'track_args': {'names':
                       ['update_norm', 
                        'energy','x', 
                        'consensus', 
                        'drift']},
        'update_thresh': 0.002}

#%% Define the objective function
mode = 'import'
if mode == 'import':
    f = Rastrigin()
else:
    @cbx_objective_fh
    def f(x):
        return np.linalg.norm(x, axis=-1)

#%% Define the initial positions of the particles
x = cbx.utils.init_particles(shape=(conf['M'], conf['N'], conf['d']), x_min=-2., x_max = 1.)

#%% Define the CBO algorithm
dyn = CBO(f, x=x, noise='isotropic', f_dim='3D', 
          **conf)
sched = scheduler([multiply(name='alpha', factor=1.1, maximum=1e15),
                   #multiply(name='sigma', factor=1.005, maximum=6.)
                   ])
#%% Run the CBO algorithm
plt.close('all')
plotter = plot_dynamic(dyn, 
                       objective_args={'x_min':-3, 'x_max':3},
                       plot_consensus=True,
                       plot_drift=True)
plotter.init_plot()
while not dyn.terminate():
    dyn.step()
    sched.update(dyn)
    
    if dyn.it%10 == 0:
        print(dyn.f_min)
        print('Alpha: ' + str(dyn.alpha))
        print('Sigma: ' + str(dyn.sigma))
        plotter.update(wait=0.5)
    
#%%
plt.close('all')
plotter = plot_dynamic_history(
            dyn, dims=[0,1], 
            objective_args={'x_min':-3, 'x_max':3, 'cmap':'viridis',
                            'num_pts':300},
            particle_args = {'s':50, 'c':'xkcd:sky', 'marker':'o'},
            drift_args = {'color':'pink', 'width':0.003},
            plot_consensus=True,
            plot_drift=True)
plotter.run_plots(wait=0.5, freq=1,)