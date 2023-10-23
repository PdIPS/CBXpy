import numpy as np
import cbx as cbx
from cbx.dynamics import CBO
from cbx.objectives import Rastrigin
from cbx.utils.objective_handling import cbx_objective_fh
from cbx.scheduler import scheduler, multiply
from cbx.plotting import plot_dynamic
import matplotlib.pyplot as plt

np.random.seed(420)
#%%
conf = {'alpha': 30.0,
        'dt': 0.01,
        'sigma': 8.1,#8,#5.1,#8.0,
        'lamda': 1.0,
        'batch_args':{
        'batch_size':200,
        'batch_partial': False},
        'd': 20,
        'max_it': 1000,
        'N': 1000,
        'M': 2,
        'track_list': ['update_norm', 'energy','x', 'consensus', 'drift'],
        'resampling': False,
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
x = cbx.utils.init_particles(shape=(conf['M'], conf['N'], conf['d']), x_min=-3., x_max = 3.)

#%% Define the CBO algorithm
dyn = CBO(f, x=x, noise='anisotropic', f_dim='3D', 
          **conf)
sched = scheduler(dyn, [multiply(name='alpha', factor=1.1, maximum=1e15),
                        #multiply(name='sigma', factor=1.005, maximum=6.)
                        ])
#%% Run the CBO algorithm
t = 0
it = 0
while not dyn.terminate():
    dyn.step()
    sched.update()
    
    if it%10 == 0:
        print(dyn.f_min)
        print('Alpha: ' + str(dyn.alpha))
        print('Sigma: ' + str(dyn.sigma))
        
    it+=1
    
#%%
plt.close('all')
plotter = plot_dynamic(dyn, dims=[0,19], 
                       contour_args={'x_min':-3, 'x_max':3},
                       plot_consensus=True,
                       plot_drift=True)
plotter.run_plots(wait=0.05, freq=1)