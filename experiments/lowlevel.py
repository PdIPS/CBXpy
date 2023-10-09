import numpy as np
import cbx as cbx
from cbx.dynamic import CBO
from cbx.objectives import Rastrigin
from cbx.utils.objective_handling import batched_objective
from cbx.utils.scheduler import scheduler, multiply
from cbx.plotting import plot_evolution

np.random.seed(420)
#%%
conf = {'alpha': 30.0,
        'dt': 0.01,
        'sigma': 8.1,#8,#5.1,#8.0,
        'lamda': 1.0,
        'batch_size':900,
        'batch_partial': False,
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
    @batched_objective
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
plot_evolution(dyn, wait=0.5, freq=1, dims=[0,19], cf_args={'x_min':-3, 'x_max':3})