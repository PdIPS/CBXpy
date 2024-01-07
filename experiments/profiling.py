import numpy as np
import cbx as cbx
from cbx.dynamics import CBO
from cbx.objectives import Rastrigin
from cbx.utils.objective_handling import cbx_objective_fh
from cbx.scheduler import scheduler, multiply
import cProfile
import pstats
import io
from pstats import SortKey
import time
import jax
import jax.numpy as jnp

pr = cProfile.Profile()
pr.enable()
t = time.time()
np.random.seed(420)
#%%
conf = {'alpha': 40.0,
        'dt': 0.01,
        'sigma': 8.1,#8,#5.1,#8.0,
        'lamda': 1.0,
        'batch_args':{
        'batch_size':50,
        'batch_partial': False},
        'd': 20,
        'term_args':{'max_it': 1000},
        'N': 1000,
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
elif mode == 'decorator':
    @cbx_objective_fh
    def f(x):
        return np.linalg.norm(x, axis=-1)
elif mode == 'jax':
    R = Rastrigin()
    f = jax.jit(R)
    @jax.jit
    def f(x):
      return jnp.linalg.norm(x, axis=-1)
#%% Define the initial positions of the particles
x = cbx.utils.init_particles(shape=(conf['M'], conf['N'], conf['d']), x_min=-3., x_max = 3.)

#%% Define the CBO algorithm
dyn = CBO(f, x=x, noise='anisotropic', f_dim='3D', 
          **conf)
sched = scheduler([multiply(name='alpha', factor=1.1, maximum=1e5),
                   #multiply(name='sigma', factor=1.005, maximum=6.)
                   ])
#%% Run the CBO algorithm
while not dyn.terminate():
    dyn.step()
    sched.update(dyn)
    
    if dyn.it%10 == 0:
        print(dyn.f_min)
        print('Alpha: ' + str(dyn.alpha))
        print('Sigma: ' + str(dyn.sigma))
#%%
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(time.time()- t)
print(s.getvalue())