import numpy as np
import cbx as cbx
from cbx.dynamics import CBO
from cbx.objectives import Quadratic
from cbx.utils.objective_handling import cbx_objective_fh
import timeit
import jax
import jax.numpy as jnp
import numba

np.random.seed(42)
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
        'M': 10,
        'track_args': {'names':
                       ['update_norm', 
                        'energy','x', 
                        'consensus', 
                        'drift']},
        'update_thresh': 0.002}

#%% Define the objective function
@cbx_objective_fh
def norm_dec(x):
    return np.linalg.norm(x, axis=-1)
@jax.jit
def norm_jnp(x):
  return jnp.linalg.norm(x, axis=-1)

@numba.jit(nopython=True)
def norm_numba(x):
    out = np.zeros(x.shape[:2])
    for m in range(x.shape[0]):
        for n in range(x.shape[1]):
            out[m,n] = np.linalg.norm(x[m,n,:])
    return out


obs = {'CBX.objectives': Quadratic(), 
       'Decorated':norm_dec, 
       'jnp': norm_jnp,
       'numba': norm_numba}
#%% Define the initial positions of the particles
x = cbx.utils.init_particles(shape=(conf['M'], conf['N'], conf['d']), x_min=-3., x_max = 3.)

#%% Define the CBO algorithm

for f in obs.keys():
    dyn = CBO(obs[f], x=x, noise='anisotropic', f_dim='3D', 
              **conf)
    T = timeit.Timer(dyn.step)
    rep = 10
    r = T.repeat(rep, number=50)
    best = min(r)
    print(f+ ' Best of ' +str(rep) + ': ' +str(best)[:6] + 's')

