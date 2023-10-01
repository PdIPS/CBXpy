import numpy as np
import cbx as cbx
from cbx.dynamic.cbo import CBO
from cbx.objectives import Quadratic
from cbx.utils.scheduler import scheduler, multiply

np.random.seed(42)
#%%
conf = {'alpha': 0.01,
        'dt': 0.05,
        'sigma': 0.5,
        'lamda': 1.0,
        'batch_size':50,
        'd': 200,
        'max_it': 5000,
        'N': 50,
        'M': 2}

#%% Define the objective function
f = Quadratic()

#%% Define the initial positions of the particles
x = cbx.utils.init_particles(shape=(conf['M'], conf['N'], conf['d']), x_min=-3., x_max = 3.)

#%% Define the noise function
noise = cbx.noise.comp_noise(dt = conf['dt'])

#%% Define the CBO algorithm
dyn = CBO(f, x=x, noise=noise, f_dim='2D', 
          **conf)
sched = scheduler(dyn, [multiply(name='alpha', factor=1.01, maximum=1e3),
                        multiply(name='sigma', factor=1.005, maximum=3.)])
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