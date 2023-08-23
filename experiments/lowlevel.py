import numpy as np

from sys import path
path.append('..')

import cbx as cbx
from cbx.dynamic.cbo import CBO
from cbx.objectives import Quadratic
from cbx.scheduler import scheduler, multiply

np.random.seed(42)
#%%
conf = cbx.utils.config(**{
        'alpha': 0.01,
        'dt': 0.01,
        'sigma': 0.5,
        'lamda': 1.0,
        'batch_size':50,
        'check_list': ['max_time'],
        'd': 200,
        'T': 20,
        'N': 50,
        'M': 2})

#%% Define the objective function
f = Quadratic()

#%% Define the initial positions of the particles
x = cbx.utils.init_particles(shape=(conf.M, conf.N, conf.d), x_min=-3., x_max = 3.)

#%% Define the noise function
noise = cbx.noise.comp_noise(dt = conf.dt)

#%% Define the CBO algorithm
dyn = CBO(x, f, noise, f_dim='2D', 
          alpha = conf.alpha, dt = conf.dt, 
          sigma = conf.sigma, lamda = conf.lamda,
          batch_size=conf.batch_size,
          check_list=conf.check_list)
scheduler = cbx.scheduler.scheduler(dyn, [multiply(name='alpha', factor=1.01, maximum=1e3),
                                          multiply(name='sigma', factor=1.005, maximum=10.)])
#%% Run the CBO algorithm
t = 0
it = 0
while not dyn.terminate():
    dyn.step()
    scheduler.update()
    
    if it%10 == 0:
        print(dyn.f_min)
        print('Alpha: ' + str(dyn.alpha))
        print('Sigma: ' + str(dyn.sigma))
        
    it+=1