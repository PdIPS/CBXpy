import numpy as np

from sys import path
path.append('..')

import cbx as cbx
from cbx.dynamic.cbo import CBO
from cbx.objectives import Quadratic

#%%
conf = cbx.utils.config(**{
        'alpha': 10.0,
        'dt': 0.01,
        'sigma': 8.1,
        'lamda': 1.0,
        'batch_size':100,
        'd': 200,
        'T': 100,
        'N': 100,
        'M': 3})

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
          batch_size=conf.batch_size)
scheduler = cbx.scheduler.exponential(dyn, r=1.1)
#%% Run the CBO algorithm
t = 0
it = 0
while not dyn.terminate():
    dyn.step()
    scheduler.update()
    
    if it%10 == 0:
        print(dyn.f_min)
        
    it+=1