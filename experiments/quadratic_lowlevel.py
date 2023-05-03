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
        'sigma': 5.0,
        'lamda': 1.0,
        'd': 20,
        'T': 100,
        'N': 50,})

#%% Define the objective function
f = Quadratic()

#%% Define the initial positions of the particles
x = cbx.utils.init_particles(d = conf.d, N = conf.N, x_min=-3., x_max = 3.)

#%% Define the noise function
noise = cbx.noise.comp_noise(dt = conf.dt)

#%% Define the CBO algorithm
dyn = CBO(x, f, noise, batch_eval=False, alpha = conf.alpha, dt = conf.dt, sigma = conf.sigma, lamda = conf.lamda)
scheduler = cbx.scheduler.exponential(dyn, r=1.1)
#%% Run the CBO algorithm
t = 0
it = 0
while t < conf.T:
    dyn.step(t)
    scheduler.update()
    t += conf.dt
    
    if it%10 == 0:
        print(dyn.f_min)
        
    it+=1

    