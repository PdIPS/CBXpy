import numpy as np
import cbx

#%% define the objective function
def f(x):
    return np.linalg.norm(x)**2

#%%
opt = cbx.solver(f, method='cbo',
                 T = 100.,
                 max_eval = 3e5,
                 N = 20,
                 d=200, M=5, verbosity=3,
                 num_runs = 3,
                 energy_tol = 1e-8,
                 sigma=8.,
                 alpha=10.,
                 r=1.001,
                 dt=0.01,
                 batch_size=50,
                 #correction = 'heavi_side_reg',
                 parallel = False)

#%%
x = opt.solve()
