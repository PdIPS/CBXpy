import numpy as np
import cbx

#%% define the objective function
def f(x):
    return np.linalg.norm(x)**2

d = 200

#%%
opt = cbx.solver(f, method='cbo',
                 T = 2.2,
                 d=d, M=3, verbosity=3,
                 energy_tol = 1e-8,
                 num_cores=4,
                 correction = 'heavi_side_reg',
                 parallel = True)

#%%
if __name__ == '__main__':
    x = opt.solve()
