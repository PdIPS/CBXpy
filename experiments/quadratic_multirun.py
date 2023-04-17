import numpy as np
import cbx

#%% define the objective function
f = lambda x: np.linalg.norm(x)**2
d = 20

opt = cbx.solver(f, method='cbo', d=d, verbosity=2)

if __name__ == "__main__":
    x = opt.multirun(M=10, parallel=True)