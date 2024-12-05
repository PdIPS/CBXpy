import cbx
from cbx.dynamics import CBO
import numpy as np
import cProfile


f = cbx.objectives.Quadratic()
x = np.random.uniform(-3,3, (10,100,200))

dyn = CBO(f, x=x, max_it=300, noise='anisotropic', verbosity=0, f_dim='3D')
cProfile.run('dyn.optimize()')

