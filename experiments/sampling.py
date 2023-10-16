from cbx.dynamics import CBS
from cbx.objectives import Quadratic

#%%
f = Quadratic()
dyn = CBS(f, d=5, M=3)

dyn.step()