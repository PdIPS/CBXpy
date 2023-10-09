import numpy as np
import cbx
import matplotlib.pyplot as plt

#%% define the objective function and solve
def f(x):
    return np.sin(x) * np.exp(-x**2)
dyn = cbx.dynamic.CBO(f, d=1)
x = dyn.optimize()

#%% visualize
plt.close('all')
s = np.linspace(-4,4,100)
plt.plot(s, f(s), linewidth=3, color='xkcd:sky')
plt.scatter(x, f(x))


