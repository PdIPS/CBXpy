from cbx.dynamic import CBO
import numpy as np

#%%
class custom_noise(CBO):
    def __init__(self, f, **kwargs):
        super().__init__(f, **kwargs)
        
    def noise(self,):
        return np.zeros(self.x.shape)
#%%
def f(x):
    return np.linalg.norm(x, axis=-1)

dyn = custom_noise(f, d=3)
dyn.optimize()
#%%
print(dyn.x)
