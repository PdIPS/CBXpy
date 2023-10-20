import warnings
#%%
import numpy as np
from .pdyn import CBXDynamic

#%%
class CBS(CBXDynamic):
    def __init__(self, f, mode='sampling', noise='covariance', **kwargs):
        super().__init__(f, **kwargs)
        
        if self.batched:
            raise NotImplementedError('Batched mode not implemented for CBS!')
            
        self.exp_dt = np.exp(self.dt)
        
        if noise not in ['covariance', 'sampling']:
            raise warnings.warn('For CBS usually covariance or sampling noise is used!', stacklevel=2)
        self.noise = self.covariance_noise
        
        
    def inner_step(self,):
        self.consensus, energy = self.compute_consensus(self.x)
        self.energy = energy
        self.drift = self.x - self.consensus
        self.x = self.consensus + self.exp_dt * self.drift + self.covariance_noise()
        