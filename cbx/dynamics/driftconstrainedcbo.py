from cbx.dynamics import CBO
import numpy as np
from cbx.constraints import MultiConstraint, get_constraints
#%%
def solve_system(A, x):
    return np.linalg.solve(A, x[..., None])[..., 0]

#%%
def dc_inner_step(self,):
    self.compute_consensus()
    self.drift = self.x - self.consensus
    noise = self.sigma * self.noise()
    const_drift = (self.dt/ self.eps) * self.G.grad_squared_sum(self.x)
    scaled_drift = self.lamda * self.dt * self.drift
    
    x_tilde = scaled_drift + const_drift + noise
    # A = np.eye(self.d[0]) + (self.dt/ self.eps) * self.G.hessian_squared_sum(self.x)
    # self.x -= solve_system(A, x_tilde)
    self.x -= self.G.solve_Id_hessian_squared_sum(
        self.x, x_tilde, factor=(self.dt/ self.eps)
    )


#%%
class DriftConstrainedCBO(CBO):
    '''
    Implements the algorithm in [1]
    
    
    [1] Carrillo, José A., et al. 
    "An interacting particle consensus method for constrained global 
    optimization." arXiv preprint arXiv:2405.00891 (2024).
    '''
    
    
    def __init__(
            self, f, 
            constraints = None, eps=0.01,  
            eps_indep=0.001, sigma_indep=0., 
            **kwargs
        ):
        super().__init__(f, **kwargs)
        self.G = MultiConstraint(get_constraints(constraints))
        self.eps = eps
        self.eps_indep = eps_indep
        self.sigma_indep = sigma_indep
        
        
    def inner_step(self,):
        self.indep_noise_step()
        dc_inner_step(self)
        
    def indep_noise_step(self,):
        if self.sigma_indep > 0 and (self.consensus is not None):
            while True:
                cxd = (np.linalg.norm(self.x - self.consensus, axis=-1)**2).mean(axis=-1)/self.d[0]
                idx = np.where(cxd < self.eps_indep)
                if len(idx[0]) == 0:
                    break
                z = np.random.normal(0,1, size=((len(idx[0]),) + self.x.shape[1:]))
                self.x[idx[0], ...] += (
                    self.sigma_indep * 
                    self.dt**0.5 * 
                    z
                )