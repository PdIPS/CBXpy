from cbx.dynamics import CBO
from cbx.dynamics.pdyn import post_process_default
from cbx.constraints import get_constraints, MultiConstraint
from cbx.regularizers import regularize_objective

class RegCombinationCBO(CBO):
    '''
    Implements the algorithm in [1]
    
    
    [1] Carrillo, José A., et al. 
    "Carrillo, J. A., Totzeck, C., & Vaes, U. (2023). 
    Consensus-based optimization and ensemble kalman inversion for 
    global optimization problems with constraints" 
    arXiv preprint https://arxiv.org/abs/2111.02970 (2024).
    '''

    def __init__(self, f, constraints = None, eps=0.01, nu = 1, **kwargs):
        super().__init__(f, **kwargs)
        self.G = MultiConstraint(get_constraints(constraints))
        self.eps = eps
        self.f = regularize_objective(self.f, self.G.squared, lamda=1/nu)
    
    def inner_step(self, ):
        self.compute_consensus()
        self.drift = self.x - self.consensus
        noise = self.sigma * self.noise()

        #  update particle positions
        
        x_tilde = self.x - self.lamda * self.dt * self.drift + noise
        if self.eps < 1e15:
            self.x = self.G.solve_Id_plus_call_grad(
                self.x,
                x_tilde,
                factor = self.dt / self.eps
            )
            #self.x = solve_system(A, x_tilde)
        else:
            self.x = x_tilde

    def set_post_process(self, post_process):
        self.post_process = (
            post_process if post_process is not None else post_process_default(copy_nan_to_num=True)
        )