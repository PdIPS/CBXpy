import numpy as np

#%%
def solve_system(A, x):
    return np.linalg.solve(A, x[..., None])[..., 0]
#%%
class MultiConstraint:
    def __init__(self, constraints):
        self.constraints = [] if constraints is None else constraints
        
    def __call__(self, x):
        out = 0 
        for c in self.constraints:
            out += c(x)
        return out
    
    def squared(self, x):
        out = 0
        for c in self.constraints:
            out += c.squared(x)
        return out
        
    def grad_squared_sum(self, x):
        out = np.zeros_like(x)
        for con in self.constraints:
            out += con.grad_squared(x)
        return out
            
    def hessian_squared_sum(self, x):
        out = np.zeros(x.shape + (x.shape[-1],))
        for con in self.constraints:
            out += con.hessian_squared(x)
        return out
    
    def hessian_sum(self, x):
        out = np.zeros(x.shape + (x.shape[-1],))
        for con in self.constraints:
            out += con.hessian(x)
        return out
    
    def call_times_hessian_sum(self, x):
        out = 0 
        for c in self.constraints:
            out += c(x)[..., None, None] * c.hessian(x)
        return out
    
    def solve_Id_call_times_hessian(self, x, x_tilde, factor = 1.):
        out = 0
        if len(self.constraints) > 1:
            A = np.eye(x.shape[-1]) + factor * self.call_times_hessian_sum(x)
            return solve_system(A, x_tilde)
        elif len(self.constraints) == 1:
            return self.constraints[0].solve_Id_call_times_hessian(x, x_tilde, factor=factor)
        else:
            return 0
        return out
    
    def solve_Id_plus_call_grad(self, x, x_tilde, factor = 1.):
        if len(self.constraints) > 1:
            raise ValueError('This function is not supported for more than one ' +
                             'constraint')
        elif len(self.constraints) == 1:
            return self.constraints[0].solve_Id_plus_call_grad(x, x_tilde, factor=factor)
        else:
            return 0


    def solve_Id_hessian_squared_sum(self, x, x_tilde, factor=1.):
        if len(self.constraints) > 1:
            A = np.eye(x.shape[-1]) + factor * self.G.hessian_squared_sum(x)
            return solve_system(A, x_tilde)
        elif len(self.constraints) == 1:
            return self.constraints[0].solve_Id_hessian_squared(
                x,
                x_tilde,
                factor = factor
            )
        else:
            return 0
    
class Constraint:
    def squared(self, x):
        return self(x)**2
    
    def grad_squared(self, x):
        return 2 * self(x)[..., None] * self.grad(x)
    
    def hessian_squared(self, x):
        grad = self.grad(x)
        outer = np.einsum('...i,...j->...ij', grad, grad)
        return 2 * (outer + self(x)[..., None, None] * self.hessian(x))
    
    def call_times_hessian(self, x):
        return self(x)[..., None, None] * self.hessian(x)
    
    def solve_Id_call_times_hessian(self, x, x_tilde, factor=1.):
        A = (
            np.eye(x.shape[-1]) + 
            factor * self.call_times_hessian(x)
        )
        return solve_system(A, x_tilde)
    
    def solve_Id_hessian_squared(self, x, x_tilde, factor=1.):
        A = np.eye(x.shape[-1]) + factor * self.hessian_squared(x)
        return solve_system(A, x_tilde)
    
class NoConstraint(Constraint):
    def __init__(self,):
        super().__init__()
        
    def __call__(self, x):
        return np.zeros(x.shape[:-1])
    
    def grad(self, x):
        return np.zeros_like(x)
    
    def hessian(self, x):
        return np.zeros(x.shape + (x.shape[-1],))
    
    
class quadricConstraint(Constraint):
    def __init__(self, A = None, b = None, c = 0):
        self.A = A
        self.b = b
        self.c = c
        
    def __call__(self, x):
         return (
             (x * (x@self.A)).sum(axis=-1) + 
             (x * self.b).sum(axis=-1) + 
             self.c
        )
        
    def grad(self, x):
        return 2 * x@self.A + self.b
    
    def hessian(self, x):
        return 2 * self.A
    
    def solve_Id_plus_call_grad(self, x, x_tilde, factor=1.):
        M = (np.eye(self.A.shape[0])[None, None, ...] + 
             4 * factor * self(x)[..., None, None] * self.A)
        return np.linalg.solve(M, (x_tilde - 2 * factor * self(x)[..., None] * self.b)[..., None])[..., 0]
    
class sphereConstraint(Constraint):
    def __init__(self, r=1.):
        super().__init__()
        self.r = r
        
    def __call__(self, x):
        return np.linalg.norm(x, axis=-1)**2 - self.r
    
    def grad(self, x):
        return 2 * x
    
    def hessian(self, x):
        return 2 * np.tile(np.eye(x.shape[-1]), x.shape[:-1] + (1,1))
    
    def solve_Id_plus_call_grad(self, x, x_tilde, factor=1.):
        return (1/(1 + 4 * factor * (np.linalg.norm(x, axis=-1, keepdims=True)**2 - self.r))) * x_tilde
    
    

class planeConstraint(Constraint):
    def __init__(self, a=0, b=1.):
        super().__init__()
        self.a = a
        self.norm_a = np.linalg.norm(a, axis=-1)
        self.b = b
        
    def __call__(self, x):
        return ((self.a * x).sum(axis=-1) - self.b)/self.norm_a
    
    def grad(self, x):
        return (self.a/self.norm_a) * np.ones_like(x)
    
    def hessian(self, x):
        return np.zeros(x.shape + (x.shape[-1],))
    
    def solve_Id_call_times_hessian(self, x, x_tilde, factor=1.):
        return x_tilde

    def solve_Id_plus_call_grad(self, x, x_tilde, factor=1.):
        return x_tilde - 2 * factor * self(x)[...,None] * self.a/self.norm_a
    
    
const_dict = {'plane': planeConstraint, 
              'sphere': sphereConstraint,
              'quadric': quadricConstraint}    

def get_constraints(const):
    CS = []
    const = [] if const is None else const
    for c in const:
        if c is None:
            pass #return NoConstraint()
        else:
            CS.append(const_dict[c['name']](**{k:v for k,v in c.items() if k not in ['name']}))
    return CS