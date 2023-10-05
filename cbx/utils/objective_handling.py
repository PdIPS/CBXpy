import numpy as np

def _promote_objective(f, f_dim):
    if not callable(f):
        raise TypeError("Objective function must be callable.")
    if f_dim == '3D':
        return f
    elif f_dim == '2D':
        return batched_objective_from_2D(f)
    elif f_dim == '1D':
        return batched_objective_from_1D(f)
    else:
        raise ValueError("f_dim must be '1D', '2D' or '3D'.")

class batched_objective:
    def __init__(self, f):
        self.f = f
        self.eval_count = 0
    def __call__(self, x):
        self.eval_count += np.prod(x.shape[:-1])
        return self.f(x)

class batched_objective_from_1D(batched_objective):
    def __init__(self, f):
        super().__init__(f)
    
    def __call__(self, x):
        self.eval_count += np.prod(x.shape[:-1])
        return np.apply_along_axis(self.f, 1, np.atleast_2d(x.reshape(-1, x.shape[-1]))).reshape(-1,x.shape[-2])
    
class batched_objective_from_2D(batched_objective):
    def __init__(self, f):
        super().__init__(f)
    
    def __call__(self, x):
        self.eval_count += np.prod(x.shape[:-1])
        return self.f(np.atleast_2d(x.reshape(-1, x.shape[-1]))).reshape(-1,x.shape[-2])