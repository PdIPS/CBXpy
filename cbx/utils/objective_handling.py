import numpy as np
from functools import wraps

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


def create_hook(func, hook):
    @wraps(func)
    def wrapper(*args, **kwargs):
        hook(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper


class cbx_objective:
    def __init__(self,):
        super().__init__()
        self.eval_count = 0
        
    def __new__(cls, *args, **kwargs):
        cls.__call__ = create_hook(cls.__call__, cls.add_eval_hook)
        return super().__new__(cls,)
    
    def add_eval_hook(self, x):
        self.eval_count += np.prod(x.shape[:-1], dtype = int)
        
class cbx_objective_fh(cbx_objective):
    def __init__(self, f):
        super().__init__()
        self.f = f
        
    def __call__(self, x):
        return self.f(x)
    


class batched_objective_from_1D(cbx_objective_fh):
    def __init__(self, f):
        super().__init__(f)
    
    def __call__(self, x):
        self.eval_count += np.prod(x.shape[:-1])
        return np.apply_along_axis(self.f, 1, np.atleast_2d(x.reshape(-1, x.shape[-1]))).reshape(-1,x.shape[-2])
    
class batched_objective_from_2D(cbx_objective_fh):
    def __init__(self, f):
        super().__init__(f)
    
    def __call__(self, x):
        self.eval_count += np.prod(x.shape[:-1])
        return self.f(np.atleast_2d(x.reshape(-1, x.shape[-1]))).reshape(-1,x.shape[-2])