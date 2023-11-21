import numpy as np
from typing import Callable, Any

def _promote_objective(f, f_dim):
    if not callable(f):
        raise TypeError("Objective function must be callable.")
    if f_dim == '3D':
        return f
    elif f_dim == '2D':
        return cbx_objective_f2D(f)
    elif f_dim == '1D':
        return cbx_objective_f1D(f)
    else:
        raise ValueError("f_dim must be '1D', '2D' or '3D'.")


class cbx_objective:
    def __init__(self, f_extra=None):
        super().__init__()
        self.num_eval = 0
        
    def __call__(self, x):
        """
        Applies the objective function to the input x and counts th number of evaluations.

        Parameters
        ----------
        x
            The input to the objective function.
        
        Returns
        -------
        The output of the objective function.
        """

        self.num_eval += np.prod(np.atleast_2d(x).shape[:-1], dtype = int)
        return self.apply(x)

    def apply(self, x): 
        NotImplementedError(f"Objective [{type(self).__name__}] is missing the required \"apply\" function")
        
    def reset(self,):
        self.num_eval = 0
        
        
class cbx_objective_fh(cbx_objective):
    """
    Creates a cbx_objective from a function handle.
    """

    def __init__(self, f):
        super().__init__()
        self.f = f
        
    def apply(self, x):
        """
        Applies the function f to the input x.

        Parameters
        ----------
        x
            The input to the function.
        
        Returns
        -------
        The output of the function.
        """

        return self.f(x)
    
class cbx_objective_f1D(cbx_objective_fh):
    def __init__(self, f):
        super().__init__(f)
    
    def apply(self, x):
        x = np.atleast_2d(x)
        return np.apply_along_axis(self.f, 1, x.reshape(-1, x.shape[-1])).reshape(-1,x.shape[-2])
    
    
class cbx_objective_f2D(cbx_objective_fh):
    """
    A class for handling 2D objective functions.
    """
    def __init__(self, f):
        super().__init__(f)
    
    def apply(self, x):
        """
        Applies the function f to the input x.
        
        Parameters
        ----------
        x
            The input to the function.
        
        Returns
        -------
        The output of the function.
        """
        x = np.atleast_2d(x)
        return self.f(np.atleast_2d(x.reshape(-1, x.shape[-1]))).reshape(-1,x.shape[-2])