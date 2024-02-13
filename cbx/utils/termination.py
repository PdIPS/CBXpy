import numpy as np

class energy_tol_term:
    """
    Check if the energy is below a certain tolerance.

    Returns:
        bool: True if the energy is below the tolerance, False otherwise.
    """
    def __init__(self, energy_tol=1e-7):
        self.energy_tol = energy_tol
    
    def __call__(self, dyn):
        return dyn.f_min < self.energy_tol
    
class diff_tol_term:
    """
    Checks if the update difference is less than the difference tolerance.

    Returns:
        bool: True if the update difference is less than the difference tolerance, False otherwise.
    """
    def __init__(self, diff_tol=1e-7):
        self.diff_tol = diff_tol
    def __call__(self, dyn):
        return dyn.update_diff < self.diff_tol
    
class max_eval_term:
    """
    Check if the number of function evaluations is greater than or equal to the maximum number of evaluations.

    Returns:
        bool: True if the number of function evaluations is greater than or equal to the maximum number of evaluations, False otherwise.
    """
    def __init__(self, max_eval=1000):
        self.max_eval = max_eval
    
    def __call__(self, dyn):
        return dyn.num_f_eval >= self.max_eval
    
class max_it_term:
    """
    Checks if the current value of `dyn.it` is greater than or equal to the value of `dyn.max_it`.

    Returns:
        bool: True if `dyn.it` is greater than or equal to `dyn.max_it`, False otherwise.
    """
    def __init__(self, max_it=1000):
        self.max_it = max_it
    
    def __call__(self, dyn):
        return (dyn.it >= self.max_it) * np.ones((dyn.M), dtype=bool)

class max_time_term:
    """
    Checks if the current value of `dyn` is greater than or equal to the value of `dyn.max_time`.

    Returns:
        bool: True if `dyn.t` is greater than or equal to `dyn.max_time`, False otherwise.
    """
    def __init__(self, max_time=10.):
        self.max_time = max_time
    def __call__(self, dyn):
        return (dyn.t >= self.max_time) * np.ones((dyn.M), dtype=bool)