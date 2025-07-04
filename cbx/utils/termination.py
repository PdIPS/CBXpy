import numpy as np

#%%
term_dict = {}
#%%
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
    
term_dict.update(
    dict.fromkeys(['energy-tol', 'energy tol', 'energy_tol'], 
                  energy_tol_term
   )
)
    
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
    
term_dict.update(
    dict.fromkeys(['diff-tol', 'diff tol', 'diff_tol'], 
                  diff_tol_term
    )
)
    
class max_eval_term:
    """
    Check if the number of function evaluations is greater than or equal to the maximum number of evaluations.

    Returns:
        bool: True if the number of function evaluations is greater than or equal to the maximum number of evaluations, False otherwise.
    """
    def __init__(self, max_eval=1000):
        self.max_eval = max_eval
    
    def __call__(self, dyn):
        estimated_eval = dyn.batch_size
        return (dyn.num_f_eval + estimated_eval) > self.max_eval
    
term_dict.update(
    dict.fromkeys(['max-eval', 'max eval', 'max_eval'], 
                  max_eval_term
    )
)
    
class max_it_term:
    """
    Checks if the current value of `dyn.it` is greater than or equal to the value of `dyn.max_it`.

    Returns:
        bool: True if `dyn.it` is greater than or equal to `dyn.max_it`, False otherwise.
    """
    def __init__(self, max_it=1000):
        self.max_it = max_it
    
    def __call__(self, dyn):
        if self.max_it is None:
            return np.zeros((dyn.M), dtype=bool)
        else:
            return (dyn.it >= self.max_it) * np.ones((dyn.M), dtype=bool)
        
term_dict.update(
    dict.fromkeys(['max-it', 'max it', 'max_it'], 
                  max_it_term
    )
)

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
    
term_dict.update(
    dict.fromkeys(['max-time', 'max time', 'max_time'], 
                  max_time_term
    )
)

#%%    
class energy_stagnation_term:
    """
    Checks if the loss was moving during the last iterations.
    """
    def __init__(self, patience=20, std_thresh=1e-9):
        self.patience = patience
        self.losses = None
        self.std_thresh = std_thresh
        
    def __call__(self, dyn):
        if self.losses is None: 
            self.losses = np.random.uniform(0., 1., size=(self.patience, dyn.M))
        if dyn.consensus is None: 
            return np.zeros((dyn.M), dtype=bool)
        # eval loss
        E = dyn.f(dyn.consensus[dyn.active_runs_idx, ...])
        dyn.num_f_eval[dyn.active_runs_idx] += 1
        # update losses
        self.losses[dyn.it%self.patience, dyn.active_runs_idx] = E
        return np.std(self.losses, axis=0) < self.std_thresh
    
term_dict.update(
    dict.fromkeys(['energy-stagnation', 'energy stagnation', 'energy_stagnation'], 
                  energy_stagnation_term
    )
)
#%%
def select_term(term):
    if isinstance(term, str):
        return term_dict[term]()
    elif hasattr(term, 'keys'):
        if 'name' in term.keys():
            return term_dict[term['name']](
                **{k:v for k,v in term.items() if k not in ['name']}
                )
        else:
            raise ValueError('The given term dict: ' + str(term) + '\n ' +
                             'does not have the necessary key ' + 
                             '"name"')
    else:
        return term