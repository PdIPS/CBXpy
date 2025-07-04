import numpy as np
from typing import Callable, List

def apply_resampling_default(dyn, idx, sigma_indep=0.1, var_name='x'):
    z = dyn.sampler(size=(len(idx), dyn.N, *dyn.d))
    getattr(dyn, var_name)[idx, ...] += sigma_indep * np.sqrt(dyn.dt) * z

class resampling:
    """
    Resamplings from a list of callables

    Parameters
    ----------
    resamplings: list
        The list of resamplings to apply. Each entry should be a callable that accepts exactly one argument (the dynamic object) and returns a one-dimensional
        numpy array of indices.
        
    apply: Callable
        - ``dyn``: The dynmaic which the resampling is applied to.
        - ``idx``: List of indices that are resampled.
        
        The function that should be performed on a given dynamic for selected indices. This function has to have the signature apply(dyn,idx).
    
    """
    def __init__(self,
        resamplings: List[Callable], 
        apply:Callable = None, 
        sigma_indep:float = 0.1,
        var_name = 'x'
    ):
        self.resamplings = resamplings
        self.num_resampling = None
        self.apply = apply if apply is not None else apply_resampling_default
        self.sigma_indep = sigma_indep
        self.var_name = var_name

    def __call__(self, dyn):
        """
        Applies the resamplings to a given dynamic
            
        Parameters
        ----------
        dyn
            The dynamic object to apply resamplings to
        
        Returns
        -------
        None
        """
        self.check_num_resamplings(dyn.M)
        idx = np.unique(np.concatenate([r(dyn) for r in self.resamplings]))
        if len(idx)>0:
            self.apply(dyn, idx, var_name = self.var_name, sigma_indep = self.sigma_indep)
            self.num_resampling[idx] += 1
            if dyn.verbosity > 0:
                print('Resampled in runs ' + str(idx))
    
    def check_num_resamplings(self, M):
        if self.num_resampling is None:
            self.num_resampling = np.zeros(shape=(M))

class ensemble_update_resampling:
    """
    Resampling based on ensemble update difference
    
    Parameters
    ----------
    
    update_thresh: float
        The threshold for ensemble update difference. When the update difference is less than this threshold, the ensemble is resampled.

    Returns
    -------

    The indices of the runs to resample as a numpy array.
    """
    def __init__(self, update_thresh:float):
        self.update_thresh = update_thresh
        
    def __call__(self, dyn):
        return np.where(dyn.update_diff < self.update_thresh)[0]


class consensus_stagnation:
    def __init__(self, patience=5, update_thresh=1e-4):
        self.patience = patience
        self.update_thresh = update_thresh
        self.consensus_updates = []
        self.it = 0

        
    def __call__(self, dyn):
        return np.where(self.check_consensus_update(dyn)  < self.update_thresh)[0]
    
    def check_consensus_update(self, dyn):
        self.it += 1
        wt = float('inf') * np.ones(dyn.M,)
        wt[dyn.active_runs_idx] = 0
        
        if hasattr(self, 'consensus_old'):
            diffs = np.zeros(dyn.M,)
            diffs[dyn.active_runs_idx] = dyn.to_numpy(
                dyn.norm(dyn.consensus - self.consensus_old[dyn.active_runs_idx, ...], 
                         axis=tuple(-i for i in range(1, dyn.x.ndim-1))
                        )
            )[:, 0]
            
            self.consensus_updates.append(diffs)
            self.consensus_updates = self.consensus_updates[-self.patience:]
            wt += np.array(self.consensus_updates).max(axis=0)
        else:
            wt += float('inf') 

        self.set_consensus_old(dyn)        
        return wt

    def set_consensus_old(self, dyn):
        self.consensus_old = 0 * dyn.copy(dyn.x[:, 0:1, ...])
        self.consensus_old[dyn.active_runs_idx, ...] = dyn.copy(dyn.consensus)
    
class loss_update_resampling:
    """
    Resampling based on loss update difference

    Parameters
    ----------
    M: int
        The number of runs in the dynamic object the resampling is applied to.

    wait_thresh: int
        The number of iterations to wait before resampling. The default is 5. If the best loss is not updated after the specified number of 
        iterations, the ensemble is resampled.

    Returns
    -------

    The indices of the runs to resample as a numpy array.
    """

    def __init__(self, wait_thresh:int = 5):
        self.wait_thresh = wait_thresh
        self.initalized = False
    
    def __call__(self, dyn):
        self.check_energy_wait(dyn.M)
        self.wait += 1
        u_idx = self.best_energy > dyn.best_energy
        self.wait[u_idx] = 0
        self.best_energy[u_idx] = dyn.best_energy[u_idx]
        idx = np.where(self.wait >= self.wait_thresh)[0]
        self.wait = np.mod(self.wait, self.wait_thresh)
        return idx
    
    def check_energy_wait(self, M):
        if not self.initalized:
            self.best_energy = float('inf') * np.ones((M,))
            self.wait = np.zeros((M,), dtype=int)
        self.initalized = True
        
