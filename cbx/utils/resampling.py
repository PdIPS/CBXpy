import numpy as np

def apply_resamplings(dyn, resamplings: list):
    """
    Apply resamplings to a given dynamic

    Parameters
    ----------
    dyn
        The dynamic object to apply resamplings to

    resamplings
        The list of resamplings to apply. Each entry should be a callable that accepts exactly one argument (the dynamic object) and returns a one-dimensional
        numpy array of indices.

    Returns
    -------
    The indices of the runs to resample as a numpy array
    """

    return np.unique(np.concatenate([r(dyn) for r in resamplings]))
    

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

    def __init__(self, M:int, wait_thresh:int = 5):
        self.M = M
        self.best_energy = float('inf') * np.ones((self.M,))
        self.wait = np.zeros((self.M,), dtype=int)
        self.wait_thresh = wait_thresh
    
    def __call__(self,dyn):
        self.wait += 1
        self.wait[self.best_energy > dyn.best_energy] = 0
        self.best_energy = dyn.best_energy
        idx = np.where(self.wait >= self.wait_thresh)[0]
        self.wait = np.mod(self.wait, self.wait_thresh)
        return idx
