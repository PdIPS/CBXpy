import numpy as np

def apply_resamplings(dyn, resamplings: list):
    return np.unique(np.concatenate([r(dyn) for r in resamplings]))
    

class ensemble_update_resampling:
    def __init__(self, update_thresh:float):
        self.update_thresh = update_thresh
        
    def __call__(self, dyn):
        return np.where(dyn.update_diff < self.update_thresh)[0]
    
class loss_update_resampling:
    def __init__(self, M:int, wait_thresh:int = 5):
        self.M = M
        self.best_energy = float('inf') * np.ones((self.M,))
        self.wait = np.zeros((self.M,), dtype=int)
        self.wait_thresh = wait_thresh
    
    def __call__(self,dyn):
        self.wait += 1
        self.wait[np.where(self.best_energy > dyn.best_energy)[0]] = 0
        self.best_energy = dyn.best_energy
        idx = np.where(self.wait >= self.wait_thresh)[0]
        self.wait = np.mod(self.wait, self.wait_thresh)
        return idx
