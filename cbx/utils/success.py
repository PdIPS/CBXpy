import numpy as np

def dist_to_min_success(x, x_true, tol=0.25, p=float('inf')):
    norm_diff = np.linalg.norm((x_true[None,...] - x.squeeze()).reshape(x.shape[0], -1), 
                               axis=-1, ord = p)
    idx = np.where(norm_diff < tol)[0]
    
    return {'num': len(idx), 
            'rate': len(idx)/x.shape[0], 
            'normdiff':norm_diff,
            'idx':idx}

class dist_to_min:
    def __init__(self, x_true, tol=0.25, p=float('inf')):
        self.x_true = x_true
        self.tol = tol
        self.p = p
        
    def __call__(self, dyn):
        return dist_to_min_success(dyn.best_particle, self.x_true, 
                                   tol=self.tol, p=self.p)

def value_success(x, f, thresh=0.1):
    vals = f(x)
    idx = np.where(vals < thresh)[0]
    
    return {'num': len(idx), 'rate': len(idx)/x.shape[0], 'idx':idx}

class value_thresh:
    def __init__(self, thresh=0.1):
        self.thresh = thresh
        
    def __call__(self, dyn):
        return value_success(dyn.best_particle[:,None,...], dyn.f, thresh=self.thresh)

class evaluation:
    def __init__(self, criteria=None, verbosity = 1):
        self.criteria = criteria
        self.verbosity = 1
        
    def __call__(self, dyn):
        idx = np.arange(dyn.M)
        for crit in self.criteria:
            result = crit(dyn)
            idx = np.intersect1d(result['idx'], idx)
            
        res = {'num': len(idx), 'rate': len(idx)/dyn.M, 'idx':idx}
        self.print_result(res)
        return res
    
    def print_result(self, res):
        if self.verbosity < 1:
            return
        print('------------------------------')
        print('Results of success evaluation:')
        print('Success Rate: ' + str(res['rate']))
        print('Succesful runs: ' + str(res['num']))
        print('Succesful idx: ' + str(res['idx']))
            