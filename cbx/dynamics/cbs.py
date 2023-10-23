import warnings
#%%
import numpy as np
from .pdyn import CBXDynamic
from ..scheduler import scheduler
#%%
class CBS(CBXDynamic):
    def __init__(self, f, mode='sampling', noise='covariance',
                 M=1,
                 track_list=None, 
                 **kwargs):
        track_list = track_list if track_list is not None else []
        super().__init__(f, track_list=track_list, M=M, **kwargs)
        
        if self.batched:
            raise NotImplementedError('Batched mode not implemented for CBS!')
            
        self.exp_dt = np.exp(-self.dt)
        if mode == 'sampling':
            self.lamda = 1/(1 + self.alpha)
        elif mode == 'optimization':
            self.lamda = 1
        else:
            raise NotImplementedError("Invalid mode")
        
        if noise not in ['covariance', 'sampling']:
            raise warnings.warn('For CBS usually covariance or sampling noise is used!', stacklevel=2)
        self.noise = self.covariance_noise
        
        
    def inner_step(self,):
        self.consensus, energy = self.compute_consensus(self.x)
        self.energy = energy
        self.drift = self.x - self.consensus
        self.update_covariance()
        self.x = self.consensus + self.exp_dt * self.drift + self.noise()
        
    def run(self, sched = 'default'):
            if self.verbosity > 0:
                print('.'*20)
                print('Starting Run with dynamic: ' + self.__class__.__name__)
                print('.'*20)

            if sched is None:
                sched = scheduler(self, [])
            elif sched == 'default':
                sched = self.default_sched()
            else:
                if not isinstance(sched, scheduler):
                    raise RuntimeError('Unknonw scheduler specified!')

            while not self.terminate(verbosity=self.verbosity):
                self.step()
                sched.update()
        
    def post_step(self):
        self.track()
        self.process_particles()
            
        self.it+=1
        
    def default_sched(self,):
        return scheduler(self, [])
        
    def process_particles(self,):
        pass