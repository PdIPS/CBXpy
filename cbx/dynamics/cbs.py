import warnings
from .cbo import CBO
from ..scheduler import scheduler
#%%
class CBS(CBO):
    def __init__(self, f, mode='sampling', noise='covariance',
                 M=1,
                 track_args=None, 
                 **kwargs):
        track_args = track_args if track_args is not None else{'names':[]}
        super().__init__(f, track_args=track_args, noise=noise, M=M, **kwargs)
        self.sigma = 1.
        
        if self.batched:
            raise NotImplementedError('Batched mode not implemented for CBS!')
        if self.x.ndim > 3:
            raise NotImplementedError('Multi dimensional domains not implemented for CBS! The particle should have the dimension M x N x d, where d is an integer!')
          
        if noise not in ['covariance', 'sampling']:
            raise warnings.warn('For CBS usually covariance or sampling noise is used!', stacklevel=2)
        
        self.noise_callable.mode = mode
        
    # def inner_step(self,):
    #     self.consensus, energy = self.compute_consensus()
    #     self.energy = energy
    #     self.drift = self.x - self.consensus
    #     self.x = self.consensus + self.exp_dt * self.drift + self.noise()
        
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

            while not self.terminate():
                self.step()
                sched.update(self)
                
    def optimize(self, sched = 'default'):
        self.run(sched=sched)
        
    def default_sched(self,):
        return scheduler([])
        
    def process_particles(self,):
        pass