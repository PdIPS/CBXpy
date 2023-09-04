#%%
from ..noise import normal_noise
from ..utils.particle_init import init_particles
from ..utils.scheduler import scheduler
from ..utils.numpy_torch_comp import copy_particles
from ..utils.objective_handling import _promote_objective

#%%
from typing import Callable, Union
import numpy as np


class ParticleDynamic():
    r"""Particle dynamic class

    This class implements the base particle dynamic class. It is used to update the particles
    in the consensus schemes.   

    Parameters
    ----------
    x : array_like, shape (J, d)
        The initial positions of the particles. For a system of :math:`J` particles, the i-th row of this array ``x[i,:]``
        represents the position :math:`x_i` of the i-th particle.
    f : obejective
        The objective function :math:`f(x)` of the system.
    alpha : float, optional
        The heat parameter :math:`\alpha` of the system. The default is 1.0.
    
    """

    def __init__(
            self, 
            f: Callable, f_dim: str = '1D',
            x: Union[None, np.ndarray] = None,
            x_min: float = -1., x_max: float = 1.,
            M: int = 1, N: int = 20, d: int = None,
            noise: Union[None, Callable] = None,
            batch_size: Union[None, int] = None,
            batch_partial: bool = False,
            batch_seed: int = 42,
            energy_tol: Union[float, None] = None, 
            diff_tol: Union[float, None] = None,
            max_eval: Union[int, None] = None,
            max_time: Union[float, None] = None,
            max_it: Union[int, None] = None,
            dt: float = 0.1, alpha: float = 1.0, sigma: float =1.0,
            lamda: float = 1.0,
            correction: Union[None, str, Callable] = None, 
            correction_eps: float = 1e-3,
            array_mode: str = 'numpy',
            check_f_dims: bool = True) -> None:
        
        # init particles        
        if x is None:
            if d is None:
                raise RuntimeError('If the inital partical system is not given, the dimension d must be specified!')
            x = init_particles(
                    shape=(M, N, d), 
                    x_min = x_min, x_max = x_max
                )
        else: # if x is given correct shape
            if len(x.shape) == 1:
                x = x[None, None, :]
            elif len(x.shape) == 2:
                x = x[None, :]

        self.M = x.shape[-3]
        self.N = x.shape[-2]
        self.d = x.shape[-1]

        # torch compatibility for copying particles
        self.array_mode = array_mode
        self.x = self.copy_particles(x)
        
        # set and promote objective function
        if f_dim != '3D' and array_mode == 'pytorch':
            raise RuntimeError('Pytorch array_mode only supported for 3D objective functions.')
        self.f = _promote_objective(f, f_dim)
        self.num_f_eval = 0 * np.ones((self.M,)) # number of function evaluations  
        if check_f_dims: # check if f returns correct shape
            x = np.random.uniform(-1,1,(self.M, self.N, self.d))
            if self.f(x).shape != (self.M,self.N):
                raise ValueError("The given objective function does not return the correct shape!")
            self.num_f_eval = N * np.ones((self.M,)) # number of function evaluations
        self.f_min = float('inf') * np.ones((self.M,)) # minimum function value
        
        self.energy = None # energy of the particles
        self.update_diff = float('inf')

        # set noise model
        if noise is None:
            self.noise = normal_noise(dt = dt)
        else:
            self.noise = noise

        # termination parameters and checks
        self.energy_tol = energy_tol
        self.diff_tol = diff_tol
        self.max_eval = max_eval
        self.max_time = max_time
        self.max_it = max_it
    
        self.checks = []
        if energy_tol is not None:
            self.checks.append(self.check_energy)
        if diff_tol is not None:
            self.checks.append(self.check_update_diff)
        if max_eval is not None:
            self.checks.append(self.check_max_eval)
        if max_time is not None:
            self.checks.append(self.check_max_time)
        if max_it is not None:
            self.checks.append(self.check_max_it)
        
        self.all_check = np.zeros(self.M, dtype=bool)
        self.term_reason = {}
        self.t = 0.
        self.it = 0

        # additional parameters
        self.dt = dt
        self.alpha = alpha
        self.sigma = sigma
        self.lamda = lamda
        
        self.m_alpha = None # mean of the particles

        if correction is None:
            self.correction = no_correction()
        elif correction == 'heavi_side':
            self.correction = heavi_side()
        elif correction == 'heavi_side_reg':
            self.correction = heavi_side_reg(eps = correction_eps)
        else:
            self.correction = correction

        # batching
        self.batch_partial = batch_partial
        if batch_size is None:
            self.batch_size = self.N
        else:
            self.batch_size = min(batch_size, self.N)
        # set indices for batching
        self.indices = np.repeat(np.arange(self.N)[None,:], self.M ,axis=0)
        self.M_idx = np.repeat(np.arange(self.M)[:,None], self.batch_size, axis=1)
        if self.batch_size == self.N:
            self.batched = False
        else:
            self.batched = True
            self.batch_rng = np.random.default_rng(batch_seed)
            self.indices = self.batch_rng.permuted(self.indices, axis=1)

    def step(self,) -> None:
        self.post_step()

    def optimize(self, 
                 verbosity:int = 0, 
                 save_particles:bool = False, 
                 print_int:int = 100,
                 sched = None):
        
        if verbosity > 0:
            print('.'*20)
            print('Starting Optimization with dynamic: ' + self.__class__.__name__)
            print('.'*20)

        if sched is None:
            sched = scheduler(self, [])

        x_history = [self.copy_particles(self.x)]

        while not self.terminate(verbosity=verbosity):
            self.step()
            sched.update()

            if (self.it % print_int == 0):
                if verbosity > 0:
                    print('Time: ' + "{:.3f}".format(self.t) + ', best energy: ' + str(self.f_min))
                    print('Number of function evaluations: ' + str(self.num_f_eval))

                if verbosity > 1:
                    print('Current alpha: ' + str(self.alpha))
                    
                if save_particles:
                    x_history.append(self.copy_particles(self.x))

        if verbosity > 0:
            print('-'*20)
            print('Finished solver.')
            print('Best energy: ' + str(self.f_min))
            print('-'*20)

        return self.best_particle(), x_history

    def copy_particles(self, x):
        return copy_particles(x, mode=self.array_mode)

    def set_batch_idx(self,):
        r"""Set batch indices

        This method sets the batch indices for the next iteration of the algorithm.
        """
            
        if self.indices.shape[1] < self.batch_size: # if indices are exhausted
            indices = np.repeat(np.arange(self.N)[None,:], self.M ,axis=0)
            if self.batched:
                indices = self.batch_rng.permuted(indices, axis=1)
            self.indices = np.concatenate((self.indices, indices), axis=1)
            #self.batch_size

        self.batch_idx = self.indices[:,:self.batch_size] # get batch indices
        self.indices = self.indices[:, self.batch_size:] # remove batch indices from indices

    def get_mean_ind(self):
        return (self.M_idx, self.batch_idx, Ellipsis)
    
    def get_ind(self):
        if self.batch_partial:
            return self.get_mean_ind()
        else:
            return Ellipsis

    def check_max_time(self):
        return self.t >= self.max_time
            
    def check_energy(self):
        return self.f_min < self.energy_tol
    
    def check_update_diff(self):
        return self.update_diff < self.diff_tol
    
    def check_max_eval(self):
        return self.num_f_eval >= self.max_eval
    
    def check_max_it(self):
        return self.it >= self.max_it
    
    def terminate(self, verbosity = 0):
        loc_check = np.zeros((self.M,len(self.checks)), dtype=bool)
        for i,check in enumerate(self.checks):
            loc_check[:,i] = check()
            
        all_check = np.sum(loc_check, axis=1)
            
        for j in range(self.M):
            if all_check[j] and not self.all_check[j]:
                self.term_reason[j] = np.where(loc_check[j,:])[0]
        self.all_check = all_check

        if np.all(self.all_check):
            for j in range(self.M):
                if verbosity > 0:
                    print('Run ' + str(j) + ' returning on checks: ')
                    for k in self.term_reason[j]:
                        print(self.checks[k].__name__)
            return True
        else:
            return False
        
    def post_step(self):
        if hasattr(self, 'x_old'):
            self.update_diff = np.linalg.norm(self.x - self.x_old)

        if self.energy is not None:
            self.f_min = self.energy.min(axis=-1)
            self.f_min_idx = self.energy.argmin(axis=-1)
        else:
            self.f_min = float('inf') * np.ones((self.M,))
            self.f_min_idx = np.zeros((self.M,), dtype=int)
        self.t += self.dt
        self.it+=1

    def best_particle(self):
        return self.x[np.arange(self.M), self.f_min_idx, :]

class no_correction:
    def __call__(self, dyn):
        return np.ones(dyn.x.shape)

class heavi_side:
    def __call__(self, dyn):
        x = dyn.energy - dyn.f(dyn.m_alpha)
        dyn.num_f_eval += dyn.m_alpha.shape[0] # update number of function evaluations

        return np.where(x > 0, 1,0)[...,None]

class heavi_side_reg:
    def __init__(self, eps=1e-3):
        self.eps = eps
    
    def __call__(self, dyn):
        x = dyn.energy - dyn.f(dyn.m_alpha)
        dyn.num_f_eval += dyn.m_alpha.shape[0] # update number of function evaluations

        return 0.5 + 0.5 * np.tanh(x/dyn.eps)