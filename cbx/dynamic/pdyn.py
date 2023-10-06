import warnings
#%%
from ..utils.particle_init import init_particles
from ..utils.scheduler import scheduler
from ..utils.numpy_torch_comp import copy_particles
from ..utils.objective_handling import _promote_objective, batched_objective

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
    f : objective
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
            max_it: Union[int, None] = 1000,
            dt: float = 0.1, alpha: float = 1.0, sigma: float =1.0,
            lamda: float = 1.0,
            correction: str = 'no_correction', 
            correction_eps: float = 1e-3,
            array_mode: str = 'numpy',
            check_f_dims: bool = True,
            track_list: list = None,
            resampling: bool = False,
            update_thresh: float = 0.1,
            verbosity: int = 1) -> None:
        
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
        if not isinstance(f, batched_objective):
            if f_dim != '3D' and array_mode == 'pytorch':
                raise RuntimeError('Pytorch array_mode only supported for 3D objective functions.')
            self.f = _promote_objective(f, f_dim)
        else:
            self.f = f
        
        self.num_f_eval = 0 * np.ones((self.M,)) # number of function evaluations  
        if check_f_dims: # check if f returns correct shape
            x = np.random.uniform(-1,1,(self.M, self.N, self.d))
            if self.f(x).shape != (self.M,self.N):
                raise ValueError("The given objective function does not return the correct shape!")
            self.num_f_eval = N * np.ones((self.M,)) # number of function evaluations
        self.f_min = float('inf') * np.ones((self.M,)) # minimum function value
        
        self.energy = float('inf') * np.ones((self.M, self.N)) # energy of the particles
        self.best_energy = float('inf') * np.ones((self.M,))
        self.best_particle = np.zeros((self.M,self.d))
        self.update_diff = float('inf')


        self.set_noise(noise)


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
        
        self.consensus = None # mean of the particles
        
        self.set_correction(correction)

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
            
        self.track_list = track_list if not track_list is None else ['update_norm', 'energy']
        self.init_history()

                
        self.resampling = resampling
        self.update_thresh = update_thresh
        self.verbosity = verbosity 

    def step(self,) -> None:
        self.post_step()

    def optimize(self,
                 print_int:int = 100,
                 sched = None):
        
        if self.verbosity > 0:
            print('.'*20)
            print('Starting Optimization with dynamic: ' + self.__class__.__name__)
            print('.'*20)

        if sched is None:
            sched = scheduler(self, [])

        self.history['x'] = [self.copy_particles(self.x)]
        self.history['update_norm'] = [np.linalg.norm(self.x, axis=(-2,-1))]

        while not self.terminate(verbosity=self.verbosity):
            self.step()
            sched.update()

            if (self.it % print_int == 0):
                if self.verbosity > 0:
                    print('Time: ' + "{:.3f}".format(self.t) + ', best energy: ' + str(self.f_min))
                    print('Number of function evaluations: ' + str(self.num_f_eval))

                if self.verbosity > 1:
                    print('Current alpha: ' + str(self.alpha))

        if self.verbosity > 0:
            print('-'*20)
            print('Finished solver.')
            print('Best energy: ' + str(self.f_min))
            print('-'*20)

        return self.best_particle()

    def copy_particles(self, x):
        return copy_particles(x, mode=self.array_mode)

    def set_batch_idx(self,):
        r"""Set batch indices

        This method sets the batch indices for the next iteration of the algorithm.
        """
            
        if self.indices.shape[1] < self.batch_size: # if indices are exhausted
            indices = np.repeat(np.arange(self.N)[None,:], self.M ,axis=0)
            if self.batched:
                pass
                #indices = self.batch_rng.permuted(indices, axis=1)
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
            self.update_diff = np.linalg.norm(self.x - self.x_old, axis=(-2,-1))/self.N
        
        self.set_best_cur_particle()
        self.update_best_particle()
        self.track()

        if self.resampling:
            self.resample()
            
        self.t += self.dt
        self.it+=1
        
        
    def init_history(self,):
        self.history = {}
        for key in self.track_list:
            if key == 'x':
                self.history[key] = []
            elif key == 'update_norm':
                self.history[key] = np.zeros((self.M, self.max_it))
            elif key == 'energy':
                self.history[key] = np.zeros((self.M, self.max_it))
            else:
                raise ValueError('Unknown key ' + str(key) + ' for tracked values.')
        
    def track(self,):
        if 'x' in self.track_list:
            self.history['x'].append(self.copy_particles(self.x))
        if 'update_norm' in self.track_list:
            self.history['update_norm'][:, self.it] = self.update_diff
        if 'energy' in self.track_list:
            self.history['energy'][:, self.it] = self.best_cur_energy
        
    
    
    def resample(self,) -> None:
        idx = np.where(self.update_diff < self.update_thresh)[0]
        if len(idx)>0:
            z = np.random.normal(0, 1., size=(len(idx), self.N, self.d))
            self.x[idx, ...] += self.sigma * np.sqrt(self.dt) * z
            if self.verbosity > 0:
                    print('Resampled in runs ' + str(idx))

    def set_best_cur_particle(self,):
        self.f_min = self.energy.min(axis=-1)
        self.f_min_idx = self.energy.argmin(axis=-1)
        self.best_cur_particle = self.x[np.arange(self.M), self.f_min_idx, :]
        self.best_cur_energy = self.energy[np.arange(self.M), self.f_min_idx]
    
    def update_best_particle(self,):
        idx = np.where(self.best_energy > self.best_cur_energy)[0]
        if len(idx) > 0:
            self.best_energy[idx] = self.best_cur_energy[idx]
            self.best_particle[idx, :] = self.best_cur_particle[idx, :]


    def set_correction(self, correction):
        if correction == 'no_correction':
            self.correction = self.no_correction
        elif correction == 'heavi_side':
            self.correction = self.heavi_side
        elif correction == 'heavi_side_reg':
            self.correction = self.heavi_side_reg
        else:
            self.correction = correction
    
    def no_correction(self,):
        return np.ones(self.x.shape)

    def heavi_side_correction(self,):
        z = self.energy - self.f(self.consensus)
        self.num_f_eval += self.consensus.shape[0] # update number of function evaluations

        return np.where(z > 0, 1,0)[...,None]

    def heavi_side_reg_correction(self,):
        z = self.energy - self.f(self.consensus)
        self.num_f_eval += self.consensus.shape[0] # update number of function evaluations

        return 0.5 + 0.5 * np.tanh(z/self.correction_eps)
    
    def set_noise(self, noise):
        # set noise model
        if noise == 'isotropic' or noise is None:
            self.noise = self.isotropic_noise
        elif noise == 'anisotropic':
            self.noise = self.anisotropic_noise
        elif noise == 'sampling':
            self.noise = self.covariance_noise
            warnings.warn('Currently not bug-free!', stacklevel=2)
        else:
            warnings.warn('Custom noise specified. This is not the recommended\
                          for choosing a custom noise model.', stacklevel=2)
            self.noise = noise

    def anisotropic_noise(self,):
        """
        """
        z = np.random.normal(0, 1, size=self.drift.shape) * self.drift
        return np.sqrt(self.dt) * z
        
    def isotropic_noise(self,):
        r"""Model for normal distributed noise

        This class implements a normal noise model with zero mean and covariance matrix :math:`\dt I_d`
        where :math:`\dt` is a parameter of the class. Given the vector :math:`x_i - \mathsf{m}(x_i)`,
        the noise vector is computed as

        .. math::

            n_i = \dt \sqrt{\|x_i - \mathsf{m}(x_i)\|_2}\ \mathcal{N}(0,1)


        Parameters
        ----------
        dt : float, optional
            The parameter :math:`\dt` of the noise model. The default is 0.1.
        
        Examples
        --------
        >>> import numpy as np
        >>> from polarcbo.noise import normal_noise
        >>> drift = np.array([[2,3], [4,5], [1,4.]])
        >>> noise = normal_noise(dt=0.1)
        >>> noise(drift)
        array([[-2.4309445 ,  1.34997294],
               [-1.08502177,  0.24030935],
               [ 0.1794014 , -1.09228077]])


        """

        z = np.sqrt(self.dt) * np.random.normal(0, 1, size=self.drift.shape)
        return z * np.linalg.norm(self.drift, axis=-1,keepdims=True)
        
    def covariance_noise(self,):
        self.update_covariance()
        z = np.random.normal(0, 1, size=self.x.shape) # num, d
        noise = np.zeros(self.x.shape)
        
        # the following needs to be optimized
        for j in range(self.x.shape[0]):
            noise[j,:] = self.C_sqrt[j,::]@(z[j,:])
        return (np.sqrt(1/self.lamda * (1 - self.alpha**2))) * noise
    
    def update_covariance(self,):
        pass
    