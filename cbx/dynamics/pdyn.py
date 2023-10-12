import warnings
#%%
from ..scheduler import scheduler
from ..utils.particle_init import init_particles
from ..utils.numpy_torch_comp import copy_particles
from ..utils.objective_handling import _promote_objective, cbx_objective

#%%
from typing import Callable, Union
import numpy as np
from numpy.random import Generator, MT19937


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
            f: Callable, 
            f_dim: str = '1D',
            check_f_dims: bool = True,
            x: Union[None, np.ndarray] = None,
            x_min: float = -1., x_max: float = 1.,
            M: int = 1, N: int = 20, d: int = None,
            energy_tol: Union[float, None] = None, 
            diff_tol: Union[float, None] = None,
            max_eval: Union[int, None] = None,
            max_it: Union[int, None] = 1000,
            array_mode: str = 'numpy',
            track_list: list = None,
            save_int: int = 1,
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
        if not isinstance(f, cbx_objective):
            if f_dim != '3D' and array_mode == 'pytorch':
                raise RuntimeError('Pytorch array_mode only supported for 3D objective functions.')
            self.f = _promote_objective(f, f_dim)
        else:
            self.f = f
        
        self.num_f_eval = 0 * np.ones((self.M,), dtype=int) # number of function evaluations  
        if check_f_dims and array_mode != 'torch': # check if f returns correct shape
            x = np.random.uniform(-1,1,(self.M, self.N, self.d))
            if self.f(x).shape != (self.M,self.N):
                raise ValueError("The given objective function does not return the correct shape!")
            self.num_f_eval += N * np.ones((self.M,), dtype=int) # number of function evaluations
        self.f_min = float('inf') * np.ones((self.M,)) # minimum function value
        
        self.energy = float('inf') * np.ones((self.M, self.N)) # energy of the particles
        self.best_energy = float('inf') * np.ones((self.M,))
        self.best_particle = np.zeros((self.M,self.d))
        self.update_diff = float('inf')


        # termination parameters and checks
        self.energy_tol = energy_tol
        self.diff_tol = diff_tol
        self.max_eval = max_eval
        self.max_it = max_it
    
        self.checks = []
        if energy_tol is not None:
            self.checks.append(self.check_energy)
        if diff_tol is not None:
            self.checks.append(self.check_update_diff)
        if max_eval is not None:
            self.checks.append(self.check_max_eval)
        if max_it is not None:
            self.checks.append(self.check_max_it)
        
        self.all_check = np.zeros(self.M, dtype=bool)
        self.term_reason = {}
        self.it = 0
            
        self.track_list = track_list if track_list is not None else ['update_norm', 'energy']
        self.save_int = save_int
        self.init_history()
        
        self.update_thresh = update_thresh
        self.verbosity = verbosity 
    
    
    def pre_step(self,):
        # save old positions
        self.x_old = self.copy_particles(self.x)
        
    def inner_step(self,):
        pass
        
    def post_step(self):
        if hasattr(self, 'x_old'):
            self.update_diff = np.linalg.norm(self.x - self.x_old, axis=(-2,-1))/self.N
        
        self.update_best_cur_particle()
        self.update_best_particle()
        self.track()
            
        self.it+=1
        
    def step(self,) -> None:
        self.pre_step()
        self.inner_step()
        self.post_step()

    def optimize(self,
                 print_int: Union[int, None] = None,
                 sched = None):
        
        print_int = print_int if print_int is not None else self.save_int
        
        if self.verbosity > 0:
            print('.'*20)
            print('Starting Optimization with dynamic: ' + self.__class__.__name__)
            print('.'*20)

        if sched is None:
            sched = scheduler(self, [])

        while not self.terminate(verbosity=self.verbosity):
            self.step()
            sched.update()

            if (self.it % print_int == 0):
                self.print_cur_state()

        if self.verbosity > 0:
            print('-'*20)
            print('Finished solver.')
            print('Best energy: ' + str(self.best_energy))
            print('-'*20)

        return self.best_particle
    
    def print_cur_state(self,):
        pass

    def copy_particles(self, x):
        return copy_particles(x, mode=self.array_mode)

            
    def reset(self,):
        self.it = 0
        self.init_history()
            
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
        
        
    track_names_known = ['x', 'update_norm', 'energy']
    
    def init_history(self,):
        self.history = {}
        self.max_save_it = int(np.ceil(self.max_it/self.save_int))
        self.track_it = 0
        for key in self.track_list:
            if key not in self.track_names_known:
                raise RuntimeError('Unknown tracking key ' + key + ' specified!' +
                                   ' You can choose from the following keys '+ 
                                   str(self.track_names_known))
            else:
                getattr(self, 'track_' + key + '_init')()
            
    def track(self,):
        if self.it % self.save_int == 0:
            for key in self.track_list:
                getattr(self, 'track_' + key)()
                
            self.track_it += 1
            
    def track_x_init(self,) -> None:
        self.history['x'] = np.zeros((self.max_save_it + 1, self.M, self.N, self.d))
        self.history['x'][0, ...] = self.x
    
    def track_x(self,) -> None:
        self.history['x'][self.track_it + 1, ...] = self.copy_particles(self.x)
        
        
    def track_update_norm_init(self, ) -> None:
        self.history['update_norm'] = np.zeros((self.max_save_it, self.M,))
        
    def track_update_norm(self, ) -> None:
        self.history['update_norm'][self.track_it, :] = self.update_diff
     
    def track_energy_init(self,) -> None:
        self.history['energy'] = np.zeros((self.max_save_it, self.M,))
        
    def track_energy(self,) -> None:
        self.history['energy'][self.track_it, :] = self.best_cur_energy
    

    def update_best_cur_particle(self,) -> None:
        self.f_min = self.energy.min(axis=-1)
        self.f_min_idx = self.energy.argmin(axis=-1)
        
        if hasattr(self, 'x_old'):
            self.best_cur_particle = self.x_old[np.arange(self.M), self.f_min_idx, :]
        self.best_cur_energy = self.energy[np.arange(self.M), self.f_min_idx]
    
    def update_best_particle(self,):
        idx = np.where(self.best_energy > self.best_cur_energy)[0]
        if len(idx) > 0:
            self.best_energy[idx] = self.best_cur_energy[idx]
            self.best_particle[idx, :] = self.copy_particles(self.best_cur_particle[idx, :])
            
            
    
    
class CBXDynamic(ParticleDynamic):
    def __init__(self, f,
            noise: Union[None, Callable] = None,
            batch_args: Union[None, dict] = None,
            dt: float = 0.01, 
            alpha: float = 1.0, 
            sigma: float = 5.1,
            lamda: float = 1.0,
            max_time: Union[None, float] = None,
            correction: str = 'no_correction', 
            correction_eps: float = 1e-3,
            resampling: bool = False,
            update_thresh: float = 0.1,
            **kwargs) -> None:
        
        super().__init__(f, **kwargs)
        
        # cbx parameters
        self.dt = dt
        self.t = 0.
        self.alpha = alpha
        self.sigma = sigma
        self.lamda = lamda
        
        self.set_correction(correction)
        self.set_noise(noise)
        
        self.init_batch_idx(batch_args)
        
        self.resampling = resampling
        self.num_resampling = np.zeros((self.M,), dtype=int)
        
        self.consensus = None #consensus point
        
        
        # add max time check
        self.max_time = max_time
        if max_time is not None:
            self.checks.append(self.check_max_time)
        
    def init_batch_idx(self, batch_args) -> None:
        batch_args = batch_args if batch_args is not None else {}
        batch_size = batch_args.get('size', self.N)
        batch_partial = batch_args.get('partial', True)
        batch_seed = batch_args.get('seed', 42)
        batch_var = batch_args.get('var', 'resample')
    
        self.batch_partial = batch_partial
        self.batch_var = batch_var
        
        # set batch size
        if batch_size is None:
            self.batch_size = self.N
        else:
            self.batch_size = min(batch_size, self.N)
        
    
        if self.batch_size == self.N:
            self.batched = False
        else: # set indices for batching
            self.batched = True
            self.M_idx = np.repeat(np.arange(self.M)[:,None], self.batch_size, axis=1)
            ind = np.repeat(np.arange(self.N)[None,:], self.M ,axis=0)
            self.batch_rng = Generator(MT19937(batch_seed))#np.random.default_rng(batch_seed)
            self.indices = self.batch_rng.permuted(ind, axis=1)
            
                
    def set_batch_idx(self,):
        r"""Set batch indices

        This method sets the batch indices for each iteration of the algorithm.
        """
        if self.batched:   
            if self.indices.shape[1] < self.batch_size: # if indices are exhausted
                indices = np.repeat(np.arange(self.N)[None,:], self.M ,axis=0)
                if self.batched:
                    indices = self.batch_rng.permuted(indices, axis=1)
                    
                if self.batch_var == 'concat':
                    self.indices = np.concatenate((self.indices, indices), axis=1)
                else:
                    self.indices = indices
                #self.batch_size
    
            self.batch_idx = self.indices[:,:self.batch_size] # get batch indices
            self.indices = self.indices[:, self.batch_size:] # remove batch indices from indices
            
            self.consensus_idx = (self.M_idx, self.batch_idx, Ellipsis)
        else:
            self.consensus_idx = Ellipsis
        
        if self.batch_partial:
            self.particle_idx = self.consensus_idx
        else:
            self.particle_idx = Ellipsis
    
    
    def set_correction(self, correction):
        if correction == 'no_correction':
            self.correction = self.no_correction
        elif correction == 'heavi_side':
            self.correction = self.heavi_side
        elif correction == 'heavi_side_reg':
            self.correction = self.heavi_side_reg
        else:
            self.correction = correction
    
    def no_correction(self, x):
        return x

    def heavi_side_correction(self, x):
        z = self.energy - self.f(self.consensus)
        self.num_f_eval += self.consensus.shape[0] # update number of function evaluations

        return np.where(z > 0, 1,0)[...,None]

    def heavi_side_reg_correction(self, x):
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
        
    track_names_known = ParticleDynamic.track_names_known + ['consensus', 'drift', 'drift_mean']
    
    def track_consensus_init(self,) -> None:
        self.history['consensus'] = np.zeros((self.max_save_it, self.M, 1, self.d))
        
    def track_consensus(self,) -> None:
        self.history['consensus'][self.track_it, ...] = self.copy_particles(self.consensus)
        
    def track_drift_mean_init(self,) -> None:
        self.history['drift_mean'] = np.zeros((self.max_save_it, self.M,))
        
    def track_drift_mean(self,) -> None:
        self.history['drift_mean'][self.track_it, :] = np.mean(np.abs(self.drift), axis=(-2,-1))
        
    def track_drift_init(self,) -> None:
        self.history['drift'] = []
        self.history['particle_idx'] = []
        
    def track_drift(self,) -> None:          
        self.history['drift'].append(self.drift)
        self.history['particle_idx'].append(self.particle_idx)
        
        
    def resample(self,) -> None:
        idx = np.where(self.update_diff < self.update_thresh)[0]
        if len(idx)>0:
            z = np.random.normal(0, 1., size=(len(idx), self.N, self.d))
            self.x[idx, ...] += self.sigma * np.sqrt(self.dt) * z
            self.num_resampling[idx] += 1
            if self.verbosity > 0:
                    print('Resampled in runs ' + str(idx))
                    
    def pre_step(self,):
        # save old positions
        self.x_old = self.copy_particles(self.x)
        
        # set new batch indices
        self.set_batch_idx()
        
    def inner_step(self,):
        pass
        
    def post_step(self):
        if hasattr(self, 'x_old'):
            self.update_diff = np.linalg.norm(self.x - self.x_old, axis=(-2,-1))/self.N
        
        self.update_best_cur_particle()
        self.update_best_particle()
        self.track()

        if self.resampling:
            self.resample()
            
        self.t += self.dt
        self.it+=1
        
    def reset(self,):
        self.it = 0
        self.init_history()
        self.t = 0.
        
    def check_max_time(self):
        return self.t >= self.max_time
    
    def print_cur_state(self,):
        if self.verbosity > 0:
            print('Time: ' + "{:.3f}".format(self.t) + ', best energy: ' + str(self.f_min))
            print('Number of function evaluations: ' + str(self.num_f_eval))

        if self.verbosity > 1:
            print('Current alpha: ' + str(self.alpha))
        
        
    