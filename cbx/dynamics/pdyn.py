import warnings
#%%
from ..scheduler import scheduler, multiply
from ..utils.particle_init import init_particles
from ..utils.numpy_torch_comp import copy_particles
from ..utils.objective_handling import _promote_objective, cbx_objective

#%%
from typing import Callable, Union, Any
from numpy.typing import ArrayLike
import numpy as np
from numpy.random import Generator, MT19937
from scipy.special import logsumexp


class ParticleDynamic:
    r"""The base particle dynamic class

    This class implements the base particle dynamic class. This class does not implement 
    any functionality that is specific to consensus schemes. The only necessary argument is the
    objective function :math:`f`. The objective can be given as a simple callable function. It 
    is assumed to be a function :math:`f:\mathbb{R}^{d}\to\mathbb{R}`. If the objective can handle 
    arrays of shape :math:`(N, d)`, i.e., it can be applied to multiple particles, then you can use the 
    argument ``f_dim=2D``, or analogously ``f_dim=3D``, if it can handle arrays of shape :math:`(M, N, d)`. 
    You can also directly provide the objective function as a :class:`cbx_objective` instance.
    
    During the initialization of the class, the dimension of the output is checked using the :meth:`check_f_dims` method. 
    If the check fails, an error is raised. This check can be turned off by setting the ``check_f_dims`` parameter to ``False``.

    The ensemble of particles is modeled as an array of shape :math:`(M, N, d)` where 

    * :math:`M` is the number of runs that should be performed. In many cases, multiple runs can be implmented on the 
    ``numpy`` array level, which allows for efficient evaluations and computations avoiding for loops.

    * :math:`N` is the number of particles in each run.

    * :math:`d` is the dimension of the system.

    This means that :math:`x_{m,n}\in\mathbb{R}^{d}` denotes the position of the :math:`n`-th particle of the :math:`m`-th run.

    Promoting the objective function :math:`f` to the :class:`cbx_objective` instance :math:`\tilde{f}` allows for the evaluation on 
    ensembles of the shape :math:`(M, N, d)`, namely,

    .. math::

        \tilde{f}(x)_{m,n} := f(x_{m,n,:})


    The class can not infer the dimension of the problem. Therfore, the dimension must be specified by the parameter ``d`` or 
    by specifiying an inintial position array ``x``.

    Parameters
    ----------
    f : Callable
        The objective function :math:`f` of the system.
    f_dim : str, optional
        The dimensionality of the objective function. The default is '1D'.
    check_f_dims : bool, optional
        If ``True``, the dimension of the objective function is checked. The default is ``True``.
    x : array_like, shape (M, N, d) or None, optional
        The initial positions of the particles. For a system of :math:`N` particles ``x[m, n, :]``
        represents the position n-th particle of the m-th run.
    x_min : float, optional
        The minimum value of the initial positions. The default is -1.
    x_max : float, optional
        The maximum value of the initial positions. The default is 1.
    M : int, optional
        The number of runs. The default is 1.
    N : int, optional
        The number of particles in each run. The default is 20.
    d : int or None, optional
        The dimension of the system. The default is None.
    energy_tol : float, optional
        The energy tolerance. The default is None.
    diff_tol : float, optional
        The difference tolerance. The default is None.
    max_eval : int, optional
        The maximum number of evaluations. The default is None.
    max_it : int, optional
        The maximum number of iterations. The default is 1000.
    max_x_thresh : float, optional
        The maximum value of the absolute value of the position. The default is 1e5.
    array_mode : str, optional
        The mode of the array. The default is 'numpy'.
    track_list : list, optional
        The list of objects that are tracked. The default is None.
        Possible objects to track are the following:
        * 'x': The positions of the particles.
        * 'update_norm': The norm of the particle update.
        * 'energy': The energy of the system.
    save_int : int, optional
        The frequency of the saving of the data. The default is 1.
    verbosity : int, optional
        The verbosity level. The default is 1.
    
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
            max_x_thresh: Union[float, None] = 1e5,
            array_mode: str = 'numpy',
            track_list: list = None,
            save_int: int = 1,
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
        self.f_min = float('inf') * np.ones((self.M,)) # minimum function value
        self.check_f_dims(check=check_f_dims) # check the dimension of the objective function

        self.energy = float('inf') * np.ones((self.M, self.N)) # energy of the particles
        self.best_energy = float('inf') * np.ones((self.M,))
        self.best_particle = np.zeros((self.M,self.d))
        self.update_diff = float('inf')


        # termination parameters and checks
        self.energy_tol = energy_tol
        self.diff_tol = diff_tol
        self.max_eval = max_eval
        self.max_it = max_it
        self.max_x_thresh = max_x_thresh
    
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
        
        self.verbosity = verbosity 
    
    def check_f_dims(self, check=True) -> None:
        """
        Check the dimensions of the objective function output.

        Parameters:
            check (bool): Flag indicating whether to perform the dimension check. Default is True.

        Returns:
            None
        """
        if check and (self.array_mode != 'torch'): # check if f returns correct shape
            x = np.random.uniform(-1,1,(self.M, self.N, self.d))
            if self.f(x).shape != (self.M,self.N):
                raise ValueError("The given objective function does not return the correct shape!")
            self.num_f_eval += self.N * np.ones((self.M,), dtype=int) # number of function evaluations
    def pre_step(self,):
        """
        The pre-step function. This function is used in meth:`step` before the inner step is performed.
        This function can be overwritten by subclasses to perform any pre-steps before the inner step.

        Parameters:
            None

        Returns:
            None
        """
        # save old positions
        self.x_old = self.copy_particles(self.x)
        
    def inner_step(self,):
        """
        This function is used in meth:`step` to perform the inner step. This function implements the actual
        update of the dynamic and is therfore the most important function in the dynamics class.
        This function should be overwritten by subclasses to perform any inner steps.

        Parameters:
            None
        
        Returns:
            None
        """
        pass
        
    def post_step(self):
        """
        Execute the post-step operations after each iteration.

        This function updates the difference between the current particle position and the previous particle position if the 'x_old' 
        attribute exists in the object. The difference is computed using the numpy.linalg.norm 
        function along the specified axes and is divided by the total number of particles.

        The function then calls the 'update_best_cur_particle' method to update the best current particle position.
        Next, it calls the 'update_best_particle' method to update the best particle position.
        After that, it calls the 'track' method to track the progress of the optimization algorithm.
        Then, it calls the 'process_particles' method to perform any necessary post-processing operations on the particles.

        Finally, the function increments the 'it' attribute by 1 to keep track of the current iteration.

        Parameters:
            None

        Return:
            None
        """
        if hasattr(self, 'x_old'):
            self.update_diff = np.linalg.norm(self.x - self.x_old, axis=(-2,-1))/self.N
        
        self.update_best_cur_particle()
        self.update_best_particle()
        self.track()
        self.process_particles()
            
        self.it+=1
        
    def process_particles(self,):
        """
        Process the particles.

        This function performs some operations on the particles. 

        Parameters:
            None

        Return:
            None
        """
        np.nan_to_num(self.x, copy=False, nan=self.max_x_thresh)
        self.x = np.clip(self.x, None, self.max_x_thresh)
        
    def step(self,) -> None:
        """
        Execute a step in the dynamic.

        This function performs the following actions:
        1. Calls the `pre_step` method to perform any necessary pre-step operations.
        2. Calls the `inner_step` method to perform the main step operation.
        3. Calls the `post_step` method to perform any necessary post-step operations.

        Parameters:
            None

        Returns:
            None
        """
        self.pre_step()
        self.inner_step()
        self.post_step()
        
    def default_sched(self,):
        """
        A function that returns a scheduler object with an empty list as the parameter.

        Parameters:
            None
        Returns:
            scheduler: The scheduler object.
        """
        return scheduler(self, [])

    def optimize(self,
                 print_int: Union[int, None] = None,
                 sched = 'default'):
        """
        Optimize the function using the dynmaic. This function perfoms the iterations as specified by step method and the update in :meth:`inner_step`.

        Parameters:
            print_int : int, optional 
                The interval at which to print the current state of the optimization. If not provided, the default value is used. Defaults to None.
            sched : str
                The scheduler to use for the optimization. If set to 'default', the default scheduler is used. 
                If set to None, a scheduler is created based on the current optimization parameters. Defaults to 'default'.

        Returns:
            best_particle: The best particle found during the optimization process.
        """
        
        print_int = print_int if print_int is not None else self.save_int
        
        if self.verbosity > 0:
            print('.'*20)
            print('Starting Optimization with dynamic: ' + self.__class__.__name__)
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
            if (self.it % print_int == 0):
                self.print_cur_state()

        self.print_post_opt()


        return self.best_particle
    
    def print_cur_state(self,):
        """
        Print the current state.

        Parameters:
            None

        Returns:
            None
        """
        pass
    
    def print_post_opt(self):
        """
        Print the optimization results if the verbosity level is greater than 0.
        
        This function prints the solver's finishing message, the best energy found during optimization.
        
        Parameters:
            None
        
        Return:
            None
        """
        if self.verbosity > 0:
            print('-'*20)
            print('Finished solver.')
            print('Best energy: ' + str(self.best_energy))
            print('-'*20)
            
    def copy_particles(self, x):
        """
        Copy particles from one location to another. This is necessary to be compatible with torch arrays.

        Parameters:
            x: The location of the particles to be copied.

        Returns:
            The copied particles.

        Note:
            This function uses the `copy_particles` function with the `array_mode` set to the class attribute `self.array_mode`.
        """
        return copy_particles(x, mode=self.array_mode)

            
    def reset(self,):
        """
        Reset the state of the object.

        This function sets the value of the object's 'it' attribute to 0 and calls the 'init_history()' method to re-initialize the history of the object.

        Parameters:
            self (object): The object instance.

        Returns:
            None
        """
        self.it = 0
        self.init_history()
            
    def check_energy(self):
        """
        Check if the energy is below a certain tolerance.

        Returns:
            bool: True if the energy is below the tolerance, False otherwise.
        """
        return self.f_min < self.energy_tol
    
    def check_update_diff(self):
        """
        Checks if the update difference is less than the difference tolerance.

        Returns:
            bool: True if the update difference is less than the difference tolerance, False otherwise.
        """
        return self.update_diff < self.diff_tol
    
    def check_max_eval(self):
        """
        Check if the number of function evaluations is greater than or equal to the maximum number of evaluations.

        Returns:
            bool: True if the number of function evaluations is greater than or equal to the maximum number of evaluations, False otherwise.
        """
        return self.num_f_eval >= self.max_eval
    
    def check_max_it(self):
        """
        Checks if the current value of `self.it` is greater than or equal to the value of `self.max_it`.

        Returns:
            bool: True if `self.it` is greater than or equal to `self.max_it`, False otherwise.
        """
        return self.it >= self.max_it
    
    def terminate(self, verbosity = 0):
        """
        Terminate the process and return a boolean value indicating if for each run the termination criterion was met.

        Parameters:
            verbosity (int): The level of verbosity for printing information. Default is 0.

        Returns:
            bool: True if all checks passed, False otherwise.
        """
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
        
    track_names_known = ['x', 'update_norm', 'energy'] # known tracking keys
    
    def init_history(self,):
        """
        Initialize the history dictionary and initialize the specified tracking keys.

        Parameters:
            None

        Returns:
            None
        """
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
        """
        Track the progress of the object.

        Parameters:
            None

        Returns:
            None
        """
        if self.it % self.save_int == 0:
            for key in self.track_list:
                getattr(self, 'track_' + key)()
                
            self.track_it += 1
            
    def track_x_init(self,) -> None:
        """
        Initializes the tracking of variable 'x' in the history dictionary.

        Parameters:
            None

        Returns:
            None
        """
        self.history['x'] = np.zeros((self.max_save_it + 1, self.M, self.N, self.d))
        self.history['x'][0, ...] = self.x
    
    def track_x(self,) -> None:
        """
        Update the history of the 'x' variable by copying the current particles to the next time step.

        Parameters:
            None

        Returns:
            None
        """
        self.history['x'][self.track_it + 1, ...] = self.copy_particles(self.x)
        
        
    def track_update_norm_init(self, ) -> None:
        """
        Initializes the 'update_norm' entry in the 'history' dictionary.

        Returns:
            None
        """
        self.history['update_norm'] = np.zeros((self.max_save_it, self.M,))
        
    def track_update_norm(self, ) -> None:
        """
        Updates the 'update_norm' entry in the 'history' dictionary with the 'update_diff' value.

        Parameters:
            None

        Returns:
            None
        """
        self.history['update_norm'][self.track_it, :] = self.update_diff
     
    def track_energy_init(self,) -> None:
        """
        Initializes the energy tracking for the dynamic.

        Parameters:
            None

        Returns:
            None
        """
        self.history['energy'] = np.zeros((self.max_save_it, self.M,))
        
    def track_energy(self,) -> None:
        """
        Updates the 'energy' history array with the current best energy value.

        Returns:
            None: This function does not return anything.
        """
        self.history['energy'][self.track_it, :] = self.best_cur_energy
    

    def update_best_cur_particle(self,) -> None:
        """
        Updates the best current particle and its energy based on the minimum energy found in the energy matrix.
        
        Parameters:
            None
        
        Returns:
            None
        """
        self.f_min = self.energy.min(axis=-1)
        self.f_min_idx = self.energy.argmin(axis=-1)
        
        if hasattr(self, 'x_old'):
            self.best_cur_particle = self.x_old[np.arange(self.M), self.f_min_idx, :]
        self.best_cur_energy = self.energy[np.arange(self.M), self.f_min_idx]
    
    def update_best_particle(self,):
        """
        Updates the best particle and best energy of the whole iteration.

        Parameters:
            None
        
        Returns:
            None
        """
        idx = np.where(self.best_energy > self.best_cur_energy)[0]
        if len(idx) > 0:
            self.best_energy[idx] = self.best_cur_energy[idx]
            self.best_particle[idx, :] = self.copy_particles(self.best_cur_particle[idx, :])
            

            
def compute_mat_sqrt(A):
    """
    Compute the square root of a matrix.

    Parameters:
        A (np.ndarray): The input matrix.

    Returns:
        np.ndarray: The square root of the matrix.
    """
    B, V = np.linalg.eigh(A)
    B = np.maximum(B,0.)
    return V@(np.sqrt(B)[...,None]*V.transpose(0,2,1))
    
    
class CBXDynamic(ParticleDynamic):
    r"""The base class for consensus based dynamics

    This class implements functionality that is specific to consensus based dynamics. It inherits from the ParticleDynamic class.

    Parameters:
        f: Callable
            The function to optimize.
        noise: str, Callable, or None, optional
            The noise function. A string can be one of 'isotropic', 'anisotropic', or 'sampling'. It is technically possible 
            to use a Callable instead of a string, but this is not recommended. A custom noise model should be implemented 
            by subclassing this class and adding it as a instance method. Default: None.
        batch_args: dict, optional
            The batch arguments. Default: None.
        dt: float, optional
            The time step size :math:`dt` of the dynamic. Default: 0.1.
        alpha: float, optional
            The alpha parameter :math:`\alpha` of the dynamic. Default: 1.0.
        sigma: float, optional
            The sigma parameter :math:`\sigma` of the dynamic, scaling the noise. Default: 5.1.
        lamda: float, optional
            The lamda parameter :math:`\lambda` of the dynamic. Default: 1.0.
        max_time: float, optional
            The maximum time to run the dynamic. Default: None.
        correction: str, optional
            The correction method. Default: 'no_correction'. One of 'no_correction', 'heavi_side', 'heavi_side_reg'.
        correction_eps: float, optional
            The parameter :math:`\epsilon` for the regularized correction. Default: 1e-3.
        resampling: bool, optional
            Whether to use resampling. Default: False.
        update_thresh: float, optional
            The threshold for resampling scheme. Default: 0.1

    Returns:
        None
    """
    def __init__(self, f,
            noise: Union[None, str, Callable] = None,
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
        
        self.correction_eps = correction_eps
        self.set_correction(correction)
        self.set_noise(noise)
        
        self.init_batch_idx(batch_args)
        
        self.resampling = resampling
        self.update_thresh = update_thresh
        self.num_resampling = np.zeros((self.M,), dtype=int)
        
        self.consensus = None #consensus point
        
        
        # add max time check
        self.max_time = max_time
        if max_time is not None:
            self.checks.append(self.check_max_time)
        
    def init_batch_idx(self, batch_args) -> None:
        """
        Initializes the batch index for the given batch arguments.
        
        Parameters:
            batch_args: dict.
                Dictionary containing the batch arguments.
                
                - size : int, optional
                    The size of each batch. Defaults to self.N.
                - partial : bool, optional
                    Whether to allow partial batches. Defaults to True.
                - seed : int, optional
                    The seed for random number generation. Defaults to 42.
                - var : str, optional
                    The variable to use for resampling. Defaults to 'resample'.
        
        Returns:
            None
        """
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
        """
        Set the batch index for the data.

        This method sets the batch index batched dynamics. 
        If the indices are exhausted, it generates new indices using `np.repeat` and `np.arange`. 

        If the `self.batch_var` is set to `'concat'`, we concatenate the new indices with the existing indices using `np.concatenate`, 
        otherwise, it replaces the existing indices with the new indices. 

        After updating the indices, we set the `batch_idx` to the first `batch_size` columns of the indices, 
        and remove these columns from the indices. 

        If `batch_partial` is `True`, we set `particle_idx` to `consensus_idx`, otherwise, to `Ellipsis`.

        Parameters:
            None

        Returns:
            None
        """
        if self.batched:   
            if self.indices.shape[1] < self.batch_size: # if indices are exhausted
                indices = np.repeat(np.arange(self.N)[None,:], self.M ,axis=0)
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
            
    def default_sched(self,) -> scheduler:
        """
        A function that returns a default scheduler.

        Parameters:
            None
        
        Returns:
            scheduler: 
                The scheduler object.
        """

        return scheduler(self, [multiply(name='alpha', factor=1.05)])
    
    
    def set_correction(self, correction):
        """
        Set the correction method for the object.
        
        Parameters:
            correction: str.
                The correction method to be set. Must be one of 
                
                * 'no_correction', 
                * 'heavi_side', 
                * 'heavi_side_reg'.
        
        Returns:
            None
        """
        if correction == 'no_correction':
            self.correction = self.no_correction
        elif correction == 'heavi_side':
            self.correction = self.heavi_side
        elif correction == 'heavi_side_reg':
            self.correction = self.heavi_side_reg
        else:
            self.correction = correction
    
    def no_correction(self, x:Any) -> Any:
        """
        The function if no correction is specified. Is equla to the identity.

        Parameters:
            x: The input value.

        Returns:
            The input value without any correction.
        """
        return x

    def heavi_side_correction(self, x:ArrayLike) -> ArrayLike:
        """
        Calculate the Heaviside correction for the given input.

        Parameters:
            x : ndarray
                The input array.

        Returns:
            ndarray: The result of the Heaviside correction.
        """
        z = self.energy - self.f(self.consensus)
        self.num_f_eval += self.consensus.shape[0] # update number of function evaluations

        return x * np.where(z > 0, 1,0)[...,None]

    def heavi_side_reg_correction(self, x:ArrayLike) -> ArrayLike:
        """
        Calculate the Heaviside regularized correction.

        Parameters:
            x : ndarray
                The input array.

        Returns:
            ndarray: The Heaviside regularized correction value.
        """
        z = self.energy - self.f(self.consensus)
        self.num_f_eval += self.consensus.shape[0] # update number of function evaluations

        return x * (0.5 + 0.5 * np.tanh(z/self.correction_eps))
    
    def set_noise(self, noise) -> None:
        """
        Set the noise model for the object.

        Parameters:
            noise (str or Callable or None): The type of noise model to be set. Can be one of the following:
                - 'isotropic' or None: Set the isotropic noise model.
                - 'anisotropic': Set the anisotropic noise model.
                - 'sampling': Set the sampling noise model.
                - 'covariance': Set the covariance noise model.
                - else: use the given input as a callable.

        Returns:
            None

        Warnings:
            If 'noise' is set to a custom value, a warning will be raised to indicate that it is not the recommended way to choose a custom noise model.
        """
        # set noise model
        if noise == 'isotropic' or noise is None:
            self.noise = self.isotropic_noise
        elif noise == 'anisotropic':
            self.noise = self.anisotropic_noise
        elif noise in ['sampling','covariance']:
            self.noise = self.covariance_noise
        else:
            warnings.warn('Custom noise specified. This is not the recommended\
                          for choosing a custom noise model.', stacklevel=2)
            self.noise = noise

    def anisotropic_noise(self,) -> ArrayLike:
        r"""
        This function implements the anisotropic noise model. From the drift :math:`d = x - c(x)`,
        the noise vector is computed as

        .. math::

            n_{m,n} = \sqrt{dt}\cdot d_{m,n} \cdot \xi.

        Here, :math:`\xi` is a random vector of size :math:`(d)` distributed according to :math:`\mathcal{N}(0,1)`.

        Returns:
            numpy.ndarray: The generated noise.

        Note
        ----
        The plain drift is used for the noise. Therefore, the noise vector is scaled with a different factor in each dimension, 
        which motivates the name **anisotropic**.
        """
        
        z = np.random.normal(0, 1, size=self.drift.shape) * self.drift
        return np.sqrt(self.dt) * z
        
    def isotropic_noise(self,) -> ArrayLike:
        r"""

        This function implements the isotropic noise model. From the drift :math:`d = x - c(x)`,
        the noise vector is computed as

        .. math::

            n_{m,n} = \sqrt{dt}\cdot \|d_{m,n}\|_2\cdot \xi.


        Here, :math:`\xi` is a random vector of size :math:`(d)` distributed according to :math:`\mathcal{N}(0,1)`.
        
        Parameters
        ----------
        None
        
        Note
        ----
        Only the norm of the drift is used for the noise. Therefore, the noise vector is scaled with the same factor in each dimension, 
        which motivates the name **isotropic**. 
        """

        z = np.sqrt(self.dt) * np.random.normal(0, 1, size=(self.drift.shape))
        return z * np.linalg.norm(self.drift, axis=-1,keepdims=True)
        
    def covariance_noise(self,) -> ArrayLike:
        r"""

        This function implements the covariance noise model. Given the covariance matrix :math:`\text{Cov}(x)\in\mathbb{R}^{M\times d\times d}` of the ensemble,
        the noise vector is computed as

        .. math::

            n_{m,n} = \sqrt{(1/\lambda)\cdot (1-\exp(-dt))^2} \cdot \sqrt{\text{Cov}(x)}\xi.

        Here, :math:`\xi` is a random vector of size :math:`(d)` distributed according to :math:`\mathcal{N}(0,1)`.

        Returns:
            ArrayLike: The covariance noise.
        """
        
        z = np.random.normal(0, 1, size = self.drift.shape) 
        noise = self.apply_cov_sqrt(z)
        
        factor = np.sqrt((1/self.lamda) * (1 - np.exp(-self.dt)**2))
        return factor * noise
    
    def apply_cov_sqrt(self, z:ArrayLike) -> ArrayLike:
        """
        Applies the square root of the covariance matrix to the input tensor.

        Args:
            z (ArrayLike): The input tensor of shape (batch_size, num_features, seq_length).

        Returns:
            ArrayLike: The output of the matrix-vector product.
        """
        return (self.Cov_sqrt@z.transpose(0,2,1)).transpose(0,2,1)
    
    def update_covariance(self,) -> None:
        r"""Update the covariance matrix :math:`\text{Cov}(x)` of the noise model
    
        Parameters

    
        Returns
        -------
        None.
    
        """                       
        weights = - self.alpha * self.energy
        coeffs = np.exp(weights - logsumexp(weights, axis=(-1,), keepdims=True))
      
        D = self.drift[...,None] * self.drift[...,None,:]
        D = np.sum(D * coeffs[..., None, None], axis = -3)
        self.Cov_sqrt = compute_mat_sqrt(D)
        
        
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
            print('Time: ' + "{:.3f}".format(self.t) + ', best current energy: ' + str(self.f_min))
            print('Number of function evaluations: ' + str(self.num_f_eval))

        if self.verbosity > 1:
            print('Current alpha: ' + str(self.alpha))
            
    def compute_consensus(self, x_batch) -> None:
        r"""Updates the weighted mean of the particles.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # evaluation of objective function on batch
        energy = self.f(x_batch) # update energy
        self.num_f_eval += np.ones(self.M,dtype=int) * x_batch.shape[-2] # update number of function evaluations
        
        weights = - self.alpha * energy
        coeffs = np.exp(weights - logsumexp(weights, axis=(-1,), keepdims=True))[...,None]
        
        problem_idx = np.where(np.abs(coeffs.sum(axis=-2)-1) > 0.1)[0]
        if len(problem_idx) > 0:
            raise RuntimeError('Problematic consensus computation!')
        
        return (x_batch * coeffs).sum(axis=-2, keepdims=True), energy
        
        
    