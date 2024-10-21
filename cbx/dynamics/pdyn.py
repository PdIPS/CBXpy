#%%
from ..noise import get_noise
from ..correction import get_correction
from ..scheduler import scheduler, multiply, effective_sample_size
from ..utils.termination import max_it_term
from ..utils.history import track_x, track_energy, track_update_norm, track_consensus, track_drift, track_drift_mean
from cbx.utils.objective_handling import _promote_objective

#%%
from pprint import pformat
from typing import Callable, Union, List
from numpy.typing import ArrayLike
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.random import Generator, MT19937
from scipy.special import logsumexp as logsumexp_scp

        
class post_process_default:
    """
    Default post processing.

    This function performs some operations on the particles, after the inner step. 

    Parameters:
        None

    Return:
        None
    """
    def __init__(self, max_thresh: float = 1e8):
        self.max_thresh = max_thresh
    
    def __call__(self, dyn):
        np.nan_to_num(dyn.x, copy=False, nan=self.max_thresh)
        dyn.x = np.clip(dyn.x, -self.max_thresh, self.max_thresh)

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
    term_criteria : list[Callable], optional
        A list of callables that determine the termination of the optimization. Each callable in the list should accept a single argument (the dynamic object) 
        and return a numpy array of size (M,), where term(dyn)[i] specifies, whether the optimization should be terminated for the i-th run.
    track_args : dict
        The arguments for the tracking certain objects in the history. The following keys are possible:
        
        * 'names' : list
            The names of the objects that should be tracked.
        * 'save_int' : int
            The frequency of the saving of the data. The default is 1.
        * 'extra_tracks' : list
            A list of extra tracks that should be performed. Each object in this list must have init_history and an update method.

    post_process : Callable
        A callbale acting on the dynamic that should be performed after each optimization step.
    copy : Callable
        A callable that copies an array. The default is ``np.copy``.
    norm : Callable
        A callable that computes the norm of an array. The default is ``np.linalg.norm``.
    sampler : Callable
        A callable that generates an array of random numbers. The default is ``np.random.normal``.

    verbosity : int, optional
        The verbosity level. The default is 1.
    
    """

    def __init__(
            self, 
            f: Callable, 
            f_dim: str = '1D',
            check_f_dims: bool = True,
            x: Union[None, np.ndarray] = None,
            M: int = 1, N: int = 20, d: int = None,
            max_it: int = 1000,
            term_criteria: List[Callable] = None,
            track_args: list = None,
            verbosity: int = 1,
            copy: Callable = None,
            norm: Callable = None,
            sampler: Callable = None,
            post_process: Callable = None,
            seed: int = None
            ) -> None:
        
        self.verbosity = verbosity
        self.seed = seed
        
        # set array backend funs
        self.set_array_backend_funs(copy, norm, sampler)

        # init particles    
        self.init_x(x, M, N, d)
        
        # set and promote objective function
        self.init_f(f, f_dim, check_f_dims)

        self.energy = float('inf') * np.ones((self.M,self.N)) # energy of the particles
        self.best_energy, self.best_cur_energy = [float('inf') * np.ones(self.M,) for _ in [0,1]]
        self.best_particle = self.copy(self.x[:, 0, :])
        self.update_diff = float('inf') * np.ones((self.M,))


        # termination parameters and checks
        self.init_term(term_criteria, max_it)
        self.it = 0
        self.init_history(track_args)
        
        # post processing
        self.set_post_process(post_process)
        
    def set_post_process(self, post_process):
        self.post_process = post_process if post_process is not None else post_process_default()

    def set_array_backend_funs(self, copy, norm, sampler):
        self.copy = copy if copy is not None else np.copy 
        self.norm = norm if norm is not None else np.linalg.norm
        rng = np.random.default_rng(self.seed)
        self.sampler = sampler if sampler is not None else rng.standard_normal
        
    def init_x(self, x, M, N, d):
        """
        Initialize the particle system with the given parameters.

        Parameters:
            x: the initial particle system. If x is None, the dimension d must be specified and the particle system is initialized randomly. 
               If x is specified, it is broadcasted to the correct shape (M,N,d).
            M: the number of particles in the first dimension
            N: the number of particles in the second dimension
            d: the dimension of the particle system
            x_min: the minimum value for x
            x_max: the maximum value for x

        Returns:
            None
        """
        if x is None:
            if d is None:
                raise RuntimeError('If the inital partical system is not given, the dimension d must be specified!')
            self.x = self.init_particles(shape=(M, N, d))
            
        else: # if x is given correct shape
            if x.ndim == 1:
                x = x[None, None, :]
            elif x.ndim == 2:
                x = x[None, ...]
            self.x = self.copy(x)
        
        self.M, self.N = self.x.shape[:2]
        self.d = self.x.shape[2:]
        self.ddims = tuple(i for i in range(2, self.x.ndim))
        
        
    def init_particles(self, shape=None):
        return np.random.uniform(-1., 1., size=shape)

    def init_f(self, f, f_dim, check_f_dims):
        self.f = _promote_objective(f, f_dim)
                
        self.num_f_eval = 0 * np.ones((self.M,), dtype=int) # number of function evaluations  
        self.f_min = float('inf') * np.ones((self.M,)) # minimum function value
        self.check_f_dims(check=check_f_dims) # check the dimension of the objective function

    def check_f_dims(self, check=True) -> None:
        """
        Check the dimensions of the objective function output.

        Parameters:
            check (bool): Flag indicating whether to perform the dimension check. Default is True.

        Returns:
            None
        """
        if check: # check if f returns correct shape
            x = self.sampler(size=self.x.shape)
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
        self.x_old = self.copy(self.x)
        
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
            self.update_diff = self.norm(self.x - self.x_old, axis=(-2,-1))/self.N
        
        self.update_best_cur_particle()
        self.update_best_particle()
        self.track()
        self.post_process(self)
        self.it+=1
        
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
        Returns the default scheduler for a dynamic.

        Parameters:
            None
        Returns:
            scheduler: The scheduler object.
        """
        return scheduler([])

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
            sched = scheduler([])
        elif sched == 'default':
            sched = self.default_sched()
        elif sched == 'effective':
            sched = effective_sample_size()
        else:
            self.sched = sched

        while not self.terminate():
            self.step()
            sched.update(self)
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

    def init_term(self, term_criteria, max_it):
        """
        Initialize the termination criteria of the object.

        This function sets the value of the object's 'it' attribute to 0 and calls the 'init_history()' method to re-initialize the history of the object.

        Parameters
        ----------
            checks : list

        """
        self.term_criteria = term_criteria if term_criteria is not None else [max_it_term(max_it)]
        self.term_reason = [None for i in range((self.M))]
        self.active_runs_idx = np.arange(self.M)
        self.num_active_runs = self.M
    
    def terminate(self,):
        self.select_active_runs()
        return self.num_active_runs <= 0
    
    def select_active_runs(self,):
        """
        Selects the inidices that meet a termination criterion.

        Parameters:
            None

        Returns:
            bool: True if all checks passed, False otherwise.
        """

        loc_term = np.zeros((self.M, len(self.term_criteria)), dtype=bool)
        for i, term in enumerate(self.term_criteria):
            loc_term[:, i] = term(self)
            
        terms = np.sum(loc_term, axis=1)
        self.active_runs_idx = np.where(terms==0)[0]
        self.num_active_runs = self.active_runs_idx.shape[0]
            
        for j in range(self.M):
            if terms[j]:
                self.term_reason[j] = np.where(loc_term[j,:])[0]
    
    known_tracks = {
        'update_norm': track_update_norm,
        'energy': track_energy,
        'x': track_x
    }
    def init_history(self, track_args: dict):
        """
        Initialize the history dictionary and initialize the specified tracking keys.

        Parameters:
            None

        Returns:
            None
        """
        track_args = track_args if track_args else {}  
        track_names = track_args.get('names', ['update_norm', 'energy'])
        extra_tracks = track_args.get('extra_tracks', [])
        self.save_int = track_args.get('save_int', 1)
        self.history = {}
        self.tracks = extra_tracks
        self.track_it = 0
        for key in track_names:
            if key in self.known_tracks.keys():
                self.tracks.append(self.known_tracks[key]())
            else:
                raise RuntimeError('Unknown tracking key ' + key + ' specified!' +
                        ' You can choose from the following keys '+ 
                        str(self.known_tracks.keys()))
            
        for track in self.tracks:
            track.init_history(self)

    def track(self,):
        """
        Track the progress of the object.

        Parameters:
            None

        Returns:
            None
        """
        if self.it % self.save_int == 0:
            for track in self.tracks:
                track.update(self)
            self.track_it += 1

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
        else:
            self.best_cur_particle = self.x[np.arange(self.M), self.f_min_idx, :]
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
            self.best_particle[idx, :] = self.copy(self.best_cur_particle[idx, :])
    
    print_vars = [
        'M', 'N', 'd', 'term_criteria',
    ]
    
    def __repr__(self):
        v_dict = {k:getattr(self,k) for k in self.print_vars}
        return str(type(self)) + '\n' + pformat(v_dict, indent=4, width=1)
              
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
    return V@(np.sqrt(B)[...,None]*np.moveaxis(V, -1, -2))

class compute_consensus_default:
    def __init__(self, check_coeffs = False):
        if check_coeffs:
            self.check_coeffs = self._check_coeffs
        else:
            self.check_coeffs = self._no_check_coeffs
    
    def __call__(self, energy, x, alpha):
        weights = - alpha * energy
        coeff_expan = tuple([Ellipsis] + [None for i in range(x.ndim-2)])
        coeffs = np.exp(weights - logsumexp_scp(weights, axis=-1, keepdims=True))[coeff_expan]
        self.check_coeffs(coeffs)
        return (x * coeffs).sum(axis=1, keepdims=True), energy
    
    def _check_coeffs(self, coeffs):
        problem_idx = np.where(np.abs(coeffs.sum(axis=1)-1) > 0.1)[0]
        if len(problem_idx) > 0:
            raise RuntimeError('Problematic consensus computation!')
    
    def _no_check_coeffs(self, coeffs):
        pass
    
    
class CBXDynamic(ParticleDynamic):
    r"""The base class for consensus based dynamics

    This class implements functionality that is specific to consensus based dynamics. It inherits from the ParticleDynamic class.

    Parameters:
        f: Callable
            The function to optimize.
        noise: str or Callable, optional
            A string can be one of 'isotropic', 'anisotropic', or 'sampling'. It is also possible 
            to use a Callable instead of a string. This Callable needs to accept a single argument, which is the dynamic object. Default: 'isotropic'.
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
        correction: str or Callable, optional
            The correction method. Default: 'no_correction'. One of 'no_correction', 'heavi_side', 'heavi_side_reg' or a Callable.
        correction_eps: float, optional
            The parameter :math:`\epsilon` for the regularized correction. Default: 1e-3.

    Returns:
        None
    """
    def __init__(self, f,
            noise: Union[str, Callable] = 'isotropic',
            batch_args: Union[None, dict] = None,
            dt: float = 0.01, 
            alpha: float = 1.0, 
            sigma: float = 5.1,
            lamda: float = 1.0,
            correction: Union[str, None] = 'no_correction', 
            correction_eps: float = 1e-3,
            compute_consensus: Callable = None,
            **kwargs) -> None:
        
        super().__init__(f, **kwargs)
        
        # cbx parameters
        self.dt = dt
        self.t = 0.
        self.init_alpha(alpha)
        self.sigma = sigma
        self.lamda = lamda
        
        self.correction_eps = correction_eps
        self.set_correction(correction)
        self.set_noise(noise)
        
        self.init_batch_idx(batch_args)
        self.init_consensus(compute_consensus)
        
    known_tracks = {
        'consensus': track_consensus,
        'drift_mean': track_drift_mean,
        'drift': track_drift,
        **ParticleDynamic.known_tracks,}
    
    def init_consensus(self, compute_consensus):
        self.consensus = None #consensus point
        self._compute_consensus = compute_consensus if compute_consensus is not None else compute_consensus_default()
    
    def init_alpha(self, alpha):
        '''
        Initialize alpha per batch. If alpha is a float it is broadcasted to an array similar to x with dimensions (x.shape[0], 1). 
        Otherwise, it is set to the given parameter.
        
        Parameters:
            alpha:
                The initial value of alpha
        Returns:
            None
        '''
        if isinstance(alpha, (int, float)):
            Mslice = tuple([Ellipsis] + [0 for i in range(self.x.ndim-1)])
            self.alpha = self.copy(self.x[Mslice])[:, None]
            self.alpha[:,0] = alpha
        else:
            self.alpha = alpha
            
    def get_reshaped_run_idx(self,):
        return as_strided(self.active_runs_idx, shape=(self.num_active_runs, self.batch_size), strides=(self.active_runs_idx.strides[0],0))
        
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
            self.set_batch_idx = self.set_batch_idx_unbatched
        else: # set indices for batching
            self.batched = True
            #self.M_idx = np.repeat(np.arange(self.M)[:,None], self.batch_size, axis=1)
            ind = np.repeat(np.arange(self.N)[None,:], self.M ,axis=0)
            self.batch_rng = Generator(MT19937(batch_seed))#np.random.default_rng(batch_seed)
            self.indices = self.batch_rng.permuted(ind, axis=1)
            self.set_batch_idx = self.set_batch_idx_batched
            
        self.consensus_idx = Ellipsis
        self.particle_idx  = Ellipsis
            
                
    def set_batch_idx_unbatched(self,):
        """
        Set the batch index for the particles.

        This method sets the indices for unbatched dynamics.

        Parameters:
            None

        Returns:
            None
        """ 

        if self.num_active_runs == self.M:
            self.consensus_idx = Ellipsis
        else:
            self.consensus_idx = (self.active_runs_idx, Ellipsis)
            
        self.particle_idx = self.consensus_idx
        
    def set_batch_idx_batched(self,):
        """
        Set the batch index for the particles.

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
        if self.indices.shape[1] < self.batch_size: # if indices are exhausted
            indices = np.repeat(np.arange(self.N)[None,:], self.M ,axis=0)
            indices = self.batch_rng.permuted(indices, axis=1)

            if self.batch_var == 'concat':
                self.indices = np.concatenate((self.indices, indices), axis=1)
            else:
                self.indices = indices
            #self.batch_size

        self.batch_idx = self.indices[:, :self.batch_size] # get batch indices
        self.indices = self.indices[:, self.batch_size:] # remove batch indices from indices

        self.consensus_idx = (self.get_reshaped_run_idx(), self.batch_idx, Ellipsis)
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

        return scheduler([multiply(name='alpha', factor=1.05)])
    
    def set_correction(self, correction):
        """
        Set the correction method for the object.
        
        Parameters
        ----------
            correction: (str or Callable): The type of correction to be set. Can be one of the following:
                - 'no_correction': No correction, correction is the identity.
                - 'heavi_side': Calculate the Heaviside correction.
                - 'heavi_side_reg': Calculate the regularized Heaviside correction, with parameter eps.
        
        Returns:
            None
        """
        if isinstance(correction, str):
            self.correction_callable = get_correction(correction, eps=self.correction_eps)
        elif callable(correction):
            self.correction_callable = correction
        else:
            raise ValueError('Invalid correction model! Choose from "no_correction", "heavi_side", or "heavi_side_reg" or a callable.')
        
    def correction(self, x: ArrayLike) -> ArrayLike:
        """
        Calculate the correction for the given input.

        Parameters
        ----------
            x (ndarray): The input array.

        Returns
        -------
            ndarray
                The correction value.
        """
        return self.correction_callable(self, x)

    def set_noise(self, noise) -> None:
        """
        Set the noise model for the object.

        Parameters
        ----------
            noise (str or Callable): The type of noise model to be set. Can be one of the following:
                - 'isotropic' or None: Set the isotropic noise model.
                - 'anisotropic': Set the anisotropic noise model.
                - 'sampling': Set the sampling noise model.
                - 'covariance': Set the covariance noise model.
                - else: use the given input as a callable.

        Returns
        -------
            None
        """
        # set noise model
        if isinstance(noise, str):
            self.noise_callable = get_noise(noise, self)
        elif callable(noise):
            self.noise_callable = noise
        else:
            raise ValueError('Invalid noise model: ' +str(noise) + '! Choose from "isotropic", "anisotropic", "sampling", "covariance", or a callable.')

    def noise(self, ) -> ArrayLike:
        """
        Calculate the noise vector. Here, we use the callable ``noise_callable``, which takes the dynamic as an input via ``self``.


        Parameters:
            None

        Returns:
            ndarray: The noise vector.
        """
        return self.noise_callable(self)
    
    def update_covariance(self,) -> None:
        r"""Update the covariance matrix :math:`\text{Cov}(x)` of the noise model
    
        Parameters

    
        Returns
        -------
        None.
    
        """                       
        weights = - self.alpha * self.energy
        coeffs = np.exp(weights - logsumexp_scp(weights, axis=(-1,), keepdims=True))
      
        D = self.drift[...,None] * self.drift[...,None,:]
        D = np.sum(D * coeffs[..., None, None], axis = -3)
        self.Cov_sqrt = compute_mat_sqrt(D)
                    
    def pre_step(self,):
        # save old positions
        self.x_old = self.copy(self.x)
        
        # set new batch indices
        self.set_batch_idx()
        
    def inner_step(self,):
        pass
        
    def post_step(self):
        self.post_process(self)
        if hasattr(self, 'x_old'):
            self.update_diff = self.norm(self.x - self.x_old, axis=(-2,-1))/self.N
        
        self.update_best_cur_particle()
        self.update_best_particle()
        self.track()
            
        self.t += self.dt
        self.it+=1
        
    def reset(self,):
        self.it = 0
        self.init_history()
        self.t = 0.

    def eval_f(self, x):
        self.num_f_eval[self.active_runs_idx] += x.shape[1] # update number of function evaluations
        return self.f(x)
    
    def print_cur_state(self,):
        if self.verbosity > 0:
            print('Time: ' + "{:.3f}".format(self.t) + ', best current energy: ' + str(self.f_min))
            print('Number of function evaluations: ' + str(self.num_f_eval))

        if self.verbosity > 1:
            print('Current alpha: ' + str(self.alpha))
            
    def compute_consensus(self,) -> None:
        r"""Updates the weighted mean of the particles.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # evaluation of objective function on batch
        
        energy = self.eval_f(self.x[self.consensus_idx]) # update energy
        self.consensus, energy = self._compute_consensus(
            energy, self.x[self.consensus_idx], 
            self.alpha[self.active_runs_idx, :]
        )
        self.energy[self.consensus_idx] = energy
        
        
    print_vars = ['dt', 'lamda', 'alpha', 'copy', 'norm', 'sampler'] + ParticleDynamic.print_vars
    