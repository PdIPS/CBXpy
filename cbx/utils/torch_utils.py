from collections import OrderedDict
from functools import wraps
from ..scheduler import bisection_solve, eff_sample_size_gap
import numpy as np

try: # try torch import
    import torch
    from torch import logsumexp
    from torch.func import functional_call, stack_module_state, vmap
except ImportError:
    _has_torch = False
else:
    _has_torch = True


#%%
def requires_torch(f):
    @wraps(f)
    def torch_decorated(*args, **kwargs):
        if not _has_torch:
            raise ImportError('The requested function: ' + f.__name__ + 'requires ' + 
                              'torch, but torch import failed! Either install torch manually ' +
                              'or install cbx with torch option: pip install cbx[torch]')
        return f(*args, **kwargs)

    return torch_decorated

@requires_torch
def norm_torch(x, axis, **kwargs):
    return torch.linalg.vector_norm(x, dim=axis, **kwargs)  

@requires_torch
def compute_consensus_torch(energy, x, alpha):
    weights = - alpha * energy
    coeff_expan = tuple([Ellipsis] + [None for i in range(x.ndim-2)])
    coeffs = torch.exp(weights - logsumexp(weights, dim=-1, keepdims=True))[coeff_expan]
    return (x * coeffs).sum(axis=1, keepdims=True), energy.cpu().numpy()

@requires_torch
def compute_polar_consensus_torch(energy, x, neg_log_eval, alpha = 1., kernel_factor = 1.):
    weights = -kernel_factor * neg_log_eval - alpha * energy[:,None,:]
    coeff_expan = tuple([Ellipsis] + [None for i in range(x.ndim-2)])
    coeffs = torch.exp(weights - torch.logsumexp(weights, dim=(-1,), keepdims=True))[coeff_expan]
    c = torch.sum(x[:,None,...] * coeffs, axis=2)
    return c, energy.detach().cpu().numpy()

@requires_torch
def exponential_torch(device):
    def _exponential_torch(size=None):
        x = torch.distributions.Exponential(1.0).sample(
                size
        )
        sign = torch.randint(0, 2, size) * 2 - 1
        x: torch.Tensor = x* sign
        return x.to(device)
    return _exponential_torch

@requires_torch
class post_process_default_torch:
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
        dyn.x = torch.nan_to_num(dyn.x, nan=self.max_thresh)
        dyn.x = torch.clamp(dyn.x, min=-self.max_thresh, max=self.max_thresh)

@requires_torch
def standard_normal_torch(device):
    def _normal_torch(size=None):
        return torch.randn(size=size).to(device)
    return _normal_torch

def set_post_process_torch(self, post_process):
    self.post_process = post_process if post_process is not None else post_process_default_torch()

def set_array_backend_funs_torch(self, copy, norm, sampler):
    self.copy = copy if copy is not None else torch.clone
    self.norm = norm if norm is not None else norm_torch
    self.sampler = sampler if sampler is not None else standard_normal_torch(self.device)
    
def init_particles(self, shape=None,):
    return torch.zeros(size=shape).uniform_(-1., 1.)

def init_consensus(self, compute_consensus):
    self.consensus = None #consensus point
    self._compute_consensus = compute_consensus if compute_consensus is not None else compute_consensus_torch

def to_torch_dynamic(dyn_cls):
    def add_device_init(self, *args, device='cpu', **kwargs):
        self.device = device
        dyn_cls.__init__(self, *args, **kwargs)
        
    return type(dyn_cls.__name__ + str('_torch'), 
         (dyn_cls,), 
         dict(
             __init__ = add_device_init,
             set_array_backend_funs=set_array_backend_funs_torch,
             init_particles=init_particles,
             init_consensus=init_consensus,
             set_post_process = set_post_process_torch,
             )
         )


@requires_torch
def eval_model(x, model, w, pprop):
    params = {p: w[pprop[p][-2]:pprop[p][-1]].view(pprop[p][0]) for p in pprop}
    return functional_call(model, (params, {}), x)

@requires_torch
def eval_models(x, model, w, pprop):
    return vmap(eval_model, (None, None, 0, None))(x, model, w, pprop)

@requires_torch
def eval_loss(x, y, loss_fct, model, w, pprop):
    with torch.no_grad():
        return loss_fct(eval_model(x, model, w, pprop), y)

@requires_torch
def eval_losses(x, y, loss_fct, model, w, pprop):
    return vmap(eval_loss, (None, None, None, None, 0, None))(x, y, loss_fct, model, w, pprop)

@requires_torch
def eval_acc(model, w, pprop, loader):
    res = 0
    num_img = 0
    for (x,y) in iter(loader):
        x = x.to(w.device)
        y = y.to(w.device)
        res += torch.sum(eval_model(x, model, w, pprop).argmax(axis=1)==y)
        num_img += x.shape[0]
    return res/num_img

@requires_torch
def flatten_parameters(models, pnames):
    params, buffers = stack_module_state(models)
    N = list(params.values())[-1].shape[0]
    return torch.concatenate([params[pname].view(N,-1).detach() for pname in pnames], dim=-1)

@requires_torch
def get_param_properties(models, pnames=None):
    params, buffers = stack_module_state(models)
    pnames = pnames if pnames is not None else params.keys()
    pprop = OrderedDict()
    for p in pnames:
        a = 0
        if len(pprop)>0:
            a = pprop[next(reversed(pprop))][-1]
        pprop[p] = (params[p][0,...].shape, a, a + params[p][0,...].numel())
    return pprop

@requires_torch
class effective_sample_size:
    def __init__(self, name = 'alpha', eta=.5, maximum=1e5, minimum=1e-5, solve_max_it = 15):
        self.name = name
        self.eta = eta
        self.J_eff = 1.0
        self.solve_max_it = solve_max_it
        self.maximum = maximum
        self.minimum = minimum
        
    def update(self, dyn):
        val = getattr(dyn, self.name)
        device = val.device
        val = bisection_solve(
            eff_sample_size_gap(dyn.energy, self.eta), 
            self.minimum * np.ones((dyn.M,)), self.maximum * np.ones((dyn.M,)), 
            max_it = self.solve_max_it, thresh=1e-2
        )
        setattr(dyn, self.name, torch.tensor(val[:, None], device=device))