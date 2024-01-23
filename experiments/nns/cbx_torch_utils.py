from collections import OrderedDict
import torch
from torch.func import functional_call, stack_module_state, vmap

def norm_torch(x, axis, **kwargs):
    return torch.linalg.norm(x, dim=axis, **kwargs)  

def compute_consensus_torch(energy, x, alpha):
    weights = - alpha * energy
    coeffs = torch.exp(weights - torch.logsumexp(weights, axis=(-1,), keepdims=True))[...,None]
    return (x * coeffs).sum(axis=-2, keepdims=True), energy.cpu().numpy()

def normal_torch(device):
    def _normal_torch(mean, std, size):
        return torch.normal(mean, std, size).to(device)
    return _normal_torch

def eval_model(x, model, w, pprop):
    params = {p: w[pprop[p][-2]:pprop[p][-1]].view(pprop[p][0]) for p in pprop}
    return functional_call(model, (params, {}), x)

def eval_models(x, model, w, pprop):
    return vmap(eval_model, (None, None, 0, None))(x, model, w, pprop)

def eval_loss(x, y, loss_fct, model, w, pprop):
    with torch.no_grad():
        return loss_fct(eval_model(x, model, w, pprop), y)
    
def eval_losses(x, y, loss_fct, model, w, pprop):
    return vmap(eval_loss, (None, None, None, None, 0, None))(x, y, loss_fct, model, w, pprop)

def eval_acc(model, w, pprop, loader):
    res = 0
    num_img = 0
    for (x,y) in iter(loader):
        x = x.to(w.device)
        y = y.to(w.device)
        res += torch.sum(eval_model(x, model, w, pprop).argmax(axis=1)==y)
        num_img += x.shape[0]
    return res/num_img

def flatten_parameters(models, pnames):
    params, buffers = stack_module_state(models)
    N = list(params.values())[-1].shape[0]
    return torch.concatenate([params[pname].view(N,-1).detach() for pname in pnames], dim=-1)

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