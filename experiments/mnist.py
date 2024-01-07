import cbx as cbx
from cbx.dynamics.cbo import CBO
import numpy as np

import torch
import torch.nn as nn
import torchvision
from cbx.noise import anisotropic_noise
from torch.func import functional_call, stack_module_state, vmap
#%% load MNIST
data_path = "../../datasets/"
train_data = torchvision.datasets.MNIST(data_path, train=True, 
                                        transform=torchvision.transforms.ToTensor(), 
                                        download=False)
test_data = torchvision.datasets.MNIST(data_path, train=False, 
                                        transform=torchvision.transforms.ToTensor(), 
                                        download=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,
                                           shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64,
                                          shuffle=False, num_workers=0)
#%% initializ network
class Perceptron(nn.Module):
    def __init__(self, mean = 0.0, std = 1.0, 
                 act_fun=nn.ReLU,
                 sizes = None):
        super(Perceptron, self).__init__()
        #
        self.mean = mean
        self.std = std
        self.act_fun = act_fun()
        self.sizes = sizes if sizes else [784, 10]
        self.linear = nn.Linear(self.sizes[0], self.sizes[1])
        self.bn = nn.BatchNorm1d(10, track_running_stats=False)
        self.sm = nn.Softmax(dim=1)

    def __call__(self, x):
        x = x.view([-1, 784])
        x = (x - self.mean)/self.std
        x = self.linear(x)
        x = self.act_fun(x)

        # apply softmax
        x = self.bn(x)
        x = self.sm(x)
        return x

#%% define loss function
device = 'cpu'
loss_fct = nn.CrossEntropyLoss()

model = Perceptron()

def wrapper(params, buffers, data, target):
    with torch.no_grad():
        return loss_fct(functional_call(model, (params, buffers), data), target)

class objective:
    def __init__(self, train_loader, N, wshape, bshape):
        self.train_loader = train_loader
        self.data_iter = iter(train_loader)
        self.N = N
        self.wshape = wshape
        self.bshape = bshape
        self.wcut = np.prod(wshape)
        self.epochs = 0
        
    def __call__(self, w):
        (x,y) = next(self.data_iter, (None, None))
        if x is None:
            self.data_iter = iter(self.train_loader)
            (x,y) = next(self.data_iter)
            self.epochs += 1
        
        x, y = x.to(device), y.to(device)
        params = {'linear.weight': w[...,:self.wcut].view(self.N, *self.wshape),
                  'linear.bias': w[...,self.wcut:].view(self.N,*self.bshape)} 
            
        out = vmap(wrapper, (0, 0, None, None))(params, {}, x, y)
        return out
  
N = 50
models = [Perceptron() for _ in range(N)]
params, buffers = stack_module_state(models)
f = objective(train_loader, N, (10,784), (10,))

w = torch.concatenate((params['linear.weight'].view(N,-1).detach(),
                       params['linear.bias'].detach()),
                      dim=-1
                      )

def eval_model(x, w, n = 0, m = 0, wshape=(10, 784), bshape=(10,)):
    w = torch.tensor(w, dtype=torch.float32)
    wcut = np.prod(wshape)
    params = {'linear.weight': w[m, n,:wcut].view(wshape),
              'linear.bias': w[m, n, wcut:].view(bshape)} 
    return functional_call(model, (params, buffers), x)

def eval_acc(w, n = 0, m = 0):
    res = 0
    num_img = 0
    for (x,y) in iter(test_loader):
        res += torch.sum(eval_model(x, w, n=n, m=m).argmax(axis=1)==y)
        num_img += x.shape[0]
    return res/num_img
#%%
kwargs = {'alpha':50.0,
        'dt': 0.1,
        'sigma': 0.1,
        'lamda': 1.0,
        'term_args':{'max_time': 20},
        'verbosity':2,
        'batch_args':{'batch_size':N},
        #'batch_size': M,
        'check_f_dims':False}
def norm_torch(x, axis, **kwargs):
    return torch.linalg.norm(x, dim=axis, **kwargs)  
noise = anisotropic_noise(norm = norm_torch, sampler = torch.normal)
resamplings = [cbx.utils.resampling.loss_update_resampling(M=1, wait_thresh=40)]

dyn = CBO(f, f_dim='3D', x=w, noise=noise, 
          resamplings=resamplings, 
          norm=norm_torch,
          copy=torch.clone,
          normal=torch.normal,
          **kwargs)
sched = cbx.scheduler.multiply(factor=1.03, name='alpha')
#%%
e = 0
while f.epochs < 5:
    dyn.step()
    sched.update(dyn)

    if dyn.it%10 == 0:
        print('Cur Best energy: ' + str(dyn.best_cur_energy))
        print('Best energy: ' + str(dyn.best_energy))
        print('Alpha: ' + str(dyn.alpha))
        print('Sigma: ' + str(dyn.sigma))
    if e != f.epochs:
        e = f.epochs
        print(30*'-')
        print('Epoch: ' +str(f.epochs))
        print('Accuracy: ' + str(eval_acc(dyn.best_particle[None,...])))
        print(30*'-')