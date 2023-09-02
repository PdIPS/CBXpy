import cbx as cbx
from cbx.dynamic.cbo import CBO
import numpy as np

import torch
import torch.nn as nn
import torchvision

#%% load MNIST
data_path = "../../datasets/"
train_data = torchvision.datasets.MNIST(data_path, train=True, 
                                        transform=torchvision.transforms.ToTensor(), 
                                        download=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16,
                                           shuffle=True, num_workers=2)

#%% initializ network
class Perceptron(nn.Module):
    def __init__(self, mean = 0.0, std = 1.0, 
                 act_fun=nn.ReLU(),
                 sizes=[784, 10]):
        super(Perceptron, self).__init__()
        #
        self.mean = mean
        self.std = std
        self.act_fun = act_fun
        self.sizes = sizes
        self.linear = nn.Linear(sizes[0], sizes[1])

    def forward(self, x):
        x = (x - self.mean)/self.std
        x = self.linear(x)
        x = self.act_fun(x)

        # apply softmax
        x = nn.Softmax()(x)
        return x
#%% Initilaize copies of the model
M = 10
models = [Perceptron() for i in range(M)]
#%% define loss function
device = 'cpu'
loss_fct = nn.CrossEntropyLoss()
def f(w):
    #%% set weights
    for i, model in enumerate(models):
        model.linear.weight.data = w[0, i, :7840]
        model.linear.bias.data   = w[0, i, 7840:]
    
    loss = np.zeros((1, M))  
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        for i in range(M):
            loss[0, i] += loss_fct(models[i](x), y)
            
    return loss
            
#%%
kwargs = {'alpha':0.01,
        'dt': 0.01,
        'sigma': 0.5,
        'lamda': 1.0,
        'check_list': ['max_time'],
        'T': 20,
        'batch_size': M}
N = 100
conf = cbx.utils.config(**kwargs)
x = np.zeros((1, N, 10*784 + 10))
for i in range(N):
    model = Perceptron()
    x[0, i, :7840] = model.linear.weight.data.flatten()
    x[0, i, 7840:] = model.linear.bias.data
    
noise = cbx.noise.comp_noise(dt = conf.dt)

dyn = CBO(x, f, noise, f_dim='3D', **kwargs)   

#%%
t = 0
it = 0
while not dyn.terminate():
    dyn.step()
    #sched.update()
    
    if it%10 == 0:
        print(dyn.f_min)
        print('Alpha: ' + str(dyn.alpha))
        print('Sigma: ' + str(dyn.sigma))
        
    it+=1    