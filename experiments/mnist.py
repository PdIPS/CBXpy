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
class Perceptron():
    def __init__(self, mean = 0.0, std = 1.0, 
                 act_fun=nn.ReLU,
                 sizes = None):
        super(Perceptron, self).__init__()
        #
        self.mean = mean
        self.std = std
        self.act_fun = act_fun()
        self.sizes = sizes if sizes else [784, 10]
        #self.linear = nn.Linear(self.sizes[0], self.sizes[1])

    def __call__(self, x, W, b):
        x = x.view([-1, 784])
        x = (x - self.mean)/self.std
        x = W@x + b
        x = self.act_fun(x)

        # apply softmax
        x = nn.Softmax()(x)
        return x

#%% define loss function
device = 'cpu'
loss_fct = nn.CrossEntropyLoss()
model = Perceptron()

def f(w):
    print(w.shape)
    # set weights
    N = w.shape[1]
    
    loss = np.zeros((1, N))  
    for (x, y) in iter(train_loader):
        x, y = x.to(device), y.to(device)
        for i in range(N):
            loss[0, i] += loss_fct(model(x, w[0,i,:7840], w[0,i,7840:]))
            
            
            
    return loss
            
#%%
kwargs = {'alpha':0.01,
        'dt': 0.01,
        'sigma': 0.5,
        'lamda': 1.0,
        'max_it':20,
        'max_time': 20,
        #'batch_size': M,
        'array_mode':'torch',
        'check_f_dims':False}
N = 10
x = torch.normal(0, 0.1, (1, N, 10*784 + 10))
    
noise = cbx.noise.comp_noise(dt = kwargs['dt'])

dyn = CBO(f, x=x, noise=noise,**kwargs)
 
if __name__=='__main__':
    f(dyn.x)
# #%%
# t = 0
# it = 0
# while not dyn.terminate():
#     dyn.step()
#     #sched.update()
    
#     if it%10 == 0:
#         print(dyn.f_min)
#         print('Alpha: ' + str(dyn.alpha))
#         print('Sigma: ' + str(dyn.sigma))
        
#     it+=1    