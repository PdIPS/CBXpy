import torch.nn as nn

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
        self.linears = nn.ModuleList([nn.Linear(self.sizes[i], self.sizes[i+1]) for i in range(len(self.sizes)-1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.sizes[i+1], track_running_stats=False) for i in range(len(self.sizes)-1)])
        self.sm = nn.Softmax(dim=1)

    def __call__(self, x):
        x = x.view([x.shape[0], -1])
        x = (x - self.mean)/self.std
        
        for linear, bn in zip(self.linears, self.bns):
            x = linear(x)
            x = self.act_fun(x)
            x = bn(x)

        # apply softmax
        x = self.sm(x)
        return x