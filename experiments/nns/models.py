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
        self.linear = nn.Linear(self.sizes[0], self.sizes[1])
        self.bn = nn.BatchNorm1d(10, track_running_stats=False)
        self.sm = nn.Softmax(dim=1)

    def __call__(self, x):
        x = x.view([x.shape[0], -1])
        x = (x - self.mean)/self.std
        x = self.linear(x)
        x = self.act_fun(x)

        # apply softmax
        x = self.bn(x)
        x = self.sm(x)
        return x
