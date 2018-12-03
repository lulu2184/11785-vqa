import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class NonLinearLayer(nn.Module):
    """
        Simple class for non-linear fully connect network

        input: an array of dimension
        output module: several linear layers and ReLU functions

        example:
            input: [5, 20, 10]
            module: [nn.Linear(5, 20), nn.ReLU, nn.Linear(20, 10), nn.ReLU]
    """
    def __init__(self, dims):
        super(NonLinearLayer, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
