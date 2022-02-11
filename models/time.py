import torch
import numpy as np


class TimeEncoder(torch.nn.Module):
    def __init__(self, dimension, use_fourier_features=True):
        super(TimeEncoder, self).__init__()
        self.use_fourier_features = use_fourier_features
        self.dimension = dimension

        if self.use_fourier_features:
            # convert timestamps to fourier features
            self.w = torch.nn.Linear(1, dimension)

            self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1))
            self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())
        else:
            # just use timestamps. Any edge wewight that can be used in place of an actual timestamp
            assert self.dimension == 1




    def forward(self, t):
        # t has shape [batch_size] seen as a batch of edges
        # Add dimension at the end --> [batch_size, 1]
        t = t.unsqueeze(dim=1)

        if self.use_fourier_features:
            # apply linear layer and cosine. output has shape [batch_size, dimension]
            t = torch.cos(self.w(t))

        return t