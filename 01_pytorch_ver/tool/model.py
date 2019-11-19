import numpy as np
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_hidden, hunits, dropout):
        super(MLP, self).__init__()
        in_sizes = [in_dim] + [hunits] * (num_hidden - 1)
        out_sizes = [hunits] * num_hidden
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size) for (in_size, out_size)
             in zip(in_sizes, out_sizes)])
        self.last_linear = nn.Linear(hunits, out_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
    
    def predict(self, x):
        for layer in self.layers:
            x = self.dropout(self.relu(layer(x)))
        x = self.last_linear(x)
        return x
