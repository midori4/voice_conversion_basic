import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain

class MLP(chainer.ChainList):
    def __init__(self, in_dim, out_dim, num_hidden, hunits, dropout):
        super().__init__()
        in_sizes = [in_dim] + [hunits] * num_hidden
        out_sizes = [hunits] * num_hidden + [out_dim]
        #self.layers = [L.Linear(in_size, out_size)
        #    for (in_size, out_size) in zip(in_sizes, out_sizes)]
        for (in_size, out_size) in zip(in_sizes, out_sizes):
            self.add_link(L.Linear(in_size, out_size))
        # self.last_linear = L.Linear(hunits, out_dim)
        self.num_hidden = num_hidden
        self.dropout = dropout
   
    def __call__(self, x):
        # for layer in self.layers:
        for idx, layer in enumerate(self.children()):
            if idx != self.num_hidden:
                x = F.dropout(F.relu(layer(x)), ratio=self.dropout)
            else:
                x = layer(x)
        return x
