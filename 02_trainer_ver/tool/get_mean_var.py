"""Get mean and variance of each dimension of mel-cepstrum

usage: get_mean_var.py [options] <source_mcep_dir> <target_mcep_dir> <save_dir>

options:
    --order=<N>         Order of mel cepstrum [default: 39].
    -h, --help          Show this help message and exit.
"""
from docopt import docopt

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils

import sys
import os
import math
from os.path import splitext, join, exists
from tqdm import tqdm

from model import MLP

import time

class DataSource():
    def __init__(self, dirname, dim):
        self.dirname = dirname
        self.dim = dim
    
    def collect_files(self):
        files = list(filter(lambda x: splitext(x)[-1] == ".mcep",
                            os.listdir(self.dirname)))
        files = sorted(list(map(lambda d: join(self.dirname, d), files)))
        return files
    
    def get_feature(self, path):
        with open(path, mode="rb") as f:
            if self.dim == 1:
                data = np.fromfile(f, dtype=np.float32, sep="").reshape(-1)
            elif self.dim > 1:
                data = np.fromfile(f, dtype=np.float32, sep="").reshape(-1, self.dim)
            else:
                stderr.write("ERROR! Dimension must be greater than 0.")
                sys.exit(1)
        return data
    
    def collect_features(self):
        files = self.collect_files()
        datalist = []
        for path in files:
           datalist += self.get_feature(path).tolist()
        return np.array(datalist, dtype=np.float32)

# Training DNN...
def train(model, optimizer, criterion, batchsize, x_train, ntrain):
    model.train()
    perm = np.random.permutation(ntrain)
    sum_loss = 0
    
    # mini batch
    for i in xrange(0, ntrain, batchsize):
        xbatch = torch.from_numpy(x_train[perm[i:i+batchsize]])
        ybatch = torch.from_numpy(y_train[perm[i:i+batchsize]])
        if use_cuda:
            xbatch = xbatch.cuda()
            ybatch = ybatch.cuda()
        xbatch, ybatch = Variable(xbatch), Variable(ybatch)
        
        # Reser optimizer state
        optimizer.zero_grad()
        
        y_pred = model.predict(xbatch)
        loss = criterion(y_pred, ybatch)
        
        # backpropagation
        loss.backward()
        optimizer.step()
        
        sum_loss += loss * batchsize
    return sum_loss / ntrain

# Evaluate DNN
def val(model, criterion, batchsize, x_val, nval):
    model.eval()
    perm = np.random.permutation(nval)
    sum_loss = 0
    
    # mini batch
    for i in xrange(0, nval, batchsize):
        xbatch = torch.from_numpy(x_val[perm[i:i+batchsize]])
        ybatch = torch.from_numpy(y_val[perm[i:i+batchsize]])
        if use_cuda:
            xbatch = xbatch.cuda()
            ybatch = ybatch.cuda()
        xbatch, ybatch = Variable(xbatch), Variable(ybatch)
        
        y_pred = model.predict(xbatch)
        loss = criterion(y_pred, ybatch)
        sum_loss += loss * batchsize
    return sum_loss / nval


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    train_dir = args["<train_dir>"]
    validation_dir = args["<validation_dir>"]
    save_dir = args["<save_dir>"]
    order = int(args["--order"])
    num_hidden = int(args["--num_hidden"])
    hunits = int(args["--hunits"])
    dropout = float(args["--dropout"])
    batchsize = int(args["--batchsize"])
    nepoch = int(args["--nepoch"])
    
    if not exists(save_dir):
        os.makedirs(save_dir)
    # Is cuda available?
    use_cuda = torch.cuda.is_available()
    
    ts = time.time()
    x_train = DataSource(join(train_dir, "X", "mcep"), order + 1).collect_features()
    y_train = DataSource(join(train_dir, "Y", "mcep"), order + 1).collect_features()
    x_val = DataSource(join(validation_dir, "X", "mcep"), order + 1).collect_features()
    y_val = DataSource(join(validation_dir, "Y", "mcep"), order + 1).collect_features()
    te = time.time()
    
    elapsed_time = te - ts
    print(elapsed_time)
    
    assert len(x_train) == len(y_train) and len(x_val) == len(y_val), "Mismatch between length of source and target data."
    # the number of data (frame)
    ntrain = len(x_train)
    nval = len(x_val)
    print(ntrain, nval)
    # model definition
    in_dim = len(x_train[0])
    print(in_dim)
    out_dim = len(y_train[0])
    model = MLP(in_dim, out_dim, num_hidden, hunits, dropout)
    if use_cuda:
        model = model.cuda()
    # loss function
    criterion = nn.MSELoss()
    # optimizer
    optimizer = optim.Adam(model.parameters())
    
    train_loss_per_epoch = ""
    test_loss_per_epoch = ""
    for epoch in tqdm(xrange(1, nepoch+1)):
        # training
        epoch_loss = train(model, optimizer, criterion, batchsize, x_train, ntrain)
        train_loss_per_epoch += str(epoch) + "epoch" + "\t" + str(epoch_loss.item()) + "\n" 
        
        # evaluate
        epoch_loss = val(model, criterion, batchsize, x_val, nval)
        test_loss_per_epoch += str(epoch) + "epoch" + "\t" + str(epoch_loss.item()) + "\n"
    
    model_path = join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    
    train_loss_filename = join(save_dir, "train_loss.txt")
    test_loss_filename = join(save_dir, "test_loss.txt")
    with open(train_loss_filename, mode="w") as f:
        f.write(train_loss_per_epoch)
    with open(test_loss_filename, mode="w") as f:
        f.write(test_loss_per_epoch)
