"""Training acostic feature model

usage: train.py [options] <train_dir> <validation_dir> <save_dir>

options:
    --order=<N>         Order of mel cepstrum [default: 39].
    --num_hidden=<N>    The number of hidden layers [default: 3].
    --hunits=<N>        The number of hidden units [default: 128].
    --dropout=<f>       The probability of dropout [default: 0.5].
    --batchsize=<N>     The size of batch [default: 1000].
    --nepoch=<N>        The total epoch [default: 1000]..
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
def train(model, optimizer, criterion, batchsize, norm_x_train, norm_y_train, ntrain):
    model.train()
    perm = np.random.permutation(ntrain)
    sum_loss = 0
    
    # mini batch
    for i in xrange(0, ntrain, batchsize):
        xbatch = torch.from_numpy(norm_x_train[perm[i:i+batchsize]])
        ybatch = torch.from_numpy(norm_y_train[perm[i:i+batchsize]])
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
        
# Evaluate DNN
def test(model, criterion, batchsize, norm_x, norm_y):
    total_num = len(norm_x)
    model.eval()
    perm = np.random.permutation(total_num)
    sum_loss = 0
    
    # mini batch
    for i in xrange(0, total_num, batchsize):
        xbatch = torch.from_numpy(norm_x[perm[i:i+batchsize]])
        ybatch = torch.from_numpy(norm_y[perm[i:i+batchsize]])
        if use_cuda:
            xbatch = xbatch.cuda()
            ybatch = ybatch.cuda()
        xbatch, ybatch = Variable(xbatch), Variable(ybatch)
        
        y_pred = model.predict(xbatch)
        loss = criterion(y_pred, ybatch)
        sum_loss += loss * len(xbatch)
    return sum_loss / total_num


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
    
    x_train = DataSource(join(train_dir, "X", "mcep"), order + 1).collect_features()
    y_train = DataSource(join(train_dir, "Y", "mcep"), order + 1).collect_features()
    x_val = DataSource(join(validation_dir, "X", "mcep"), order + 1).collect_features()
    y_val = DataSource(join(validation_dir, "Y", "mcep"), order + 1).collect_features()
    
    assert len(x_train) == len(y_train) and len(x_val) == len(y_val), "Mismatch between length of source and target data."
    x_mean = np.mean(x_train, axis=0, keepdims=True)
    x_var = np.var(x_train, axis=0, keepdims=True)
    y_mean = np.mean(y_train, axis=0, keepdims=True)
    y_var = np.var(y_train, axis=0, keepdims=True)
    x_mean.tofile(join(train_dir, "x_mean.mcep"))
    x_var.tofile(join(train_dir, "x_var.mcep"))
    y_mean.tofile(join(train_dir, "y_mean.mcep"))
    y_var.tofile(join(train_dir, "y_var.mcep"))
    
    norm_x_train = (x_train - x_mean) / np.sqrt(x_var)
    norm_x_val = (x_val - x_mean) / np.sqrt(x_var)
    norm_y_train = (y_train - y_mean) / np.sqrt(y_var)
    norm_y_val = (y_val - y_mean) / np.sqrt(y_var)
    
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
        epoch_loss = train(model, optimizer, criterion, batchsize, norm_x_train, norm_y_train, ntrain)
        
        # training loss
        epoch_loss = test(model, criterion, batchsize, norm_x_train, norm_y_train)
        train_loss_per_epoch += str(epoch) + "epoch" + "\t" + str(epoch_loss.item()) + "\n"
        
        # evaluate
        epoch_loss = test(model, criterion, batchsize, norm_x_val, norm_y_val)
        test_loss_per_epoch += str(epoch) + "epoch" + "\t" + str(epoch_loss.item()) + "\n"
    
    model_path = join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    
    train_loss_filename = join(save_dir, "train_loss.txt")
    test_loss_filename = join(save_dir, "test_loss.txt")
    with open(train_loss_filename, mode="w") as f:
        f.write(train_loss_per_epoch)
    with open(test_loss_filename, mode="w") as f:
        f.write(test_loss_per_epoch)
