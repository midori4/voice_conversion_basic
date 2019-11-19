"""Training acostic feature model

usage: train.py [options] <train_dir> <validation_dir> <save_dir>

options:
    --order=<N>         Order of mel cepstrum [default: 39].
    --num_hidden=<N>    The number of hidden layers [default: 3].
    --hunits=<N>        The number of hidden units [default: 128].
    --dropout=<f>       The probability of dropout [default: 0.5].
    --batchsize=<N>     The size of batch [default: 1000].
    --nepoch=<N>        The total epoch [default: 1000].
    --gpu_id=<N>        GPU id [default: -1].
    -h, --help          Show this help message and exit.
"""
from docopt import docopt

import numpy as np
import chainer
from chainer import Variable
from chainer.datasets import tuple_dataset
from chainer import cuda
from chainer import optimizers
from chainer import training, iterators
from chainer.dataset import convert
from chainer.iterators import MultiprocessIterator
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L

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


class MyRegressor(chainer.Chain):
    def __init__(self, predictor):
        super(MyRegressor, self).__init__(predictor=predictor)
    
    def __call__(self, x, y):
        pred = self.predictor(x)
        loss = F.mean_squared_error(pred, y)
        chainer.report({'loss': loss}, self)
        
        return loss

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
    gpu_id = int(args["--gpu_id"])
    
    if not exists(save_dir):
        os.makedirs(save_dir)
    
    x_train = DataSource(join(train_dir, "X", "mcep"), order + 1).collect_features()
    y_train = DataSource(join(train_dir, "Y", "mcep"), order + 1).collect_features()
    x_val = DataSource(join(validation_dir, "X", "mcep"), order + 1).collect_features()
    y_val = DataSource(join(validation_dir, "Y", "mcep"), order + 1).collect_features()
    
    assert len(x_train) == len(y_train) and len(x_val) == len(y_val), "Mismatch between length of source and target data."

    # parameter save for normalization
    x_mean = np.mean(x_train, axis=0, keepdims=True)
    x_var = np.var(x_train, axis=0, keepdims=True)
    y_mean = np.mean(y_train, axis=0, keepdims=True)
    y_var = np.var(y_train, axis=0, keepdims=True)
    x_mean.tofile(join(train_dir, "x_mean.mcep"))
    x_var.tofile(join(train_dir, "x_var.mcep"))
    y_mean.tofile(join(train_dir, "y_mean.mcep"))
    y_var.tofile(join(train_dir, "y_var.mcep"))
    # print(x_train.shape, y_train.shape)
    # normalize
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
    out_dim = len(y_train[0])
    model = MLP(in_dim, out_dim, num_hidden, hunits, dropout)
    # model = L.Classifier(
    #     model,
    #     lossfun=F.mean_squared_error,
    # )
    # model.compute_accuracy = False
    model = MyRegressor(model)
    if gpu_id >= 0:
        model.to_gpu(gpu_id)
    
    train = tuple_dataset.TupleDataset(norm_x_train, norm_y_train)
    val = tuple_dataset.TupleDataset(norm_x_val, norm_y_val)
    
    train_iter = iterators.SerialIterator(train, batchsize)
    validation_iter = iterators.SerialIterator(val, batchsize, repeat=False, shuffle=False)
    
    # optimizer
    opt = optimizers.Adam()
    opt.setup(model)
    
    # trigger_log = (1, 'epoch')
    trigger_snapshot = (10, 'epoch')
    
    updater = training.StandardUpdater(train_iter, opt, device=gpu_id) 
    trainer = training.Trainer(updater, stop_trigger=(nepoch, 'epoch'), out=save_dir)
    
    trainer.extend(extensions.LogReport(
        # trigger=trigger_log,
    ))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
    
    ext = extensions.Evaluator(validation_iter, model, device=gpu_id)
    trainer.extend(
        ext,
        # trigger=trigger_log,
    )
    
    trainer.extend(extensions.dump_graph('main/loss'))
    
    ext = extensions.snapshot_object(model, filename='model_{.updater.epoch}.npz')
    trainer.extend(ext, trigger=trigger_snapshot)
     
    trainer.extend(extensions.ProgressBar())
    trainer.run()
 
