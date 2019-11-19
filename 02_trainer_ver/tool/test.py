"""Test trained model by performing voice conversion.

usage: test.py [options] <inwav> <model_path> <train_data_dir> <out_dir>

options:
    --frame_period=<fp>    Frame period in msec [default: 5.0].
    --freq_warp=<fw>       Frequency warping parameter [default: 0.42].
    --order=<N>            Order of mel cepstrum [default: 39].
    --num_hidden=<N>       The number of hidden layers [default: 3].
    --hunits=<N>           The number of hidden units [default: 128].
    --dropout=<f>          The probability of dropout [default: 0.5].
    --gpu_id=<N>           GPU id [default: -1]. 
    -h, --help             Show this help message and exit.
"""
from docopt import docopt

import numpy as np

import sys
import os
from os.path import join, basename, splitext, exists

import chainer
import chainer.cuda as cuda
from model import MLP

from scipy.io import wavfile
import pyworld
import pysptk

from pyrawnorm import energy_norm

def get_feature(wav_path, frame_period, freq_warp, order):
    fs, data = wavfile.read(wav_path)
    data = data.astype(np.float64)
    # extract f0 contour
    f0, timeaxis = pyworld.dio(data, fs, frame_period=frame_period)
    f0 = pyworld.stonemask(data, f0, timeaxis, fs)
    # aperiodicity
    ap = pyworld.d4c(data, f0, timeaxis, fs)
    # spectrogram
    sp = pyworld.cheaptrick(data, f0, timeaxis, fs)
    # mel cepstrum
    mcep = pysptk.sp2mc(sp, order=order, alpha=freq_warp)
    return fs, f0.astype(np.float32), mcep.astype(np.float32), ap

if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    inwav = args["<inwav>"]
    model_path = args["<model_path>"]
    train_data_dir = args["<train_data_dir>"]
    out_dir = args["<out_dir>"]
    frame_period = float(args["--frame_period"])
    freq_warp = float(args["--freq_warp"])
    order = int(args["--order"])
    num_hidden = int(args["--num_hidden"])
    hunits = int(args["--hunits"])
    dropout = float(args["--dropout"])
    gpu_id = int(args["--gpu_id"])
    
    for d in [join(out_dir, "mcep"), join(out_dir, "f0"), join(out_dir, "wav")]:
        if not exists(d):
            os.makedirs(d)
    
    x_mcep_mean = np.fromfile(join(train_data_dir, "x_mean.mcep"), dtype=np.float32, sep="").reshape(1, 40)
    x_mcep_var = np.fromfile(join(train_data_dir, "x_var.mcep"), dtype=np.float32, sep="").reshape(1, 40)
    y_mcep_mean = np.fromfile(join(train_data_dir, "y_mean.mcep"), dtype=np.float32, sep="").reshape(1, 40)
    y_mcep_var = np.fromfile(join(train_data_dir, "y_var.mcep"), dtype=np.float32, sep="").reshape(1, 40)
    
    # read model
    in_dim = out_dim  = order + 1
    model = MLP(in_dim, out_dim, num_hidden, hunits, dropout)
    chainer.serializers.load_npz(model_path, model, strict=False)
    if gpu_id >= 0:
        model.to_gpu(gpu_id)
    
    # get acostic features (f0, 0th and the other coefficient of mcep)
    fs, f0, mcep, ap = get_feature(inwav, frame_period, freq_warp, order)
    # normalize
    norm_mcep = (mcep - x_mcep_mean) / np.sqrt(x_mcep_var)
    # print(mcep)
    # print(norm_mcep)
    if gpu_id >= 0:
        norm_mcep = cuda.to_gpu(norm_mcep, device=gpu_id)
    norm_mcep = chainer.Variable(norm_mcep)
    
    with chainer.using_config('train', False):
        # take data from chainer.Variable
        estimated_mcep = model(norm_mcep).data
    # print(estimated_mcep)
    if gpu_id >= 0:
        estimated_mcep = cuda.to_cpu(estimated_mcep)
     
    # denormalize
    estimated_mcep = estimated_mcep * np.sqrt(y_mcep_var) + y_mcep_mean
    # print(estimated_mcep)
    estimated_mcep.tofile(join(out_dir, "mcep", splitext(basename(inwav))[0] + ".mcep"))
    
    # mcep -> spectral envelope
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    sp = pysptk.mc2sp(estimated_mcep.astype(np.float64), alpha=freq_warp, fftlen=fftlen)
    
    # get f0 statics (average and variance of log f0) of source and target speaker
    f0_statics_path = join(train_data_dir, "f0_statics.npy")
    f0_statics = np.load(f0_statics_path).item()
    src_ave, tgt_ave, src_var, tgt_var = f0_statics["src_ave"], f0_statics["tgt_ave"], f0_statics["src_var"], f0_statics["tgt_var"]
    # f0 conversion (linear transfomation in log domain)
    conv_f0 = []
    for x in f0:
        if x == 0.:
            conv_f0.append(0.)
        else:
            conv_f0.append(np.exp(tgt_ave + (np.sqrt(tgt_var) / np.sqrt(src_var)) * (np.log(x) - src_ave)))
    conv_f0 = np.array(conv_f0, dtype=np.float32)
    conv_f0.tofile(join(out_dir, "f0", splitext(basename(inwav))[0] + ".f0"))
 
    # make wave form and save as wav file
    outwav_path = join(out_dir, "wav", basename(inwav))
    waveform = pyworld.synthesize(conv_f0.astype(np.float64), sp, ap, fs, frame_period)
    waveform = energy_norm(waveform, const_energy=1500)
    wavfile.write(outwav_path, fs, waveform.astype(np.int16))
