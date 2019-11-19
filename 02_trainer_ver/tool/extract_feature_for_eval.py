"""Extract acoustic features for one-to-one voice conversion.

usage: extract_feature.py [options] <source_wav> <target_wav> <dst_dir>

options:
    --frame_period=<fp>    Frame period in msec [default: 5.0].
    --freq_warp=<fw>       Frequency warping parameter [default: 0.42].
    --order=<O>            Order of mel-cepstrum [default: 24].
    --traindata            Extraction for training data? True or False
    -h, --help             Show this help message and exit.
"""
from __future__ import division, print_function, absolute_import

from docopt import docopt
import numpy as np

from scipy.io import wavfile
import pyworld
import pysptk
from fastdtw import fastdtw as dtw

from os.path import join, exists, splitext, basename
import os
import sys


def get_feature(wav_path, frame_period, freq_warp, order):
    fs, data = wavfile.read(wav_path)
    data = data.astype(np.float64)
    # extract f0 contour
    f0, timeaxis = pyworld.dio(data, fs, frame_period=frame_period)
    f0 = pyworld.stonemask(data, f0, timeaxis, fs)
    # spectrogram
    sp = pyworld.cheaptrick(data, f0, timeaxis, fs)
    # mel cepstrum
    mcep = pysptk.sp2mc(sp, order=order, alpha=freq_warp)
    return f0.astype(np.float32), mcep.astype(np.float32)

if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    source_wav = args["<source_wav>"]
    target_wav = args["<target_wav>"]
    data_dir = args["<dst_dir>"]
    frame_period = float(args["--frame_period"])
    freq_warp = float(args["--freq_warp"])
    order = int(args["--order"])
    traindata = args["--traindata"]
    
    features = ["mcep", "f0"]
    for sp in ["X", "Y"]:
        for feature in features:
            d = join(data_dir, sp, feature)
            if not exists(d):
                os.makedirs(d)
    f01, mcep1 = get_feature(source_wav, frame_period, freq_warp, order)
    f02, mcep2 = get_feature(target_wav, frame_period, freq_warp, order)
    
    mcep_dir1 = join(data_dir, "X", "mcep")
    mcep_dir2 = join(data_dir, "Y", "mcep")
    mcep1.tofile(join(mcep_dir1, splitext(basename(source_wav))[0] + ".mcep"))
    mcep2.tofile(join(mcep_dir2, splitext(basename(target_wav))[0] + ".mcep"))
    
    f0_dir1 = join(data_dir, "X", "f0")
    f0_dir2 = join(data_dir, "Y", "f0")
    f01.tofile(join(f0_dir1, splitext(basename(source_wav))[0] + ".f0"))
    f02.tofile(join(f0_dir2, splitext(basename(target_wav))[0] + ".f0"))
    
    sys.exit(0)
