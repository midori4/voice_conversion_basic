"""Calculate average and variance of log f0 each source and target speaker.

usage: f0_statics.py [options] <f0_dir>

options:
    -h, --help    Show this help message and exit.
"""
from docopt import docopt

import numpy as np
from os.path import splitext, join
import os
import sys

class DataSource():
    def __init__(self, dirname):
        self.dirname = dirname
    
    def collect_files(self):
        files = list(filter(lambda x: splitext(x)[-1] == ".f0",
                            os.listdir(self.dirname)))
        files = sorted(list(map(lambda d: join(self.dirname, d), files)))
        return files
    
    def get_feature(self, path):
        with open(path, mode="rb") as f:
            data = np.fromfile(f, dtype=np.float32, sep="").reshape(-1)
        return data
    
    def collect_features(self):
        files = self.collect_files()
        datalist = []
        for path in files:
           datalist += self.get_feature(path).tolist()
        return np.array(datalist, dtype=np.float32)

if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    f0_dir = args["<f0_dir>"]
    
    # get f0 statics of source and target speaker
    src_f0 = DataSource(join(f0_dir, "X", "f0")).collect_features()
    tgt_f0 = DataSource(join(f0_dir, "Y", "f0")).collect_features()
    print(src_f0, tgt_f0)
    src_lf0 = np.log(src_f0[np.nonzero(src_f0)])
    tgt_lf0 = np.log(tgt_f0[np.nonzero(tgt_f0)])
    src_ave, src_var = np.average(src_lf0), np.var(src_lf0)
    tgt_ave, tgt_var = np.average(tgt_lf0), np.var(tgt_lf0)
    
    f0_statics = {"src_ave": src_ave, "tgt_ave": tgt_ave, "src_var": src_var, "tgt_var": tgt_var}
    np.save(join(f0_dir, "f0_statics"), f0_statics)
#    sys.exit(0)
