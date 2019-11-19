from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy
import argparse

import pysptk
import pyworld

from os.path import join, exists
import os

# from become_yukarin.config import create_from_json as create_config
from mcepdtw import MelCepstrumAligner, DTWAligner


parser = argparse.ArgumentParser()
parser.add_argument('--target_feature_directory', '-i1', type=str)
parser.add_argument('--converted_mcep_directory', '-i2', type=str)
parser.add_argument('--out_dir', '-o', type=str)
parser.add_argument('--order', type=int, default=39)
arguments = parser.parse_args()

print(arguments.target_feature_directory)
print(arguments.converted_mcep_directory)
if not exists(join(arguments.out_dir, "evaluation")):
    os.makedirs(join(arguments.out_dir, "evaluation"))

paths1 = list(sorted(glob(arguments.target_feature_directory + '/*')))
paths2 = list(sorted(glob(arguments.converted_mcep_directory + '/*')))

assert len(paths1) == len(paths2)
num_files = len(paths1)

def cdist(c1, c2, otype=0, frame=False):
    """Calculation of cepstral distance
    Parameters
    ----------
    c1 : array
        Minimum-phase cepstrum
    c2 : array
        Minimum-phase cepstrum
    otype : int
        Output data type
            (0) [db]
            (1) squared error
            (2) root squared error
        Default is 0.
    frame : bool
        If True, returns frame-wise distance, otherwise returns mean distance.
        Default is False.
    Returns
    -------
    distance
    """
    if not otype in [0, 1, 2]:
        raise ValueError("unsupported otype: %d, must be in 0:2" % otype)
    assert c1.shape[0] == c2.shape[0]
    T = c1.shape[0]

    s = ((c1[:, 1:] - c2[:, 1:])**2).sum(-1)
    if otype == 0:
        s = numpy.sqrt(2 * s) * 10 / numpy.log(10)
    elif otype == 2:
        s = numpy.sqrt(s)
    if frame:
        return s
    else:
        return s.mean()


dist = 0
target_GV = [0]*(arguments.order+1)
converted_GV = [0]*(arguments.order+1)

# cv = [0]*40
# cv1 = 0
# cv2 = 0

for num in range(0, num_files):
    tgt_mcep = numpy.fromfile(paths1[num], dtype=numpy.float32, sep="").reshape(-1, (arguments.order+1))
    converted_mcep = numpy.fromfile(paths2[num], dtype=numpy.float32, sep="").reshape(-1, (arguments.order+1))
    
    # GV: Global Variance
    for dim in range(0, (arguments.order+1)):
        target_GV[dim] += numpy.var(tgt_mcep, axis=0)[dim]
        converted_GV[dim] += numpy.var(converted_mcep, axis=0)[dim]
        
    # CV: Coefficient of variation
    # for dim in range(0,40):
    #     cv[dim] += numpy.abs(numpy.std(mc3[:,dim])/numpy.mean(mc3[:,dim]))
    
    # cv1 += numpy.std(mc2[:,8])/numpy.mean(mc2[:,8])
    # cv2 += numpy.std(mc2[:,39])/numpy.mean(mc2[:,39])
    
    aligner = MelCepstrumAligner(tgt_mcep, converted_mcep)
    tgt_mcep, converted_mcep = aligner.align(tgt_mcep, converted_mcep)
    print(tgt_mcep.shape)
    print(converted_mcep.shape)
    mcd = cdist(tgt_mcep, converted_mcep)
    dist += mcd
    print(mcd)
    # drawMcep(mc1,num)


with open(join(arguments.out_dir, "evaluation", "MCD.txt"), mode="w") as f:
    f.write(str(dist/num_files) + "\n")

target_GV = numpy.array(target_GV, dtype=numpy.float32) / num_files
converted_GV = numpy.array(converted_GV, dtype=numpy.float32) / num_files
target_GV.tofile(join(arguments.out_dir, "evaluation", "taget.gv"))
converted_GV.tofile(join(arguments.out_dir, "evaluation", "converted.gv"))

plt.figure(figsize=(16,6))
plt.plot(target_GV, "--", linewidth=2, label="Target: global variances")
plt.plot(converted_GV, linewidth=2, label="baseline: global variances")
plt.legend(prop={"size": 18})
plt.yscale("log")
plt.xlim(0, arguments.order)
plt.xlabel("Dimention", fontsize=16)
plt.ylabel("Global Varianvce", fontsize=16)
plt.savefig(join(arguments.out_dir, "evaluation", "GV.svg"))


# print("CV")
# for DIM in range(0,40):
#     print(cv[DIM]/50)
# print(cv1/50)
# print(cv2/50)
