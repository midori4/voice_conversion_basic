#!/bin/tcsh -f

set src_dir = $argv[1]
set tgt_dir = $argv[2]
set save_dir = $argv[3]
set order = $argv[4]
set num_hidden = $argv[5]
set hunits = $argv[6]
set dropout = $argv[7]
set batchsize = $argv[8]
set nepoch = $argv[9]
set gpu_id = $argv[10]

python -i ../tool/train.py $src_dir $tgt_dir $save_dir \
    --order=$order --num_hidden=$num_hidden --hunits=$hunits \
    --dropout=$dropout --batchsize=$batchsize --nepoch=$nepoch --gpu_id=$gpu_id

set f0_dir = ../data/train
python ../tool/f0_statics.py $f0_dir
