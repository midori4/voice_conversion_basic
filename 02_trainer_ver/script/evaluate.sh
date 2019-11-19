#!/bin/tcsh -f

set tgt_sp = fkn

set train_subset = (a)
set test_subset = (j)
set data_dir = ../data
set converted_dir = $argv[1]

set PYTHONPATH = ../tool
set out_dir = $converted_dir/closed
python ../tool/evaluate.py \
    -i1 $data_dir/test/closed/Y/mcep -i2 $converted_dir/closed/mcep -o $out_dir

set out_dir = $converted_dir/open
python ../tool/evaluate.py \
    -i1 $data_dir/test/open/Y/mcep -i2 $converted_dir/open/mcep -o $out_dir
