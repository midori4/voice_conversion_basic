#!/bin/tcsh -f

ln -fns /disk/fs1/bigtmp/hayasaka/corpus/ATR/B-set/wav/16k ../data/ATRDIR

set ATR_dir = ../data/ATRDIR

set src_sp = mht
set tgt_sp = fkn
set source_wav_dir = $ATR_dir/$src_sp/sd
set target_wav_dir = $ATR_dir/$tgt_sp/sd
set model_dir = ../out/trained_model
set conversion_dir = ../out/conversion_result
mkdir -p $model_dir
mkdir -p $conversion_dir

set order = 39
set data_dir = ../data
# ./extract_feature.sh $source_wav_dir $target_wav_dir $order $data_dir


# set num_hidden = 3
# set hunits = 256
set dropout = 0.5
set batchsize = 256
set nepoch = 50
set gpu_id = 1

set train_dir = $data_dir/train
set validation_dir = $data_dir/validation

set PYTHONPATH = ../tool
foreach num_hidden (5) # (1 2 3 4 5)
    foreach hunits (256)  # (32 64 128 256)
        set save_dir = ${model_dir}/layer${num_hidden}_unit${hunits}
        set converted_dir = $conversion_dir/layer${num_hidden}_unit${hunits}
        ./train.sh $train_dir $validation_dir $save_dir $order $num_hidden $hunits $dropout $batchsize $nepoch $gpu_id
        ./test.sh $source_wav_dir $order $num_hidden $hunits $dropout $save_dir/model_50.npz $converted_dir $gpu_id
        ./evaluate.sh $converted_dir
    end
end
