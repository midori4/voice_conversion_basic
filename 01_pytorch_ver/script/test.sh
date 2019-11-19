#!/bin/tcsh -f

set source_wav_dir = $argv[1]
set order = $argv[2]
set num_hidden = $argv[3]
set hunits = $argv[4]
set dropout = $argv[5]
set model_path = $argv[6]
set out_dir = $argv[7]

set train_subset = (a)
set test_subset = (j)

set train_data_dir = ../data/train/

foreach subset ($train_subset)
    foreach number (`seq -f %02g 1 50`)
        python ../tool/test.py $source_wav_dir/$subset/*$number.wav \
            $model_path $train_data_dir $out_dir/closed \
            --order=$order --num_hidden=$num_hidden --hunits=$hunits \
            --dropout=$dropout
    end
end

foreach subset ($test_subset)
    foreach number (`seq -f %02g 1 53`)
        python ../tool/test.py $source_wav_dir/$subset/*$number.wav \
            $model_path $train_data_dir $out_dir/open \
            --order=$order --num_hidden=$num_hidden --hunits=$hunits \
            --dropout=$dropout
    end
end

