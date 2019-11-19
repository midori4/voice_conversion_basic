#!/bin/tcsh -f

set source_wav_dir = $argv[1]
set target_wav_dir = $argv[2]
set order = $argv[3]
set data_dir = $argv[4]

set train_subset = (a)
set validation_subset = (i)
set test_subset = (j)

#    Max num files to be collected.
#    a ~ i: 50 files per one subset.
#    j: 53 files.
set train_max_num = 50
set validation_max_num = 50
set test_max_num = 53

foreach subset ($train_subset)
    foreach number (`seq -f %02g 1 $train_max_num`)
        python ../tool/extract_feature.py \
            $source_wav_dir/$subset/*$number.wav $target_wav_dir/$subset/*$number.wav $data_dir/train --order=$order --traindata
    end
end

foreach subset ($validation_subset)
    foreach number (`seq -f %02g 1 $validation_max_num`)
        python ../tool/extract_feature.py \
            $source_wav_dir/$subset/*$number.wav $target_wav_dir/$subset/*$number.wav $data_dir/validation --order=$order
    end
end

# extract feature (mel-cepstrum) for evaluation, NOT ALIGNED
foreach subset ($train_subset)
    foreach number (`seq -f %02g 1 $train_max_num`)
        python ../tool/extract_feature_for_eval.py \
            $source_wav_dir/$subset/*$number.wav $target_wav_dir/$subset/*$number.wav $data_dir/test/closed --order=$order
    end
end

foreach subset ($test_subset)
    foreach number (`seq -f %02g 1 $test_max_num`)
        python ../tool/extract_feature_for_eval.py \
            $source_wav_dir/$subset/*$number.wav $target_wav_dir/$subset/*$number.wav $data_dir/test/open --order=$order
    end
end
