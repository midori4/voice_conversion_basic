#!/bin/tcsh -f

# model directory ex) base_dir=data/model_mcep
set base_dir = $argv[1]

mkdir -p $base_dir/loss

perl -ne 'if(/"main\/loss.+\s+(.+),/){print "$1\n"}' $base_dir/log | head -n 200 | less > $base_dir/loss/train-loss.txt
perl -ne 'if(/"validation\/main\/loss.+\s+(.+),/){print "$1\n"}' $base_dir/log | head -n 200 | less > $base_dir/loss/val-loss.txt

# (validation loss) - (train loss)
paste $base_dir/loss/train-loss.txt $base_dir/loss/val-loss.txt | awk '{print $2 - $1}' | less > $base_dir/loss/diff.txt
