#!/bin/bash
prefix="regression_check_`date '+%Y-%m-%d-%H:%M'`_`git rev-parse --short HEAD`"
./fb_sweep/benchmark_lm.py -g 8 -t 1 -n 1 --ddp fully_sharded --dl 12 --embed-dim 1024 \
  --bs 8 --li 50 --epg 0 --mu 7200 --ebs 128 \
  --constraint volta32gb --partition learnaccel \
  --snapshot-code --wu 2000 \
  --resume-failed --nw 0 -p $prefix \
  --checkpoints-dir /checkpoint/$USER/reg_checks \
  "$@"
