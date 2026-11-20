#!/bin/bash
# V2: paper-method TFLOPS, unlocked clocks (no sudo). usage: v2.sh <gpu> <tag> <reps> <sleep> trees...
gpu=$1; tag=$2; reps=$3; sl=$4; shift 4; W=$HOME/work3187; L=$W/logs/v2; mkdir -p $L; cd /tmp
for r in 1 2 3; do for tree in "$@"; do
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$W/stub:$W/$tree $HOME/venvs/fa3187/bin/python -W ignore $HOME/handoff-3187/scripts/paper_timer2.py --reps $reps --sleep $sl > $L/${tag}_g${gpu}_${tree}_r$r.log 2>$L/${tag}_g${gpu}_${tree}_r$r.err
done; done; echo done $tag
