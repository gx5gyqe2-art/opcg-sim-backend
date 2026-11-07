#!/bin/bash
# 使い方: train.sh <名前> [形]   例: train.sh pd   /  train.sh pdh head
cd /home/user/opcg-sim-backend
NAME=$1; SH=${2:-}; WT=${3:-balanced}
ins=""; for w in 70f 71f 72f 73f; do ins="$ins $HOME/n$w/n_records/n${w}_w*_p[0-3]"; done
extra=""; [ -n "$SH" ] && [ "$SH" != all ] && extra="--pair-diff-shapes $SH"
OPCG_LOG_SILENT=1 PYTHONPATH=$HOME/torch_env:/home/user/opcg-sim-backend python -m opcg_sim.learned.train.n_rel_train train --in $ins \
  --ablate rel --aux-weight 0.1 --lr 2e-4 --epochs 2 --pi-teacher q_improved --pi-steps-mult 1.0 --fast --save-each-epoch \
  --warm-start opcg_sim/data/learned/nrel_r28x4.npz --pair-diff $HOME/pdl/pairs.npz --pair-diff-weight $WT $extra --out $HOME/pdl/nets/nrel_$NAME.npz > $HOME/pdl/train_$NAME.log 2>&1
echo TRAIN_EXIT $? >> $HOME/pdl/train_$NAME.log
