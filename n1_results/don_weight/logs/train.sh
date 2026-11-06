#!/bin/bash
cd /home/user/opcg-sim-backend
ins=""; for w in 70f 71f 82j; do ins="$ins $HOME/n$w/n_records/n${w}_w*_p[0-3]"; done
OPCG_DUMP_CACHE=$HOME/packcache OPCG_LOG_SILENT=1 PYTHONPATH=$HOME/torch_env:/home/user/opcg-sim-backend python -m opcg_sim.learned.train.n_rel_train train --in $ins \
  --ablate rel --aux-weight 0.1 --lr 2e-4 --epochs 2 --pi-teacher q_improved --pi-steps-mult 1.0 --fast --save-each-epoch \
  --warm-start opcg_sim/data/learned/nrel_r28x4.npz --jitter-weight balanced --out $HOME/dw/nets/nrel_dw.npz > $HOME/dw/train_dw.log 2>&1
echo TRAIN_EXIT $? >> $HOME/dw/train_dw.log
