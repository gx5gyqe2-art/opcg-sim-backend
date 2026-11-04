# $1 = name, rest = waves (e.g. 66x 67x ...)
cd /home/user/opcg-sim-backend
name=$1; shift
ins=""; for w in "$@"; do ins="$ins $HOME/n$w/n_records/n${w}_w*_p[0-3]"; done
OPCG_DUMP_CACHE=$HOME/packlinks PYTHONPATH=$HOME/torch_env:/home/user/opcg-sim-backend OPCG_LOG_SILENT=1 python -m opcg_sim.learned.train.n_rel_train train --in $ins \
  --ablate rel --aux-weight 0.1 --lr 2e-4 --epochs 2 --pi-teacher q_improved --pi-steps-mult 1.0 --fast --save-each-epoch \
  --warm-start opcg_sim/data/learned/nrel_r28x4.npz --out $HOME/nets/nrel_$name.npz > $HOME/ds/logs/train_$name.log 2>&1
echo TRAIN_EXIT $? >> $HOME/ds/logs/train_$name.log
