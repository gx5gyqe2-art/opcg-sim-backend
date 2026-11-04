# $1 = name, rest = --in dirs
cd /home/user/opcg-sim-backend
name=$1; shift
PYTHONPATH=$HOME/torch_env:/home/user/opcg-sim-backend OPCG_LOG_SILENT=1 python -m opcg_sim.learned.train.n_rel_train train --in "$@" \
  --ablate rel --aux-weight 0.1 --lr 2e-4 --epochs 2 --pi-teacher q_improved --pi-steps-mult 1.0 --fast --save-each-epoch \
  --warm-start opcg_sim/data/learned/nrel_r28x4.npz --out $HOME/nrel_$name.npz > $HOME/dj/train_$name.log 2>&1
echo TRAIN_EXIT $? >> $HOME/dj/train_$name.log
