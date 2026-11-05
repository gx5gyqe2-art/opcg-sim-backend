# $1 = name (例 thin), 残り = 追加の引数
cd /home/user/opcg-sim-backend
name=$1; shift
mkdir -p $HOME/ti
PYTHONPATH=$HOME/torch_env:/home/user/opcg-sim-backend OPCG_LOG_SILENT=1 OPCG_DUMP_CACHE=$HOME/packcache \
python -m opcg_sim.learned.train.n_rel_train train \
  --in $HOME/n70f/n_records/n70f_w*_p[0-3] $HOME/n71f/n_records/n71f_w*_p[0-3] $HOME/n72f/n_records/n72f_w*_p[0-3] $HOME/n73f/n_records/n73f_w*_p[0-3] \
  --ablate rel --aux-weight 0.1 --lr 2e-4 --epochs 2 --pi-teacher q_improved --pi-steps-mult 1.0 --fast --save-each-epoch \
  --warm-start opcg_sim/data/learned/nrel_r28x4.npz "$@" --out $HOME/ti/nrel_$name.npz > $HOME/ti/train_$name.log 2>&1
echo TRAIN_EXIT $? >> $HOME/ti/train_$name.log
