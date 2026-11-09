#!/bin/bash
# 訓練が終わってから順に 1 本ずつ（並べない）
cd /home/user/opcg-sim-backend
export OPCG_LOG_SILENT=1 OPCG_DUMP_CACHE=$HOME/packcache PYTHONPATH=$HOME/torch_env:/home/user/opcg-sim-backend
D=n1_results/don_weight; mkdir -p $D
BASE=opcg_sim/data/learned/nrel_r28x4.npz
CT="$HOME/dw/ctrl/nrel_djt.ep0.npz $HOME/dw/ctrl/nrel_djt.ep1.npz"
DW="$HOME/dw/nets/nrel_dw.ep0.npz $HOME/dw/nets/nrel_dw.ep1.npz"
ALL="$BASE $CT $DW"
python tests/scripts/value_teach.py tvval --out n1_results/value_teach --tag dw --nets $ALL > $D/tvval_dw.log 2>&1; rc=$?; mv n1_results/value_teach/tvval_dw.json $D/tvval_dw.json; echo "TVVAL_EXIT $rc $(date -u +%H:%M)" >> $HOME/dw/beds.log
python tests/scripts/don_jitter_rows.py rows --nets $ALL > $D/jitter_rows_dw.log 2>&1; rc=$?
mv n1_results/don_jitter/jitter_rows.json $D/jitter_rows_dw.json; git checkout -q n1_results/don_jitter/jitter_rows.json
echo "ROWS_EXIT $rc $(date -u +%H:%M)" >> $HOME/dw/beds.log
python -m opcg_sim.learned.train.n_rel_band --in $HOME/n74c/n_records/n74c_w*_p[0-3] --nrel $ALL --holdout-mod 7 --out $D/band_n74c.json > $D/band_n74c.log 2>&1; echo "BAND_EXIT $? $(date -u +%H:%M)" >> $HOME/dw/beds.log
python tests/scripts/value_teach.py rocks --out $D --tag dw --nets $DW > $D/rocks_dw.log 2>&1; echo "ROCKS_EXIT $? $(date -u +%H:%M)" >> $HOME/dw/beds.log
echo BEDS_DONE >> $HOME/dw/beds.log
