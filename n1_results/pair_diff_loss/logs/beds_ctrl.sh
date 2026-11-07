#!/bin/bash
cd /home/user/opcg-sim-backend
export OPCG_LOG_SILENT=1 PYTHONPATH=$HOME/torch_env:/home/user/opcg-sim-backend
B=~/pdl/beds; NETS="opcg_sim/data/learned/nrel_r28x4.npz $HOME/pdl/ctrl/nrel_r29f4.ep0.npz $HOME/pdl/ctrl/nrel_r29f4.ep1.npz"
python tests/scripts/value_teach.py tvval --out n1_results/value_teach --tag pdctrl --nets $NETS > $B/tvval_ctrl.log 2>&1; mv n1_results/value_teach/tvval_pdctrl.json $B/tvval_ctrl.json; echo "TVVAL $(date -u +%H:%M)" >> $B/ctrl.log
python tests/scripts/don_jitter_rows.py rows --nets $NETS > $B/jitter_rows_ctrl.log 2>&1; mv n1_results/don_jitter/jitter_rows.json $B/jitter_rows_ctrl.json; git checkout -q n1_results/don_jitter/jitter_rows.json; echo "ROWS $(date -u +%H:%M)" >> $B/ctrl.log
python -m opcg_sim.learned.train.n_rel_band --in $HOME/n74c/n_records/n74c_w*_p[0-3] --nrel $NETS --holdout-mod 7 --out $B/band_n74c_ctrl.json > $B/band_n74c_ctrl.log 2>&1; echo "BAND $(date -u +%H:%M)" >> $B/ctrl.log
echo CTRL_DONE >> $B/ctrl.log
