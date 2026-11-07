#!/bin/bash
# 訓練が終わってから順に 1 本ずつ（並べない）。使い方: beds_arm.sh <名前>
cd /home/user/opcg-sim-backend
export OPCG_LOG_SILENT=1 PYTHONPATH=$HOME/torch_env:/home/user/opcg-sim-backend
NAME=$1; B=~/pdl/beds; NETS="$HOME/pdl/nets/nrel_$NAME.ep0.npz $HOME/pdl/nets/nrel_$NAME.ep1.npz"
python tests/scripts/value_teach.py tvval --out n1_results/value_teach --tag $NAME --nets $NETS > $B/tvval_$NAME.log 2>&1; mv n1_results/value_teach/tvval_$NAME.json $B/tvval_$NAME.json; echo "TVVAL $(date -u +%H:%M)" >> $B/$NAME.log
python tests/scripts/don_jitter_rows.py rows --nets $NETS > $B/jitter_rows_$NAME.log 2>&1; mv n1_results/don_jitter/jitter_rows.json $B/jitter_rows_$NAME.json; git checkout -q n1_results/don_jitter/jitter_rows.json; echo "ROWS $(date -u +%H:%M)" >> $B/$NAME.log
python -m opcg_sim.learned.train.n_rel_band --in $HOME/n74c/n_records/n74c_w*_p[0-3] --nrel $NETS --holdout-mod 7 --out $B/band_n74c_$NAME.json > $B/band_n74c_$NAME.log 2>&1; echo "BAND $(date -u +%H:%M)" >> $B/$NAME.log
python tests/scripts/value_teach.py rocks --out $B --tag $NAME --nets $NETS > $B/rocks_$NAME.log 2>&1; echo "ROCKS $(date -u +%H:%M)" >> $B/$NAME.log
echo BEDS_DONE >> $B/$NAME.log
