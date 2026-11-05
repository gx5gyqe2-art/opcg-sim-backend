#!/bin/bash
cd /home/user/opcg-sim-backend
export OPCG_LOG_SILENT=1 OPCG_DUMP_CACHE=$HOME/packcache
D=n1_results/value_next/beds
until [ -s $HOME/vn/nets/nrel_vn.ep0.npz ]; do sleep 30; done
sleep 20; cp $HOME/vn/nets/nrel_vn.ep0.npz n1_results/value_next/nrel_vn.ep0.npz
echo "EP0_FOUND $(date -u +%H:%M) rss_train $(ps -o rss= -p $(pgrep -f n_rel_train | head -1) | awk '{printf "%.1f", $1/1048576}') GB"
bash ~/vn/beds.sh vn_e0 fast n1_results/value_next/nrel_vn.ep0.npz
grep npz $D/tvval_vn_e0.log; grep npz $D/jitter_rows_vn_e0.log | cut -c1-200
N="opcg_sim/data/learned/nrel_r28x4.npz $HOME/nets/nrel_r29f4.ep0.npz $HOME/nets/nrel_r29f4.ep1.npz n1_results/value_next/nrel_vn.ep0.npz"
python -m opcg_sim.learned.train.n_rel_band --in $HOME/n74c/n_records/n74c_w*_p[0-3] --nrel $N --holdout-mod 7 --limit 100000 --out $D/band_n74c_lim100k_e0.json > $D/band_n74c_lim100k_e0.log 2>&1
grep -E "^  nrel|calib|    all " $D/band_n74c_lim100k_e0.log | cut -c1-110
echo "EP0_BEDS_DONE $(date -u +%H:%M)"
