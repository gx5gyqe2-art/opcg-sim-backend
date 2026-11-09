#!/bin/bash
cd /home/user/opcg-sim-backend
export OPCG_LOG_SILENT=1 OPCG_DUMP_CACHE=$HOME/packcache
D=n1_results/value_next/beds
ALL="opcg_sim/data/learned/nrel_r28x4.npz $HOME/nets/nrel_r29f4.ep0.npz $HOME/nets/nrel_r29f4.ep1.npz n1_results/value_next/nrel_vn_run1.ep0.npz"
bash ~/vn/beds.sh run1 rocks n1_results/value_next/nrel_vn_run1.ep0.npz $HOME/nets/nrel_r29f4.ep0.npz
grep "点" $D/rocks_run1.log
bash ~/vn/beds.sh run1 band74c $ALL
grep -E "^  nrel|calib|    all " $D/band_n74c_run1.log | cut -c1-110
bash ~/vn/beds.sh run1 band70f $ALL
grep -E "^  nrel|calib|    all " $D/band_n70f_run1.log | cut -c1-110
echo "RUN1_BEDS_DONE $(date -u +%H:%M)"
