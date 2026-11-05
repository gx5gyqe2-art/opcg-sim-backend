#!/bin/bash
# 不変量: 対照 3 本（r28x4・r29f4 ep0・ep1）を tvval・82j の群の差・74c の取り置きの帯にかける
cd /home/user/opcg-sim-backend
N="opcg_sim/data/learned/nrel_r28x4.npz $HOME/nets/nrel_r29f4.ep0.npz $HOME/nets/nrel_r29f4.ep1.npz"
bash ~/vn/beds.sh ctrl fast $N
bash ~/vn/beds.sh ctrl band74c $N
echo INVARIANTS_DONE >> n1_results/value_next/beds/band_n74c_ctrl.log
