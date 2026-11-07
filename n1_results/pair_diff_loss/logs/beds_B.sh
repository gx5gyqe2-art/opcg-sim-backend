#!/bin/bash
cd /home/user/opcg-sim-backend
export OPCG_LOG_SILENT=1 PYTHONPATH=$HOME/torch_env:/home/user/opcg-sim-backend
python tests/scripts/pair_diff.py beds --pairs $HOME/pdl/pairs.npz --nets $HOME/pdl/ctrl/nrel_r29f4.ep0.npz $HOME/pdl/ctrl/nrel_r29f4.ep1.npz $HOME/pdl/nets/nrel_pdh.ep0.npz $HOME/pdl/nets/nrel_pdh.ep1.npz --ref $HOME/pdl/ctrl/nrel_r29f4.ep0.npz --out-json $HOME/pdl/beds/beds_pairs_B.json > $HOME/pdl/beds/beds_pairs_B.log 2>&1; echo "PAIRS $(date -u +%H:%M)" >> $HOME/pdl/beds/pdh.log
$HOME/pdl/beds_arm.sh pdh
