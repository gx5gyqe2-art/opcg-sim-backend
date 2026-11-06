#!/bin/sh
cd /home/user/opcg-sim-backend
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
B=n1_results/reaudit_r29f4/ba.py
python $B judge   --workers 4 --out n1_results/reaudit_r29f4/shards/s01 > n1_results/reaudit_r29f4/shards/s01/judge.log 2>&1
python $B recheck --workers 4 --out n1_results/reaudit_r29f4/shards/s01 > n1_results/reaudit_r29f4/shards/s01/recheck.log 2>&1
python $B verify  --workers 4 --out n1_results/reaudit_r29f4/shards/s01 > n1_results/reaudit_r29f4/shards/s01/verify.log 2>&1
python $B winstab --workers 4 --out n1_results/reaudit_r29f4/shards/s01 > n1_results/reaudit_r29f4/shards/s01/winstab.log 2>&1
python $B summary --out n1_results/reaudit_r29f4/shards/s01 > n1_results/reaudit_r29f4/shards/s01/summary.log 2>&1
echo ALL_DONE > n1_results/reaudit_r29f4/shards/s01/ALL_DONE
