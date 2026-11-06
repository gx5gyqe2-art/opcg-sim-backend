#!/bin/sh
cd /home/user/opcg-sim-backend
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
python tests/scripts/defense_audit.py run --games 100 --seed-base 4462200 --workers 4 --out n1_results/reaudit_r29f4/defense_s08 > n1_results/reaudit_r29f4/defense_s08/run.log 2>&1
python tests/scripts/defense_audit.py verify --games 100 --seed-base 4462200 --workers 4 --out n1_results/reaudit_r29f4/defense_s08 > n1_results/reaudit_r29f4/defense_s08/verify.log 2>&1
python tests/scripts/defense_audit.py summarize --games 100 --seed-base 4462200 --out n1_results/reaudit_r29f4/defense_s08 > n1_results/reaudit_r29f4/defense_s08/summary.log 2>&1
echo ALL_DONE > n1_results/reaudit_r29f4/defense_s08/ALL_DONE
