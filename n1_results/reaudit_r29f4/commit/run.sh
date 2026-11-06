#!/bin/sh
cd /home/user/opcg-sim-backend
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
O=n1_results/reaudit_r29f4/commit
python tests/scripts/commit_audit.py run --games 50 --seed-base 4463000 --workers 4 --judge-window-new --out $O > $O/run.log 2>&1
python tests/scripts/commit_audit.py summary --out $O > $O/summary.log 2>&1
echo ALL_DONE > $O/ALL_DONE
