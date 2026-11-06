#!/bin/sh
# run_local.sh の守りの run を引き継ぐ（続きの 50 局はシャード s06 へ移した）: run の終わりを待って verify → summarize。
cd /home/user/opcg-sim-backend
D=n1_results/reaudit_r29f4
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
F=$D/defense
while kill -0 22425 2>/dev/null; do sleep 30; done
echo STAGE_DEF_RUN_DONE >> $D/progress.log
python tests/scripts/defense_audit.py verify --games 300 --seed-base 4462000 --workers 4 --out $F > $D/defense_verify.log 2>&1
python tests/scripts/defense_audit.py summarize --games 300 --seed-base 4462000 --out $F > $D/defense_summary.log 2>&1
echo DEFENSE_DONE >> $D/progress.log
