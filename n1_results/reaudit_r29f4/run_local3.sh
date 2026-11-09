#!/bin/sh
# 守りの見逃し: このセッションは seed 4462000〜4462099 の 100 局（4462100〜199・200〜299 はシャード s07・s08）。
# run_local.sh／run_local2.sh の 300 局の run（21 局ぶん）を止め、同じ記録を points_4462000_100.jsonl に移して再開した。
cd /home/user/opcg-sim-backend
D=n1_results/reaudit_r29f4
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
F=$D/defense
python tests/scripts/defense_audit.py run --games 100 --seed-base 4462000 --workers 4 --out $F > $D/defense_run.log 2>&1
echo STAGE_DEF_RUN_DONE >> $D/progress.log
python tests/scripts/defense_audit.py verify --games 100 --seed-base 4462000 --workers 4 --out $F > $D/defense_verify.log 2>&1
python tests/scripts/defense_audit.py summarize --games 100 --seed-base 4462000 --out $F > $D/defense_summary.log 2>&1
echo DEFENSE_DONE >> $D/progress.log
