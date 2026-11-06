#!/bin/sh
# WP reaudit-r29f4（§7.84）: このセッションで回す分＝守りの見逃し 300 局 → 自分の手番の続き 50 局（各 4 並列）。
cd /home/user/opcg-sim-backend
D=n1_results/reaudit_r29f4
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
F=$D/defense
python tests/scripts/defense_audit.py run --games 300 --seed-base 4462000 --workers 4 --out $F > $D/defense_run.log 2>&1
echo STAGE_DEF_RUN_DONE >> $D/progress.log
python tests/scripts/defense_audit.py verify --games 300 --seed-base 4462000 --workers 4 --out $F > $D/defense_verify.log 2>&1
python tests/scripts/defense_audit.py summarize --games 300 --seed-base 4462000 --out $F > $D/defense_summary.log 2>&1
echo DEFENSE_DONE >> $D/progress.log
python tests/scripts/commit_audit.py run --games 50 --seed-base 4463000 --workers 4 --judge-window-new --out $D/commit > $D/commit_run.log 2>&1
python tests/scripts/commit_audit.py summary --out $D/commit > $D/commit_summary.log 2>&1
echo COMMIT_DONE >> $D/progress.log
