#!/bin/sh
# 守りの再監査（§7.44・§7.24 の設定）: reaudit-wr と同じ局面（seed 3351000〜3351299・両席 r13d・世界 1 本 × 1,280 回・予算なし・証明 off・random × synth_roles）を修正後の wheel で。
cd /home/user/opcg-sim-backend
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
D=n1_results/overflow_trigger
F=$D/defense
NET=opcg_sim/data/learned/nrel_r13d.npz
python tests/scripts/defense_audit.py run --games 300 --seed-base 3351000 --workers 3 --net $NET --out $F >> $D/defense_run.log 2>&1
echo "DEF_RUN_DONE $(date -u +%H:%M)" >> $D/progress.log
python tests/scripts/defense_audit.py verify --games 300 --seed-base 3351000 --workers 3 --net $NET --out $F > $D/defense_verify.log 2>&1
python tests/scripts/defense_audit.py summarize --games 300 --seed-base 3351000 --net $NET --out $F > $D/defense_summary.log 2>&1
echo "DEFENSE_DONE $(date -u +%H:%M)" >> $D/progress.log
