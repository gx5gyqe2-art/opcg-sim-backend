#!/bin/sh
# 守りの再監査の局面（seed 3351000〜3351299・両席 r13d・§7.24 の設定）で、守りの証明で手が分かれる局を探す（§7.51）。
cd /home/user/opcg-sim-backend
D=n1_results/defense_proof
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
python $D/diverge.py --seed-base 3351000 --games 300 --workers 4 --no-stop --out $D/diverge_3351000.jsonl > $D/diverge_3351000.log 2>&1
echo "DIVERGE_DONE $(date -u +%H:%M)" >> $D/progress.log
