#!/bin/sh
# 守りの再監査の 2 つの局面で、修正（§7.45）で手が分かれる局を探す（分かれた局だけ監査を打ち直す）。
cd /home/user/opcg-sim-backend
D=n1_results/window_tree_blocker
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
python $D/diverge.py --seed-base 3013000 --games 300 --workers 4 --out $D/diverge_3013000.jsonl > $D/diverge_3013000.log 2>&1
python $D/diverge.py --seed-base 3351000 --games 300 --workers 4 --out $D/diverge_3351000.jsonl > $D/diverge_3351000.log 2>&1
echo DIVERGE_ALL_DONE >> $D/progress.log
