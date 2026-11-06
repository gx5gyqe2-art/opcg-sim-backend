#!/bin/sh
# 分かれた局だけ、旗を入れた形の守りの監査を分かれた点から先だけ打ち、確かめ・数える（§7.85）。入れない形は既存の監査を絞って使う。
cd /home/user/opcg-sim-backend
D=n1_results/defense_dialog_tree
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
for m in life1 all; do
  F=$D/partial_$m
  python $D/defense_partial.py $m run --seeds $D/div_$m.txt --workers 4 --out $F --defense-dialog-tree $m > $D/partial_${m}_run.log 2>&1
  P=$F/points_seeds_div_$m.jsonl
  python $D/defense_partial.py $m verify --points $P --workers 4 --out $F --defense-dialog-tree $m > $D/partial_${m}_verify.log 2>&1
  python $D/defense_partial.py $m summarize --points $P --out $F > $D/partial_${m}_summary.log 2>&1
  echo "PARTIAL_${m}_DONE $(date -u +%H:%M)" >> $D/progress.log
done
