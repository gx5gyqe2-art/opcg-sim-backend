#!/bin/sh
# life1 の部分監査の打ち直し（1 回目はゲート〔make test・audit-cross〕と同時に回り、判定器の時間切れ＝判定不能が増えた疑い）。
cd /home/user/opcg-sim-backend
D=n1_results/defense_dialog_tree
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
m=life1; F=$D/partial_${m}_r2
python $D/defense_partial.py $m run --seeds $D/div_$m.txt --workers 4 --out $F --defense-dialog-tree $m > $D/partial_${m}_r2_run.log 2>&1
P=$F/points_seeds_div_$m.jsonl
python $D/defense_partial.py $m verify --points $P --workers 4 --out $F --defense-dialog-tree $m > $D/partial_${m}_r2_verify.log 2>&1
python $D/defense_partial.py $m summarize --points $P --out $F > $D/partial_${m}_r2_summary.log 2>&1
echo "PARTIAL_life1_r2_DONE $(date -u +%H:%M)" >> $D/progress.log
