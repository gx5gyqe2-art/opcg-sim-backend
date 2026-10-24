#!/bin/sh
# 分かれた局の守りの監査（分かれた手から先だけ）→ 組む → 新しい見逃しの点を確かめる → 数える（2 つの局面）。
cd /home/user/opcg-sim-backend
D=n1_results/window_tree_blocker
R13D=opcg_sim/data/learned/nrel_r13d.npz
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
for B in 3013000 3351000; do
  python $D/defense_partial.py $B run --seeds $D/part_seeds_$B.txt --workers 4 --net $R13D \
    --out $D/defense_${B}_partial > $D/defense_partial_run_$B.log 2>&1
  echo DEF_PART_DONE_$B >> $D/progress.log
done
for B in 3013000 3351000; do
  python $D/merge_defense.py $B points >> $D/defense_merge.log 2>&1
  python tests/scripts/defense_audit.py verify --points $D/defense_$B/merged/new_points.jsonl --workers 4 --net $R13D \
    --out $D/defense_$B/merged/new_verify > $D/defense_verify_$B.log 2>&1
  python $D/merge_defense.py $B verify >> $D/defense_merge.log 2>&1
  python tests/scripts/defense_audit.py summarize --seed-base $B --games 300 --out $D/defense_$B/merged > $D/defense_summary_$B.log 2>&1
  echo DEF_POST_DONE_$B >> $D/progress.log
done
echo POST_DONE >> $D/progress.log
