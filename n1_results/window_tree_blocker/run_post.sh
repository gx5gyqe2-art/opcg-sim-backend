#!/bin/sh
# 打ち直した局の見逃しの点を確かめ、修正前の記録と組んで 300 局ずつ数える。
cd /home/user/opcg-sim-backend
D=n1_results/window_tree_blocker
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
until grep -q CHAIN_DONE $D/progress.log 2>/dev/null; do sleep 30; done
for B in 3013000 3351000; do
  python tests/scripts/defense_audit.py verify --points $D/defense_$B/points_seeds_div_seeds_$B.jsonl --workers 4 \
    --net opcg_sim/data/learned/nrel_r13d.npz --out $D/defense_$B > $D/defense_verify_$B.log 2>&1
  python $D/merge_defense.py $B >> $D/defense_merge.log 2>&1
  python tests/scripts/defense_audit.py summarize --seed-base $B --games 300 --out $D/defense_$B/merged > $D/defense_summary_$B.log 2>&1
  echo DEF_POST_DONE_$B >> $D/progress.log
done
echo POST_DONE >> $D/progress.log
