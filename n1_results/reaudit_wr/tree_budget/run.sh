#!/bin/sh
# §7.43 の 3: 3301005・107（と対照の 3301025・117）を短い木の予算 ×1／×2／×4（sims＝1,280／2,560／5,120 → 木は sims ÷ 候補の数）で 8 回ずつ選ばせる。
cd /home/user/opcg-sim-backend
S=n1_results/reaudit_wr/tree_budget
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
for m in 1 2 4; do
  sims=$((1280 * m))
  nice python tests/scripts/blunder_audit.py winstab --workers 1 --out $S --tag x$m --opts "{\"sims\":$sims}" > $S/x$m.log 2>&1
  nice python tests/scripts/blunder_audit.py winstab --workers 1 --out $S --tag x${m}_w4 --opts "{\"sims\":$sims,\"window_tree_worlds\":4}" > $S/x${m}_w4.log 2>&1
done
echo TREE_DONE > $S/done
