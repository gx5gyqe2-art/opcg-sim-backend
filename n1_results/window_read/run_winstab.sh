#!/bin/sh
# 窓の揺れの K の掃き（§7.42 の 2）: 修正 2 だけ（続き・木は旧い形）で K=1/2/4/8、全部修正後で K=4/8。
W='OPCG_LOG_SILENT=1 python tests/scripts/blunder_audit.py winstab --workers 4 --tag-out n1_results/window_read'
for k in 1 2 4 8; do
  t0=$(date +%s)
  sh -c "$W --tag w$k --opts '{\"window_carry_legacy\":true,\"window_tree_legacy\":true,\"window_worlds\":$k}'"
  echo "w$k $(( $(date +%s) - t0 )) sec"
done
for k in 4 8; do
  t0=$(date +%s)
  sh -c "$W --tag fixed$k --opts '{\"window_worlds\":$k}'"
  echo "fixed$k $(( $(date +%s) - t0 )) sec"
done
echo SWEEP_DONE
