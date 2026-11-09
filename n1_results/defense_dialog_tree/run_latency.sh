#!/bin/sh
# 費用（§7.85）: 出荷の設定（r29f4・1,280 回・世界 4 本・予算なし・詰みの証明 root）で 20 局、off／life1／all を 1 コアずつ同時に。
cd /home/user/opcg-sim-backend
D=n1_results/defense_dialog_tree
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
L="python tests/scripts/rs_decide_latency.py --sims 1280 --worlds 4 --budget 0 --lethal-proof root --games 20"
taskset -c 0 $L > $D/latency_off.json 2> $D/latency_off.log &
taskset -c 1 $L --defense-dialog-tree life1 > $D/latency_life1.json 2> $D/latency_life1.log &
taskset -c 2 $L --defense-dialog-tree all > $D/latency_all.json 2> $D/latency_all.log &
wait
echo "LATENCY_DONE $(date -u +%H:%M)" >> $D/progress.log
