#!/bin/sh
# 1 手あたりの時間（§7.87）: 出荷の設定（r29f4・1,280 回・予算なし・詰みの証明 root・守りの対話の木＝既定 all）で 20 局、世界 4／6／8 本を 1 コアずつ同時に。
cd /home/user/opcg-sim-backend
D=n1_results/ddt_default
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
L="python tests/scripts/rs_decide_latency.py --sims 1280 --budget 0 --lethal-proof root --games 20"
taskset -c 0 $L --worlds 4 > $D/latency_w4.json 2> $D/latency_w4.log &
taskset -c 1 $L --worlds 6 > $D/latency_w6.json 2> $D/latency_w6.log &
taskset -c 2 $L --worlds 8 > $D/latency_w8.json 2> $D/latency_w8.log &
wait
echo "LATENCY_DONE $(date -u +%H:%M)" >> $D/progress.log
