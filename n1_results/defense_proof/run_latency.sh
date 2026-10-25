#!/bin/sh
# 費用（§7.51）: 出荷の設定（r23・1,280 回・世界 2 本・予算なし・詰みの証明 root）で 20 局、入れない形と入れた形を 1 コアずつ同時に。
cd /home/user/opcg-sim-backend
D=n1_results/defense_proof
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
L="python tests/scripts/rs_decide_latency.py --sims 1280 --worlds 2 --budget 0 --lethal-proof root --games 20"
taskset -c 0 $L > $D/latency_off.json 2> $D/latency_off.log &
taskset -c 1 $L --extra '{"defense_proof": true}' > $D/latency_on.json 2> $D/latency_on.log &
wait
echo "LATENCY_DONE $(date -u +%H:%M)" >> $D/progress.log
