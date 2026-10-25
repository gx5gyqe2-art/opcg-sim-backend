#!/bin/sh
# §7.49: 1 手の時間（世界 2 本・1,280 回・枝予算なし・証明 root・20 局・r23・1 コアずつ同時）
cd "$(dirname "$0")/../.."
i=0
for n in 20000 40000 80000 160000; do
  taskset -c $i python tests/scripts/rs_decide_latency.py --sims 1280 --worlds 2 --budget 0 --games 20 \
    --lethal-proof root --lethal-nodes $n > n1_results/lethal_nodes/lat_n$((n/1000))k.json 2> n1_results/lethal_nodes/lat_n$((n/1000))k.log &
  i=$((i+1))
done
wait
