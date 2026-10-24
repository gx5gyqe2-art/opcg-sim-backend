#!/bin/sh
# 費用（出荷の設定・20 局・1 コアずつ・修正後と旧い旗を同時に）と、a1 帯の再生 golden の手の列（修正後・旧い旗）。
cd /home/user/opcg-sim-backend
D=n1_results/window_tree_blocker
export PYTHONHASHSEED=0 OPCG_LOG_SILENT=1
L="python tests/scripts/rs_decide_latency.py --worlds 2 --lethal-proof root --games 20"
taskset -c 0 $L > $D/latency_new.json 2> $D/latency_new.err &
taskset -c 1 $L --window-tree-counter-legacy > $D/latency_legacy.json 2> $D/latency_legacy.err &
taskset -c 2 python $D/golden_play.py $D/golden_new.json > $D/golden_play_new.log 2>&1 &
taskset -c 3 python $D/golden_play.py $D/golden_new_legacy.json '{"window_tree_counter_legacy": true}' > $D/golden_play_new_legacy.log 2>&1 &
wait
echo COST_GOLDEN_DONE >> $D/progress.log
