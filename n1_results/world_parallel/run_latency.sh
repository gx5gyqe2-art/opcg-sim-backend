#!/bin/sh
# §7.55 の 1 手の時間（出荷の設定: sims 1280・枝予算なし・根の詰みの証明・20 局・4 コア）。
# 同じ seed の局集合を世界の本数ごとに打つ（スレッドの上限は打つ手を変えない＝同じ局を打つ）。
cd "$(dirname "$0")/../.."
OUT=n1_results/world_parallel/latency.jsonl
for cfg in "1 1" "2 1" "2 2" "2 4" "4 1" "4 2" "4 4"; do
  set -- $cfg
  OPCG_LOG_SILENT=1 python tests/scripts/rs_decide_latency.py --worlds $1 --world-threads $2 \
    --games 20 --lethal-proof root | tail -1 >> $OUT
done
echo ALL_DONE >> $OUT
