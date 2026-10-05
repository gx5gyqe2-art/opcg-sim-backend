#!/bin/bash
# 第 2a 段の L5（器まるごと）: crossing_bridge を実 5 局（w41）・合成 5 局（w39）で OPCG_RD_KERNEL=py と =rs で回す。
# usage: DATA=<sigma_wip/data を展開した親> OUT=<出力> bash rust2a/run_l5.sh   （再開可: 出力の json があれば飛ばす）
#   記録の取り出し: git archive origin/claude/sigma-wip sigma_wip/data/w41 sigma_wip/data/w39 | tar -x -C $DATA
#   切り出し: 各ディレクトリの先頭 5 局（crossing_bridge の --limit-games 5＝ファイル順の最初の 5 局）
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
: "${DATA:?}" "${OUT:?}"
mkdir -p "$OUT"
export OPCG_LOG_SILENT=1
unset OPCG_PLAN_STORE
for T in real syn; do
  if [ $T = real ]; then IN=$DATA/sigma_wip/data/w41; else IN=$DATA/sigma_wip/data/w39; fi
  for K in py rs; do
    out=$OUT/cb_${T}_${K}.json
    if [ -s "$out" ]; then echo "skip $out"; continue; fi
    echo "run $(basename $out) $(date -u +%H:%M:%S)"
    s=$(date +%s.%N)
    OPCG_RD_KERNEL=$K python tests/scripts/crossing_bridge.py --in $IN --limit-games 5 --out $out.tmp > /dev/null 2> ${out%.json}.err \
      && mv $out.tmp $out || { echo "FAILED $out"; rm -f $out.tmp; continue; }
    e=$(date +%s.%N)
    echo "$T $K $(python -c "print(round($e-$s,1))")" >> $OUT/times.txt
    echo "done $(basename $out) $(date -u +%H:%M:%S)"
  done
done
echo L5_DONE
