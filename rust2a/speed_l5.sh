#!/bin/bash
# 第 2a 段の速さ（器まるごと）: crossing_bridge を実 5 局・合成 5 局で py／rs1（第 1 段だけ）／rs（第 2a 段）× 3 回。
# usage: DATA=... OUT=... bash rust2a/speed_l5.sh   （再開可: 済んだ回は飛ばす）
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
: "${DATA:?}" "${OUT:?}"
mkdir -p "$OUT"
export OPCG_LOG_SILENT=1
unset OPCG_PLAN_STORE
for rep in 1 2 3; do
for T in real syn; do
  if [ $T = real ]; then IN=$DATA/sigma_wip/data/w41; else IN=$DATA/sigma_wip/data/w39; fi
  for K in py rs1 rs; do
    tag=${T}_${K}_$rep
    if grep -q "^$tag " $OUT/speed_times.txt 2>/dev/null; then continue; fi
    out=$OUT/sp_$tag.json
    s=$(date +%s.%N)
    if [ $K = rs1 ]; then
      OPCG_RD_KERNEL=rs python rust2a/cb_stage1.py --in $IN --limit-games 5 --out $out > /dev/null 2> $OUT/sp_$tag.err || { echo FAILED $tag; continue; }
    else
      OPCG_RD_KERNEL=$K python tests/scripts/crossing_bridge.py --in $IN --limit-games 5 --out $out > /dev/null 2> $OUT/sp_$tag.err || { echo FAILED $tag; continue; }
    fi
    e=$(date +%s.%N)
    echo "$tag $(python -c "print(round($e-$s,1))")" >> $OUT/speed_times.txt
    echo "done $tag"
  done
done
done
echo SPEED_DONE
