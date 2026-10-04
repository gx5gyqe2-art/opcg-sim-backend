#!/bin/bash
# usage: bash cand_wip/run_cand.sh <base|take|block>   (resumable: skips a step whose output json exists)
# Run detached: setsid nohup bash cand_wip/run_cand.sh take > cand_wip/run_session.log 2>&1 &
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd); W=$ROOT/cand_wip; OUT=$W/results; mkdir -p "$OUT"; cd "$ROOT"
export OPCG_LOG_SILENT=1 OPCG_PLAN_STORE=$OUT/store_$1       # on-disk plan store (one per variant: keys include the switch values anyway)
N=100                                                       # games per record set (first N games; same sample for baseline and candidates)
case $1 in
  base)  VAR="base";  C="" ;;
  take)  VAR="take";  C="slope_take=life" ;;
  block) VAR="block"; C="slope_block=on" ;;
  *) echo "unknown variant"; exit 2 ;;
esac
step() { out=$1; shift; if [ -s "$out" ]; then echo "skip $out"; return; fi; echo "run $(basename $out) $(date -u +%H:%M:%S)"; CAND="$C" "$@" > /dev/null 2> "${out%.json}.err" || { echo "FAILED $out"; rm -f "$out"; }; echo "done $(basename $out) $(date -u +%H:%M:%S)"; }
SH="python $W/cand_shim.py"
for rec in real syn; do
  if [ $rec = real ]; then IN="$W/data/w41"; else IN="$W/data/w39 $W/data/w42"; fi
  step $OUT/cb_${rec}_$VAR.json $SH crossing_bridge --in $IN --limit-games $N --out $OUT/cb_${rec}_$VAR.json
  step $OUT/wc_${rec}_$VAR.json $SH win_calib --in $IN --games $N --pre-settle on --json $OUT/wc_${rec}_$VAR.json
  step $OUT/rl_${rec}_$VAR.json $SH relative_ledger --in $IN --games 60 --json $OUT/rl_${rec}_$VAR.json
done
if [ $1 = base ]; then      # the third candidate only touches the relative ledger: run it here, same sample
  C="kappa_sigma=match"
  for rec in real syn; do
    if [ $rec = real ]; then IN="$W/data/w41"; else IN="$W/data/w39 $W/data/w42"; fi
    step $OUT/rl_${rec}_kappa.json $SH relative_ledger --in $IN --games 60 --json $OUT/rl_${rec}_kappa.json
  done
fi
echo SESSION_DONE
