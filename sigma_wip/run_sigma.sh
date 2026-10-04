#!/bin/bash
# usage: bash sigma_wip/run_sigma.sh real|syn     (resumable: skips a step whose output json exists)
# Run detached: setsid nohup bash sigma_wip/run_sigma.sh real > sigma_wip/run_session.log 2>&1 &
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd); W=$ROOT/sigma_wip; OUT=$W/results; mkdir -p "$OUT"; cd "$ROOT"
export OPCG_LOG_SILENT=1 OPCG_PLAN_STORE=$OUT/store     # on-disk plan store: later steps reuse earlier solves
if [ "$1" = real ]; then IN="$W/data/w41"; else IN="$W/data/w39 $W/data/w42"; fi
T=$1
step() { out=$1; shift; if [ -s "$out" ]; then echo "skip $out"; return; fi; echo "run $(basename $out) $(date -u +%H:%M:%S)"; "$@" > /dev/null 2> "${out%.json}.err" || { echo "FAILED $out"; rm -f "$out"; }; echo "done $(basename $out) $(date -u +%H:%M:%S)"; }
step $OUT/cb_$T.json  python tests/scripts/crossing_bridge.py --in $IN --out $OUT/cb_$T.json
step $OUT/tb_$T.json  python tests/scripts/theory_bridge.py   --in $IN --out $OUT/tb_$T.json
step $OUT/wc_$T.json  python tests/scripts/win_calib.py       --in $IN --pre-settle on --json $OUT/wc_$T.json
step $OUT/pa_$T.json  python tests/scripts/pre_settle_asymmetry.py --in $IN --json $OUT/pa_$T.json
echo SESSION_DONE
