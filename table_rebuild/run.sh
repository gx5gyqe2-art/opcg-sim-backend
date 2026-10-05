#!/bin/bash
# usage: bash table_rebuild/run.sh <phase>   phase = before | s2 | s2b | s3 | after
# resumable: a step whose output json exists is skipped. Run detached:
#   setsid nohup bash table_rebuild/run.sh before >> table_rebuild/run_session.log 2>&1 &
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd); OUT=$ROOT/table_rebuild/results; mkdir -p "$OUT"; cd "$ROOT"
D=${TR_DATA:-/home/user/tr/sigma_wip/data}
export OPCG_LOG_SILENT=1 OPCG_RD_KERNEL=rs OPCG_PLAN_STORE=${TR_STORE:-/home/user/tr/store}
REAL="$D/w41"; SYN="$D/w39 $D/w42"
P=$1
step() { out=$1; shift; if [ -s "$out" ]; then echo "skip $(basename $out)"; return; fi
  echo "run $(basename $out) $(date -u +%F' '%H:%M:%S)"
  "$@" > "${out%.json}.log" 2>&1 || { echo "FAILED $(basename $out)"; rm -f "$out"; return; }
  echo "done $(basename $out) $(date -u +%F' '%H:%M:%S)"; }
cb() { step $OUT/${P}_cb_real.json python tests/scripts/crossing_bridge.py --in $REAL --out $OUT/${P}_cb_real.json
       step $OUT/${P}_cb_syn.json  python tests/scripts/crossing_bridge.py --in $SYN  --out $OUT/${P}_cb_syn.json; }
evals() { for T in real syn; do if [ $T = real ]; then IN=$REAL; else IN=$SYN; fi
  step $OUT/${P}_wc_$T.json python tests/scripts/win_calib.py --in $IN --pre-settle on --json $OUT/${P}_wc_$T.json
  step $OUT/${P}_pa_$T.json python tests/scripts/pre_settle_asymmetry.py --in $IN --json $OUT/${P}_pa_$T.json
  step $OUT/${P}_rl_$T.json python tests/scripts/relative_ledger.py --in $IN --games 60 --json $OUT/${P}_rl_$T.json
  done; }
case $P in
  before|after) cb; evals ;;
  s2|s2b) cb ;;
  s3) step $OUT/s3_tb_real.json python tests/scripts/theory_bridge.py --in $REAL --out $OUT/s3_tb_real.json
      step $OUT/s3_tb_syn.json  python tests/scripts/theory_bridge.py --in $SYN  --out $OUT/s3_tb_syn.json ;;
  *) echo "unknown phase $P"; exit 2 ;;
esac
echo "PHASE_DONE $P $(date -u +%F' '%H:%M:%S)"
