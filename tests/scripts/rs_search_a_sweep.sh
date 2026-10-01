#!/usr/bin/env bash
# WP `rs-search-a`（計画 §20.5）の掃引を回す（1 条件＝1 ディレクトリ・9 シナリオ×seed 0/1）。
#
#   S1: sims   160・visits・t=1（今の serve 既定）        … 9 シナリオ
#   S2: sims   640・visits・t=1                           … 9 シナリオ
#   S3: sims 2,560・visits・t=1                           … 9 シナリオ
#   S4: sims 10,240・visits・t=1                          … 3 シナリオ（KEY）
#   S5: sims 16,000・visits・t=1                          … 3 シナリオ（KEY）
#   R1: sims   160・q_min_n(1/8)・t=2                     … 9 シナリオ
#   R2: sims   640・q_min_n(1/8)・t=2                     … 9 シナリオ
#
# 使い方: tests/scripts/rs_search_a_sweep.sh <出力ディレクトリ> [条件...]
#   例) tests/scripts/rs_search_a_sweep.sh scenario_out_a S1 R1 S2 R2 S3 S4 S5
#
# 1 プロセス＝1 シナリオ（seed 0/1 を続けて打つ）。`xargs -P` で並列に回す（既定 4）。
set -u
cd "$(dirname "$0")/../.."
OUT=${1:?出力ディレクトリ}
shift
CONDS=${*:-S1 R1 S2 R2 S3 S4 S5}
JOBS=${JOBS:-4}

ALL=$(ls tests/fixtures/scenarios/*.json | xargs -n1 basename | sed 's/\.json$//')
KEY="enel_human_20260810_t9-9 human_enel_vs_roger_20260904_t7-8 human_doflamingo_vs_luffy_20260904_t10-11"

for cond in $CONDS; do
  case "$cond" in
    S1) sims=160;   names="$ALL"; extra="" ;;
    S2) sims=640;   names="$ALL"; extra="" ;;
    S3) sims=2560;  names="$ALL"; extra="" ;;
    S4) sims=10240; names="$KEY"; extra="" ;;
    S5) sims=16000; names="$KEY"; extra="" ;;
    R1) sims=160;   names="$ALL"; extra="--select-rule q_min_n --q-min-frac 0.125 --root-prior-temp 2.0" ;;
    R2) sims=640;   names="$ALL"; extra="--select-rule q_min_n --q-min-frac 0.125 --root-prior-temp 2.0" ;;
    *)  echo "[sweep] 知らない条件: $cond" >&2; exit 2 ;;
  esac
  echo "[sweep] === $cond sims=$sims $extra ==="
  date -u +'[sweep] start %FT%TZ'
  # shellcheck disable=SC2086
  printf '%s\n' $names | xargs -P "$JOBS" -I{} env OPCG_LOG_SILENT=1 PYTHONPATH=tests \
    python tests/scripts/rs_scenario_play.py play --scenario {} --seeds 2 --sims "$sims" \
      --out "$OUT/$cond" $extra
  date -u +'[sweep] done  %FT%TZ'
done
