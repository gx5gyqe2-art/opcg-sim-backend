#!/usr/bin/env bash
# w2 generator supervisor: auto-restart pd_gen on crash. Logs to LOG.
# Not committed to the branch (scratch operational wrapper).
set -u
REPO=/home/user/opcg-sim-backend
LOG=${OPCG_W2_LOG:-/tmp/pd-w2-gen.log}
cd "$REPO" || exit 1

export OPCG_P3_LEAD_SLOTS=2 OPCG_P3_EFF_DIM=116
export OPCG_PD_NET_BRANCH=claude/v5-net OPCG_PD_DATA_BRANCH=claude/v5-data-w2
export OPCG_PD_WT=/tmp/pd-w2 OPCG_LOG_SILENT=1 PYTHONPATH=tests

WORKERS=${W2_WORKERS:-3}
GAMES=${W2_GAMES:-32}

while true; do
  echo "=== [supervise] $(date -u +%FT%TZ) starting pd_gen (workers=$WORKERS games=$GAMES) ===" >> "$LOG"
  python tests/scripts/pd_gen.py --enc-version 4 --sims 160 --games "$GAMES" --workers "$WORKERS" \
    --l1-mix 0.25 --mark-seed-frac 0.15 --dirichlet-eps 0.15 >> "$LOG" 2>&1
  code=$?
  echo "=== [supervise] $(date -u +%FT%TZ) pd_gen exited code=$code; restart in 10s ===" >> "$LOG"
  sleep 10
done
