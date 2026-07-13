#!/usr/bin/env bash
# v5 本走 generator worker w1 launcher (session-local, not committed).
set -u
cd /home/user/opcg-sim-backend || exit 1
git fetch origin claude/cpu-spec-improvements-yw91jd 2>&1 | tail -1
git checkout claude/cpu-spec-improvements-yw91jd 2>&1 | tail -1
export OPCG_P3_LEAD_SLOTS=2 OPCG_P3_EFF_DIM=116
export OPCG_PD_NET_BRANCH=claude/v5-net OPCG_PD_DATA_BRANCH=claude/v5-data-w1
export OPCG_PD_WT=/tmp/pd-w1 OPCG_LOG_SILENT=1 PYTHONPATH=tests
exec python tests/scripts/pd_gen.py --enc-version 4 --sims 160 --games 32 --workers 3 \
  --l1-mix 0.25 --mark-seed-frac 0.15 --dirichlet-eps 0.15
