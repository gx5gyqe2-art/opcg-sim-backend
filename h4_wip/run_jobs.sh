#!/bin/bash
# usage: bash h4_wip/run_jobs.sh <comma-separated job indices from jobs_all.txt>  (e.g. 0,1,6)
# Runs ONE heavy job at a time, skips a job whose --json/--out output already exists in $OUT (resumable after container restart).
# Run it detached:  setsid nohup bash h4_wip/run_jobs.sh 0,1,6 > h4_wip/run_$$.log 2>&1 &
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export W=$ROOT/h4_wip OUT=$ROOT/h4_wip/results OPCG_LOG_SILENT=1
export SC=$W    # (kept for scripts that read $SC)
mkdir -p "$OUT"; cd "$ROOT"
cp -n "$W"/plans/*.pkl "$OUT"/ 2>/dev/null || true   # plan caches (not committed from results/)
IFS=, read -ra IDX <<< "$1"
for i in "${IDX[@]}"; do
  line=$(sed -n "$((i+1))p" "$W/jobs_all.txt")
  out=$(echo "$line" | grep -oE '(--out|--json) [^ ]+' | tail -1 | awk '{print $2}')
  out=$(eval echo "$out")
  if [ -n "$out" ] && [ -s "$out" ]; then echo "skip $i ($out exists)"; continue; fi
  echo "run $i $(date -u +%H:%M:%S)"
  ( eval "$line" ) >> "$OUT/run.log" 2>&1
  echo "done $i $(date -u +%H:%M:%S)"
done
echo SESSION_DONE
