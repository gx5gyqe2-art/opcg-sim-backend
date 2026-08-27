#!/bin/bash
set -uo pipefail
cd /home/user/opcg-sim-backend
GEN_PID=1639
INTERVAL=1800
BRANCH=claude/n8-w02
RECORDS_DIR=n8_records
LOGFILE=n8_gen.log

(tail -n +1 -F "$LOGFILE" 2>/dev/null | grep -E --line-buffered "games_done|shard|Traceback|Error|Exception|N_RECORD_DONE|Killed|error") &
TAILPID=$!

push_with_retry() {
  attempt=0
  while true; do
    out=$(git push -u origin "$BRANCH" 2>&1)
    rc=$?
    if [ $rc -eq 0 ]; then
      echo "$out" | tail -5
      echo "PUSH_OK"
      return 0
    fi
    attempt=$((attempt+1))
    echo "$out" | tail -5
    if [ $attempt -ge 4 ]; then
      echo "PUSH_FAILED_ALL_RETRIES"
      return 1
    fi
    delay=$((2**attempt))
    echo "PUSH_RETRY attempt=$attempt sleeping ${delay}s"
    sleep "$delay"
  done
}

commit_push() {
  if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
    git checkout "$BRANCH" 2>&1 | tail -3
  else
    git checkout -b "$BRANCH" 2>&1 | tail -3
  fi
  git add -f "$RECORDS_DIR" "$LOGFILE"
  if git diff --cached --quiet; then
    echo "COMMIT_SKIP_NO_CHANGES"
  else
    git commit -m "n8棋譜 w02: 進捗 seed1002000" 2>&1 | tail -5
    push_with_retry
  fi
}

LAST=$(date +%s)
while true; do
  sleep 60
  if ! kill -0 "$GEN_PID" 2>/dev/null; then
    echo "PROCESS_EXITED"
    commit_push
    echo "FINAL_COMMIT_DONE"
    break
  fi
  NOW=$(date +%s)
  if [ $((NOW-LAST)) -ge $INTERVAL ]; then
    LAST=$NOW
    echo "PERIODIC_COMMIT_TRIGGER"
    commit_push
  fi
done
kill "$TAILPID" 2>/dev/null
echo "MONITOR_EXIT"
