#!/usr/bin/env bash
# w2 generator 10-min self-check. One heartbeat line per cycle (keeps session
# alive + lets the agent glance). Restarts the supervisor if it died.
LOG=/tmp/pd-w2-gen.log
SUP=tests/scripts/pd_gen_w2_supervise.sh
cd /home/user/opcg-sim-backend || exit 1
while true; do
  now=$(date -u +%FT%TZ)
  sup=$(pgrep -f pd_gen_w2_supervise >/dev/null && echo OK || echo DEAD)
  proc=$(pgrep -f "scripts/pd_gen.py" >/dev/null && echo OK || echo DEAD)
  # Auto-heal: if the supervisor loop itself died, relaunch it.
  if [ "$sup" = DEAD ]; then
    nohup bash "$SUP" >/dev/null 2>&1 &
    sup="RESTARTED"
  fi
  ok=$(grep -c "push=OK" "$LOG" 2>/dev/null || echo 0)
  lastid=$(grep -oE "batch[0-9]+ .*push=OK" "$LOG" 2>/dev/null | tail -1 | grep -oE "batch[0-9]+" | tail -1)
  recentfail=$(tail -60 "$LOG" 2>/dev/null | grep -c "push=FAIL")
  tb=$(tail -80 "$LOG" 2>/dev/null | grep -c "Traceback")
  if [ -f "$LOG" ]; then stale=$(( $(date +%s) - $(stat -c %Y "$LOG") )); else stale=-1; fi
  last=$(tail -1 "$LOG" 2>/dev/null | cut -c1-120)
  flag=""
  [ "$proc" = DEAD ] && flag="ALERT "
  [ "$sup" = RESTARTED ] && flag="ALERT "
  [ "${recentfail:-0}" -ge 3 ] && flag="ALERT "
  [ "${tb:-0}" -ge 3 ] && flag="ALERT "
  # L1-mix games (--l1-mix) can make a single batch legitimately long, and
  # pd_gen prints nothing mid-batch; only alarm on a truly pathological stall.
  [ "${stale:-0}" -gt 5400 ] && flag="ALERT "
  echo "${flag}HB $now proc=$proc sup=$sup okBatches=${ok:-0} last=${lastid:-none} recentFAIL=${recentfail:-0} tb=${tb:-0} stale=${stale}s | $last"
  sleep 600
done
