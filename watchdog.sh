#!/bin/bash
# Resident watchdog for n28 w08 generation.
# Every 15 min: if the generator died and we are short of 960 games, relaunch
# from the next unused seed band; then push all records + logs to the WIP branch.
SRC=/root/opcg-sim-backend
WT=/root/pushtree
BR=claude/n28-w08-wip
LOCK=/root/push.lock
TARGET=960
SEED_LO=2288000
SEED_HI=2288999   # inclusive band cap

log() { echo "[$(date -u +%H:%M:%S)] $*" >> /root/watchdog.log; }

# Unique completed games across every records dir (ground truth, not log text).
count_done() {
  cd "$SRC" && python - <<'EOF' 2>/dev/null || echo "ERR"
import glob, numpy as np
s = set()
for p in glob.glob('n28_records*/n_record_*.npz'):
    try: s |= set(np.load(p)['seed'].tolist())
    except Exception: pass
print(f"{len(s)} {max(s) if s else 0}")
EOF
}

push_wip() {
  exec 9>"$LOCK"; flock -w 300 9 || { log "push: lock timeout"; return 1; }
  cd "$WT" || return 1
  git checkout -q --detach res 2>/dev/null
  rm -rf "$WT"/n28_records*
  cp -r "$SRC"/n28_records* "$WT"/ 2>/dev/null
  cp /root/n28_gen*.log "$WT"/ 2>/dev/null
  # Ship the helper scripts too, so a total environment loss is recoverable
  # from this branch alone (each cycle rebuilds the commit from `res`).
  cp /root/resume.sh /root/watchdog.sh /root/watchdog.log "$WT"/ 2>/dev/null
  git checkout -qB "$BR"
  git add -f n28_records* n28_gen*.log resume.sh watchdog.sh watchdog.log 2>/dev/null
  git -c user.email=g.x5gyqe2@gmail.com -c user.name=worker \
      commit -q -m "wip gen n28 w08 ($1/960局)" 2>/dev/null
  for i in 1 2 3 4; do
    git push -qf -u origin "$BR" 2>/dev/null && { log "pushed wip $1/960"; flock -u 9; return 0; }
    sleep $((2 ** i))
  done
  log "push FAILED"; flock -u 9; return 1
}

while true; do
  read -r DONE MAXSEED <<<"$(count_done)"
  [ "$DONE" = "ERR" ] && { log "count failed"; sleep 900; continue; }

  # Anchored so only the real python process matches -- an unanchored pattern
  # also matches transient tool-call shells that merely quote the command.
  if pgrep -f "^python tests/scripts/n_record_gen\.py" >/dev/null; then
    log "running, done=$DONE"
  elif [ "$DONE" -ge "$TARGET" ]; then
    log "TARGET REACHED done=$DONE"; push_wip "$DONE"; break
  else
    REMAIN=$((TARGET - DONE))
    # Pick the smallest gap K that keeps the whole run inside the seed band.
    NEXT=0
    for K in 8 2 1; do
      CAND=$((MAXSEED + K))
      if [ $((CAND + REMAIN - 1)) -le "$SEED_HI" ]; then NEXT=$CAND; break; fi
    done
    if [ "$NEXT" -eq 0 ]; then
      NEXT=$((MAXSEED + 1))
      REMAIN=$((SEED_HI - NEXT + 1))
      log "band tight: clamping remain=$REMAIN"
    fi
    if [ "$REMAIN" -le 0 ]; then log "no seeds left, done=$DONE"; push_wip "$DONE"; break; fi
    N=2; while [ -e "$SRC/n28_records_r$N" ]; do N=$((N + 1)); done
    log "restart: done=$DONE next=$NEXT remain=$REMAIN -> n28_records_r$N"
    cd "$SRC" && OPCG_LOG_SILENT=1 PYTHONPATH=tests nohup python tests/scripts/n_record_gen.py \
      --neff-net /root/gen_net.npz --games "$REMAIN" --seed-base "$NEXT" --sims 128 \
      --workers 4 --shard-games 10 --dump-v2 --out "n28_records_r$N" \
      > "/root/n28_gen_r$N.log" 2>&1 &
    sleep 20
  fi

  push_wip "$DONE"
  sleep 900
done
