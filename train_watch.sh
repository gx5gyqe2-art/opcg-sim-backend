#!/bin/bash
# Resident watchdog for the r2b training run.
# Every 15 min: verify the trainer is alive and push log + current net to the WIP branch.
SRC=/root/opcg-sim-backend
WT=/root/pushtree
BR=claude/train-r2b-wip
LOCK=/root/push.lock

log() { echo "[$(date -u +%H:%M:%S)] $*" >> /root/train_watch.log; }

push_wip() {
  exec 9>"$LOCK"; flock -w 300 9 || { log "push: lock timeout"; return 1; }
  cd "$WT" || return 1
  git checkout -q --detach res 2>/dev/null
  rm -rf n1_results train_r2b.log nrel_r2b.npz 2>/dev/null
  mkdir -p n1_results
  cp /root/nrel_r2b.npz n1_results/ 2>/dev/null
  cp /root/train_r2b.log . 2>/dev/null
  cp /root/train_watch.sh /root/train_watch.log . 2>/dev/null
  git checkout -qB "$BR"
  git add -f n1_results train_r2b.log train_watch.sh train_watch.log 2>/dev/null
  git -c user.email=g.x5gyqe2@gmail.com -c user.name=worker \
      commit -q -m "wip train r2b ($1)" 2>/dev/null
  for i in 1 2 3 4; do
    if git push -qf -u origin "$BR" 2>/dev/null; then
      log "pushed wip ($1)"; flock -u 9; return 0
    fi
    sleep $((2 ** i))
  done
  log "push FAILED -- check auth"; flock -u 9; return 1
}

while true; do
  LAST=$(grep -E "^ *ep |step" /root/train_r2b.log 2>/dev/null | tail -1)
  if pgrep -f "^python tests/scripts/n_rel_train\.py" >/dev/null; then
    log "running | ${LAST:-no progress line yet}"
    push_wip "${LAST:-running}"
  else
    if grep -q "N_REL_TRAIN_DONE" /root/train_r2b.log 2>/dev/null; then
      log "TRAIN DONE"; push_wip "done"; break
    fi
    log "trainer NOT running and no DONE marker -- died? ${LAST:-none}"
    push_wip "${LAST:-died}"
    break
  fi
  sleep 900
done
