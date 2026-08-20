#!/bin/bash
while ps aux | grep -q "[p]lan_cf2_gen"; do
  sleep 900
  cd /home/user/opcg-sim-backend || exit 1
  ls /home/user/plancf2_s1/*.npz >/dev/null 2>&1 || continue
  git fetch origin claude/plancf2-shard1 >/dev/null 2>&1
  git worktree remove /tmp/cf2_wt --force 2>/dev/null
  git worktree add -B claude/plancf2-shard1 /tmp/cf2_wt origin/claude/plancf2-shard1 >/dev/null 2>&1 || \
    git worktree add -B claude/plancf2-shard1 /tmp/cf2_wt HEAD >/dev/null 2>&1 || continue
  mkdir -p /tmp/cf2_wt/audit_results/plancf2/shard1
  cp /home/user/plancf2_s1/*.npz /tmp/cf2_wt/audit_results/plancf2/shard1/ 2>/dev/null
  cp /home/user/plancf2_s1.log /tmp/cf2_wt/audit_results/plancf2/shard1/gen.log 2>/dev/null
  cp /home/user/run_shard.sh /home/user/push_fast.sh /tmp/cf2_wt/audit_results/plancf2/shard1/ 2>/dev/null
  cd /tmp/cf2_wt && git add -A && git commit -qm "plancf2 shard1: checkpoint" && \
    git push -qf origin claude/plancf2-shard1
done
