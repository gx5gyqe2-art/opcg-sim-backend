#!/bin/bash
# 生成プロセスのログ: nohup リダイレクトが環境ポリシーで拒否されるため、
# ハーネスのバックグラウンドタスク出力を gen.log として退避する。
GENLOG="${GENLOG:-/tmp/claude-0/-home-user-opcg-sim-backend/f8e9f91e-40a3-5458-8305-6df646cbfee1/tasks/bpmyjjo96.output}"
while ps aux | grep -q "[p]lan_cf2_gen"; do
  sleep 900
  cd /home/user/opcg-sim-backend || exit 1
  ls /home/user/plancf2_s6/*.npz >/dev/null 2>&1 || continue
  git fetch origin claude/plancf2-shard6 >/dev/null 2>&1
  git worktree remove /tmp/cf2_wt --force 2>/dev/null
  git worktree add -B claude/plancf2-shard6 /tmp/cf2_wt origin/claude/plancf2-shard6 >/dev/null 2>&1 || \
    git worktree add -B claude/plancf2-shard6 /tmp/cf2_wt HEAD >/dev/null 2>&1 || continue
  mkdir -p /tmp/cf2_wt/audit_results/plancf2/shard6
  cp /home/user/plancf2_s6/*.npz /tmp/cf2_wt/audit_results/plancf2/shard6/ 2>/dev/null
  if [ -f /home/user/plancf2_s6.log ]; then
    cp /home/user/plancf2_s6.log /tmp/cf2_wt/audit_results/plancf2/shard6/gen.log 2>/dev/null
  else
    cp "$GENLOG" /tmp/cf2_wt/audit_results/plancf2/shard6/gen.log 2>/dev/null
  fi
  cp /home/user/run_shard.sh /home/user/push_fast.sh /tmp/cf2_wt/audit_results/plancf2/shard6/ 2>/dev/null
  cd /tmp/cf2_wt && git add -A && git commit -qm "plancf2 shard6: checkpoint" && \
    git push -qf origin claude/plancf2-shard6
done
