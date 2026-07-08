#!/bin/bash
# P3 cluster-learning bootstrap. No-op unless OPCG_P3_WT and OPCG_P3_BRANCH are
# set in the environment. Each color-pool environment (yellow/red/blue/...)
# configures its own worktree path + checkpoint branch via those two env vars
# (Environment settings, not git) so this single script is safe to share
# across every color's environment without cross-contamination.
set -uo pipefail

echo '{"async": true, "asyncTimeout": 120000}'

python3 -c "import numpy" 2>/dev/null || pip install -q numpy

if [ -z "${OPCG_P3_WT:-}" ] || [ -z "${OPCG_P3_BRANCH:-}" ]; then
  exit 0
fi

REPO="${CLAUDE_PROJECT_DIR:-$(pwd)}"
WT="$OPCG_P3_WT"
BR="$OPCG_P3_BRANCH"
PIDFILE="/tmp/p3-cluster.pid"
RUNLOG="/tmp/p3-cluster-run.log"
WATCHLOG="/tmp/p3-cluster-watch.log"
WATCHSCRIPT="/tmp/p3-cluster-watch.sh"

cd "$REPO" || exit 0
rm -rf "$WT"
git worktree prune 2>/dev/null || true
git worktree add "$WT" "$BR" >/dev/null 2>&1

cat > "$WATCHSCRIPT" << EOF
#!/bin/bash
REPO="$REPO"
RUNLOG="$RUNLOG"
WATCHLOG="$WATCHLOG"
PIDFILE="$PIDFILE"
echo "\$(date -u +%Y-%m-%dT%H:%M:%SZ) watcher started (pid \$\$)" >> "\$WATCHLOG"
while true; do
  if [ -f "\$PIDFILE" ] && kill -0 "\$(cat "\$PIDFILE")" 2>/dev/null; then
    sleep 30; continue
  fi
  echo "\$(date -u +%Y-%m-%dT%H:%M:%SZ) p3_run not running -> resume" >> "\$WATCHLOG"
  cd "\$REPO" || exit 1
  OPCG_P3_WT="$WT" OPCG_P3_BRANCH="$BR" OPCG_LOG_SILENT=1 PYTHONPATH=tests nohup python tests/scripts/p3_run.py \\
    --enc-version 2 --rotate-leaders --shard-games 60 --sims 40 --workers 4 \\
    --target 100000000 --max-shards 100000000 >> "\$RUNLOG" 2>&1 &
  echo \$! > "\$PIDFILE"
  echo "\$(date -u +%Y-%m-%dT%H:%M:%SZ) launched p3_run.py pid=\$!" >> "\$WATCHLOG"
  sleep 30
done
EOF
chmod +x "$WATCHSCRIPT"
rm -f "$PIDFILE"
nohup bash "$WATCHSCRIPT" > /dev/null 2>&1 &
disown
