#!/bin/bash
# v4学習run（docs/cpu_v4_plan.md）の生成ワーカー w4 を、コンテナ再起動後も自動で再開させるフック。
# メインチェックアウト（このリポジトリ）はブランチを変えず、パイプラインコードは
# 専用の git worktree にチェックアウトして実行する（本ブランチをパイプライン用の
# ブランチ切り替えで汚さないため）。
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo '{"async": true, "asyncTimeout": 180000}'

PIPELINE_BRANCH="claude/cpu-spec-improvements-yw91jd"
WORKTREE_DIR="/tmp/v4-w4-pipeline"
LOG_DIR="/tmp/v4-w4-logs"
mkdir -p "$LOG_DIR"

cd "$CLAUDE_PROJECT_DIR"

git fetch origin "$PIPELINE_BRANCH" >>"$LOG_DIR/hook.log" 2>&1 || true

if git worktree list | grep -q "$WORKTREE_DIR"; then
  git -C "$WORKTREE_DIR" fetch origin "$PIPELINE_BRANCH" >>"$LOG_DIR/hook.log" 2>&1 || true
  git -C "$WORKTREE_DIR" checkout -q "origin/$PIPELINE_BRANCH" >>"$LOG_DIR/hook.log" 2>&1 || true
else
  rm -rf "$WORKTREE_DIR"
  git worktree add "$WORKTREE_DIR" "origin/$PIPELINE_BRANCH" >>"$LOG_DIR/hook.log" 2>&1
fi

python3 -c "import numpy" >/dev/null 2>&1 || pip install --quiet numpy >>"$LOG_DIR/hook.log" 2>&1

# 既に生成ループが動いていれば何もしない（べき等性）。pgrep -f は env var を見ないため
# argv にブランチ名等が現れない本コマンドでは使えない。pidfile + flock で判定・排他する。
PIDFILE="$LOG_DIR/pd_gen.pid"
LOCKFILE="$LOG_DIR/pd_gen.lock"
(
  flock -x 9
  RUNNING=0
  if [ -f "$PIDFILE" ]; then
    OLD_PID="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
      RUNNING=1
    fi
  fi
  if [ "$RUNNING" = "0" ]; then
    cd "$WORKTREE_DIR"
    OPCG_P3_LEAD_SLOTS=2 OPCG_P3_EFF_DIM=116 \
    OPCG_PD_NET_BRANCH=claude/v4-net OPCG_PD_DATA_BRANCH=claude/v4-data-w4 \
    OPCG_PD_WT=/tmp/v4-w4 OPCG_LOG_SILENT=1 PYTHONPATH=tests \
    nohup python3 tests/scripts/pd_gen.py --enc-version 3 --sims 160 --games 32 --workers 4 --l1-mix 0.25 \
      >>"$LOG_DIR/pd_gen.log" 2>&1 &
    NEW_PID=$!
    disown
    echo "$NEW_PID" >"$PIDFILE"
    echo "$(date -u +%FT%TZ) pd_gen restarted pid=$NEW_PID" >>"$LOG_DIR/hook.log"
  else
    echo "$(date -u +%FT%TZ) pd_gen already running pid=$OLD_PID, skip" >>"$LOG_DIR/hook.log"
  fi
) 9>"$LOCKFILE"
