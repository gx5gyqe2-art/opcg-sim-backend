#!/bin/bash
# v4 w4生成ワーカーの死活監視・自動再起動（べき等・pidfile+flock排他）。
# SessionStart hookと、セッション自前の10分ごとのチェックインの両方から呼ばれる共通実装。
# コンテナ再起動を跨いでも判定できるよう、再起動回数はgitにコミットして永続化する
# （/tmpはコンテナ回収で消えるため使えない）。繰り返し落ちる場合はメモリ不足を疑い、
# --workers を 4→3 へ自動的に落として安定性を優先する。
set -uo pipefail

PIPELINE_BRANCH="claude/cpu-spec-improvements-yw91jd"
DEV_BRANCH="claude/v4-learning-worker-w4-cd7gna"
WORKTREE_DIR="/tmp/v4-w4-pipeline"
LOG_DIR="/tmp/v4-w4-logs"
PIDFILE="$LOG_DIR/pd_gen.pid"
LOCKFILE="$LOG_DIR/pd_gen.lock"
RESTART_COUNT_FILE="$CLAUDE_PROJECT_DIR/.claude/hooks/state/restart_count"

mkdir -p "$LOG_DIR" "$(dirname "$RESTART_COUNT_FILE")"

cd "$CLAUDE_PROJECT_DIR"
git pull --ff-only origin "$DEV_BRANCH" >>"$LOG_DIR/hook.log" 2>&1 || true

{
  echo "$(date -u +%FT%TZ) --- watchdog check ---"
  free -m
} >>"$LOG_DIR/hook.log" 2>&1 || true

(
  flock -x 9
  RUNNING=0
  OLD_PID=""
  if [ -f "$PIDFILE" ]; then
    OLD_PID="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
      RUNNING=1
    fi
  fi

  if [ "$RUNNING" = "1" ]; then
    echo "$(date -u +%FT%TZ) pd_gen already running pid=$OLD_PID, skip" >>"$LOG_DIR/hook.log"
    exit 0
  fi

  COUNT=0
  [ -f "$RESTART_COUNT_FILE" ] && COUNT="$(cat "$RESTART_COUNT_FILE" 2>/dev/null || echo 0)"
  COUNT=$((COUNT + 1))
  echo "$COUNT" > "$RESTART_COUNT_FILE"

  WORKERS=4
  if [ "$COUNT" -ge 2 ]; then
    WORKERS=3
  fi
  echo "$(date -u +%FT%TZ) pd_gen dead(pid=$OLD_PID), restart_count=$COUNT -> workers=$WORKERS" >>"$LOG_DIR/hook.log"

  cd "$CLAUDE_PROJECT_DIR"
  git add "$RESTART_COUNT_FILE" >>"$LOG_DIR/hook.log" 2>&1
  git commit -m "v4 w4 watchdog: restart_count=$COUNT (workers=$WORKERS)" >>"$LOG_DIR/hook.log" 2>&1 || true
  git push origin "$DEV_BRANCH" >>"$LOG_DIR/hook.log" 2>&1 || true

  git fetch origin "$PIPELINE_BRANCH" >>"$LOG_DIR/hook.log" 2>&1 || true
  if git worktree list | grep -q "$WORKTREE_DIR"; then
    git -C "$WORKTREE_DIR" fetch origin "$PIPELINE_BRANCH" >>"$LOG_DIR/hook.log" 2>&1 || true
    git -C "$WORKTREE_DIR" checkout -q "origin/$PIPELINE_BRANCH" >>"$LOG_DIR/hook.log" 2>&1 || true
  else
    rm -rf "$WORKTREE_DIR"
    git worktree add "$WORKTREE_DIR" "origin/$PIPELINE_BRANCH" >>"$LOG_DIR/hook.log" 2>&1
  fi

  python3 -c "import numpy" >/dev/null 2>&1 || pip install --quiet numpy >>"$LOG_DIR/hook.log" 2>&1

  cd "$WORKTREE_DIR"
  OPCG_P3_LEAD_SLOTS=2 OPCG_P3_EFF_DIM=116 \
  OPCG_PD_NET_BRANCH=claude/v4-net OPCG_PD_DATA_BRANCH=claude/v4-data-w4 \
  OPCG_PD_WT=/tmp/v4-w4 OPCG_LOG_SILENT=1 PYTHONPATH=tests \
  nohup python3 tests/scripts/pd_gen.py --enc-version 3 --sims 160 --games 32 --workers "$WORKERS" --l1-mix 0.25 \
    >>"$LOG_DIR/pd_gen.log" 2>&1 &
  NEW_PID=$!
  disown
  echo "$NEW_PID" >"$PIDFILE"
  echo "$(date -u +%FT%TZ) pd_gen restarted pid=$NEW_PID workers=$WORKERS" >>"$LOG_DIR/hook.log"
) 9>"$LOCKFILE"
