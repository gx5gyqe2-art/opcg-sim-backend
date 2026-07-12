#!/bin/bash
# v4学習run（docs/cpu_v4_plan.md）の生成ワーカー w4 を、コンテナ再起動後も自動で再開させるフック。
# 実際の生死判定・再起動ロジックは pd_gen_watchdog.sh に集約（10分ごとのセッション自前
# チェックインからも同じスクリプトを呼ぶため、ロジックの二重化を避けている）。
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo '{"async": true, "asyncTimeout": 180000}'

bash "$CLAUDE_PROJECT_DIR/.claude/hooks/pd_gen_watchdog.sh"
