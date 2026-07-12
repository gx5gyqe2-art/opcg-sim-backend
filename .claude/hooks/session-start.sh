#!/bin/bash
# v4 w4生成ワーカーは2026-07-12にユーザ指示で停止済み。
# 自動再起動しないよう、このフックは無効化してある（no-op）。
# 再開する場合は下のbash呼び出しのコメントアウトを解除すること:
#   bash "$CLAUDE_PROJECT_DIR/.claude/hooks/pd_gen_watchdog.sh"
exit 0
