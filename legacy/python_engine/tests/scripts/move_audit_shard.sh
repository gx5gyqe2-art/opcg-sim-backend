#!/usr/bin/env bash
# 手の監査シャード実行（セッション分割用・2026-08-17）。
#
# 使い方: tests/scripts/move_audit_shard.sh <シャード番号 0..N> [段1の局数] [カテゴリ毎の点数]
#
# なぜ seed 帯で割るか: 各セッションは自分の帯で段1（容疑者抽出）→段2（regret 実測）を
# **完結**できる＝ワーカー間で入力ファイルを受け渡さなくてよい（決定論なので帯が違えば
# 判断点も重複しない）。結果はチャットへ貼り戻す運用（作業台帳は環境の巻き戻しで消える）。
set -euo pipefail
cd "$(dirname "$0")/../.."

SHARD="${1:?シャード番号を指定してください（0,1,2,...）}"
GAMES="${2:-8}"
PER_CAT="${3:-2}"
SEED_BASE=$((500000 + SHARD * 1000))
OUT_DIR="${OUT_DIR:-/tmp/move_audit_shard${SHARD}}"
mkdir -p "$OUT_DIR"

echo "=== シャード ${SHARD}: seed 帯 ${SEED_BASE}〜 / 段1 ${GAMES}局 / カテゴリ毎 ${PER_CAT}点 ==="

# 段1: 本番仕様（sims=160・ランダム対面×生成デッキ）で容疑者を抽出する。
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/move_audit.py \
  --games "$GAMES" --seed-base "$SEED_BASE" --sims 160 --workers 4 \
  --leaders random --decks synth \
  --out "$OUT_DIR/suspects.jsonl" 2>&1 | tee "$OUT_DIR/stage1.log"

# 段2: カテゴリごとに上位 PER_CAT 点だけ regret を実測（層化抽出＝カテゴリ別平均を作る）。
# 逐次書き出しなので、打ち切られても測れた分は残る。
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/move_regret.py \
  --suspects "$OUT_DIR/suspects.jsonl" \
  --worlds 6 --max-options 3 --per-category "$PER_CAT" --max-suspects 12 \
  --workers 4 --sims 160 \
  --out "$OUT_DIR/regret.jsonl" 2>&1 | tee "$OUT_DIR/stage2.log"

echo "=== シャード ${SHARD} 完了: $OUT_DIR ==="
