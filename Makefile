# テスト・監査コマンドの正本。CLAUDE.md / README 系はここを参照する
# （生コマンドをコピーしない）。変更したらここだけ直せばよい。
#
# CI（GitHub Actions）は無い（2026-07-11 廃止・ローカル品質ゲートと二重実行だったため）。
# `make test` がマージ前の唯一の確認手段。詳細は CLAUDE.md。
#
# 構造監査（full_card_audit の EXCEPTION/CARD_LOSS/TEMP_LEAK = 0）は
# tests/test_full_card_audit.py が pytest 内で実行済み（`make test` に含まれる）。
# audit/regen-baseline は診断・ベースライン更新用の単体コマンド。lint は任意（CI 無し・必須ゲートではない）。
#
# test-fast は開発中のイテレーション用（cpu_infra＝探索/自己対戦/学習パイプラインの内部機構の
# 健全性のみを見るテストを除外。分類基準・対象は docs/TEST_SPEC.md §重要度分類）。
# push前ゲートの代替ではない＝push前は必ず test（フルスコープ）を通す。

.PHONY: test test-fast test-slow audit audit-cross regen-baseline lint

test:
	OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -n auto -m "not slow" -p no:cacheprovider

test-fast:
	OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -n auto -m "not slow and not cpu_infra" -p no:cacheprovider

test-slow:
	OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -m slow -p no:cacheprovider

audit:
	OPCG_LOG_SILENT=1 python tests/harness/full_card_audit.py --show

# 交差対面の実プレイ監査（エンジン/パーサを変更したときに push 前へ追加する。約10分）。
# ミラー（同一リーダー同士）では一度も通らない経路を実プレイに乗せる＝ここでしか出ない欠陥がある
# （2026-08-16: 3欠陥をこれで検出。詳細は docs/reports/void_root_causes_20260816.md）。
# 合格条件は hang / timeout / error = 0。CROSS_SEED を変えると別の対面集合を引ける
# （既定 0 は再現性のため固定。回帰の確認は同じ seed、探索を広げたいときは変える）。
CROSS ?= 120
CROSS_SEED ?= 0
audit-cross:
	OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/deck_synth_audit.py \
	  --cross $(CROSS) --cross-seed $(CROSS_SEED) --sims 8 --max-steps 700 \
	  --workers 4 --timeout 300

regen-baseline:
	OPCG_LOG_SILENT=1 python tests/harness/full_card_audit.py --regen

lint:
	ruff check opcg_sim/
