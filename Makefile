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

.PHONY: test test-fast test-slow audit regen-baseline lint

test:
	OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -n auto -m "not slow" -p no:cacheprovider

test-fast:
	OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -n auto -m "not slow and not cpu_infra" -p no:cacheprovider

test-slow:
	OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -m slow -p no:cacheprovider

audit:
	OPCG_LOG_SILENT=1 python tests/harness/full_card_audit.py --show

regen-baseline:
	OPCG_LOG_SILENT=1 python tests/harness/full_card_audit.py --regen

lint:
	ruff check opcg_sim/
