# テスト・監査コマンドの正本。CLAUDE.md / README 系はここを参照する
# （生コマンドをコピーしない）。変更したらここだけ直せばよい。
#
# CI（GitHub Actions）は無い（2026-07-11 廃止・ローカル品質ゲートと二重実行だったため）。
# `make test` がマージ前の唯一の確認手段。詳細は CLAUDE.md。
#
# 構造監査（full_card_audit の EXCEPTION/CARD_LOSS/TEMP_LEAK = 0）は
# tests/test_full_card_audit.py が pytest 内で実行済み（`make test` に含まれる）。
# audit/regen-baseline は診断・ベースライン更新用の単体コマンド。lint は任意（CI 無し・必須ゲートではない）。

.PHONY: test test-slow audit regen-baseline lint

test:
	OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -n auto -m "not slow" -p no:cacheprovider

test-slow:
	OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -m slow -p no:cacheprovider

audit:
	OPCG_LOG_SILENT=1 python tests/harness/full_card_audit.py --show

regen-baseline:
	OPCG_LOG_SILENT=1 python tests/harness/full_card_audit.py --regen

lint:
	ruff check opcg_sim/
