# テスト・監査コマンドの正本。CLAUDE.md / README 系はここを参照する
# （生コマンドをコピーしない）。変更したらここだけ直せばよい。
#
# CI（GitHub Actions）は無い（2026-07-11 廃止・ローカル品質ゲートと二重実行だったため）。
# `make test` がマージ前の唯一の確認手段。詳細は CLAUDE.md。
#
# Rust 化後のテストゲート 2 段化（2026-09-07・docs/rust_engine_plan.md §16.1）:
# 全カード監査（旧 test_full_card_audit／test_full_card_baseline・Python エンジンで全カードを
# 発動）と実対局の再生は、Rust エンジンだけで回る golden 照合（tests/test_rs_golden_audit.py／
# tests/test_rs_golden_replay.py・tests/fixtures/rs_goldens/）に置き換わった。`make test` は
# 「cargo test（Rust 単体テスト）＋ golden 2 本を含む Rust 裏付けの pytest 集合」＝
# `-m "not slow and not legacy"` になり、Python エンジン（opcg_sim/src/core・effects・learned）を
# 直に叩くテスト（`legacy` マーカー）は対象から外れる。**Python エンジンを変更したときは
# 引き続き `make test-legacy` も通す**（golden はあくまで「Rust が Python と一致した時点」の
# 記録であり、Python 側の挙動変更そのものを検出しない）。golden の作り直しは
# `make golden-audit`／`make golden-replay`（**挙動を意図的に変えたときだけ**）。
#
# audit/regen-baseline は診断・ベースライン更新用の単体コマンド（Python エンジン直叩き・
# `make test-legacy` の一部として残す）。lint は任意（CI 無し・必須ゲートではない）。
#
# test-fast は開発中のイテレーション用（cpu_infra＝探索/自己対戦/学習パイプラインの内部機構の
# 健全性のみを見るテストを除外。分類基準・対象は docs/TEST_SPEC.md §重要度分類）。
# push前ゲートの代替ではない＝push前は必ず test（新定義のフルスコープ）を通す。

.PHONY: test test-legacy test-fast test-slow audit audit-cross regen-baseline lint golden-audit golden-replay

# 新定義（2026-09-07）: cargo test（Rust 単体テスト）＋ Rust 裏付けの pytest 集合。
# golden 2 本（tests/test_rs_golden_audit.py／test_rs_golden_replay.py）が
# ゲームプレイ退行の一次防衛線＝Python エンジンは動かさない（3 分以内を実測・RESULT.json）。
test: rust-test
	OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -n auto -m "not slow and not legacy" -p no:cacheprovider

# 従来の全数（Rust 化前の push前ゲートと同じ・`legacy` マーカーの起点）。
# **Python エンジン（opcg_sim/src/core・effects・learned）を変更したときは、
# make test に加えてこちらも通す**（golden は Rust↔Python の一致を固定するだけで、
# Python 側の挙動変更そのものは検出しない）。
test-legacy:
	OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -n auto -m "not slow" -p no:cacheprovider

test-fast:
	OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -n auto -m "not slow and not cpu_infra" -p no:cacheprovider

test-slow:
	OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -m slow -p no:cacheprovider

# golden の作り直し（**挙動を意図的に変えた場合のみ**・docs/rust_engine_plan.md §16.1）。
# Python エンジンで記録し直し、その場で Rust の golden_audit/replay と突き合わせて
# mismatch=0 を確認してから書く（rs_audit_replay.py／rs_diff_replay.py の --golden-out）。
golden-audit:
	OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_audit_replay.py \
	  --golden-out tests/fixtures/rs_goldens/audit.json

# random 150 局＋L1 50 局を打ち直す（seed 帯は既存の tests/fixtures/rs_goldens/replay/ と同じ
# 5000000〜・5000150〜）。一致した局だけを書く＝mismatch のある局は無言で golden から漏れる
# ことはない（rs_diff_replay.py の対応する RS_DIFF summary の mismatch を必ず確認すること）。
golden-replay:
	OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_diff_replay.py \
	  --games 150 --policy random --seed-base 5000000 --golden-out tests/fixtures/rs_goldens/replay
	OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_diff_replay.py \
	  --games 50 --policy l1 --seed-base 5000150 --golden-out tests/fixtures/rs_goldens/replay

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

# --- Rust エンジン（rust/opcg_engine・docs/rust_engine_plan.md） ---------------
# Rust 側の品質ゲート。`make test`（Python）とは独立で、**Rust を触ったときに通す**。
# cargo は extension-module を外して回す（有効なままだとテストバイナリが libpython の
# シンボルを解決できずリンクに失敗する。既定 feature の説明は crate の Cargo.toml）。
# rust-develop は現在の Python 環境へ拡張を入れる（maturin develop は venv/conda を要求するため、
# 無い環境では build → pip install へ自動で退避する。どちらでも `import opcg_engine` が通る）。
RUST_DIR = rust/opcg_engine
.PHONY: rust rust-test rust-clippy rust-develop rust-wheel

rust: rust-test rust-clippy rust-develop

rust-test:
	cd $(RUST_DIR) && cargo test --no-default-features

rust-clippy:
	cd $(RUST_DIR) && cargo clippy --no-default-features --all-targets -- -D warnings

rust-develop:
	@if [ -n "$$VIRTUAL_ENV" ] || [ -n "$$CONDA_PREFIX" ]; then \
	  cd $(RUST_DIR) && maturin develop --release; \
	else \
	  echo "[rust-develop] venv 無し: maturin build + pip install で代替する"; \
	  cd $(RUST_DIR) && maturin build --release --compatibility linux --out target/wheels \
	    && pip install --force-reinstall --no-deps target/wheels/*.whl; \
	fi

# 配布用 wheel（Dockerfile のビルド段と同じコマンド）。
rust-wheel:
	cd $(RUST_DIR) && maturin build --release --compatibility linux --out target/wheels
