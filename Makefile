# テスト・監査コマンドの正本。CLAUDE.md / README 系はここを参照する
# （生コマンドをコピーしない）。変更したらここだけ直せばよい。
#
# CI（GitHub Actions）は無い（2026-07-11 廃止・ローカル品質ゲートと二重実行だったため）。
# `make test` がマージ前の唯一の確認手段。詳細は CLAUDE.md。
#
# **エンジンは Rust**（`rust/opcg_engine`・PyO3 wheel）。Python エンジンは
# `legacy/python_engine/` へ退避し、テストゲートの対象外になった（2026-09-07・第 2 段
# `rs-archive-cutover`・`docs/rust_engine_plan.md` §16.3）。よってゲートは 1 本:
#
#   make test = cargo test（Rust 単体 381 本）＋ Rust 裏付けの pytest 集合
#               （golden 2 種・API／契約・パーサ・ツール・loop／train）
#
# golden 2 種（`tests/fixtures/rs_goldens/`）がゲームプレイ退行の一次防衛線:
#   監査 golden … カード×トリガー 3,386 件を汎用盤面で発動→既定解決した各段の sha1
#   再生 golden … 実対局 200 局（random 150・a1 50）の盤面／合法手／イベントの sha1
# 作り直しは `make golden-audit`／`make golden-replay`（**挙動を意図的に変えたときだけ**）。
# 2026-09-07 から **golden の正本は Rust**＝作り直しも Rust だけで回る
# （`tests/scripts/rs_golden_make.py`）。差分は必ずレビューする。
#
# 凍結した Python エンジンのテスト（旧 `make test-legacy`・1,786 本）は
# **tag `py-engine-final` を checkout して回す**（手順は docs/TEST_SPEC.md）。
# lint は任意（CI 無し・必須ゲートではない）。

.PHONY: test test-fast test-slow audit-cross lint golden-audit golden-replay

# push 前の必須ゲート。cargo test（Rust 単体テスト）＋ Rust 裏付けの pytest 集合。
test: rust-test
	OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -n auto -m "not slow and not legacy" -p no:cacheprovider

# 開発中のイテレーション用（基盤健全性 `cpu_infra` を除外）。push 前ゲートの代替ではない。
test-fast:
	OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -n auto -m "not slow and not cpu_infra" -p no:cacheprovider

test-slow:
	OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -m slow -p no:cacheprovider

# golden の作り直し（**挙動を意図的に変えた場合のみ**・`docs/rust_engine_plan.md` §16.3）。
# どちらも Rust だけで回る（Python エンジンは要らない）。
golden-audit:
	OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_golden_make.py audit

# random 150 局（seed 帯 5000000〜）＋ a1 50 局（5000150〜）を打ち直す。
# a1 帯は旧 L1 帯の置き換え（L1 廃止・§16.3-8）。
golden-replay:
	OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_golden_make.py replay \
	  --policy random --games 150 --seed-base 5000000
	OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_golden_make.py replay \
	  --policy a1 --games 50 --seed-base 5000150

# 交差対面の実プレイ監査（エンジン/パーサを変更したときに push 前へ追加する）。
# ミラー（同一リーダー同士）では一度も通らない経路を実プレイに乗せる＝ここでしか出ない欠陥がある
# （2026-08-16: 3欠陥をこれで検出。詳細は docs/reports/void_root_causes_20260816.md）。
# 合格条件は void（決着せず）= 0。CROSS_SEED を変えると別の対面集合を引ける。
CROSS ?= 120
CROSS_SEED ?= 0
audit-cross:
	OPCG_LOG_SILENT=1 python -m opcg_sim.loop.arena_shard \
	  --candidate "" --pairs $(CROSS) --bands 1 --max-pairs $(CROSS) --workers 4 --sims 32 \
	  --leaders random --decks synth --seed-base $(CROSS_SEED) \
	  --out /tmp/audit_cross_$(CROSS_SEED).jsonl

lint:
	ruff check opcg_sim/

# --- Rust エンジン（rust/opcg_engine・docs/rust_engine_plan.md） ---------------
# Rust 側の品質ゲート。`make test` が `rust-test` を先に回す。
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
