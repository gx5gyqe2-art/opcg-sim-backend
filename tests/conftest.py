"""pytest 共通セットアップ。

sys.path 設定と google.cloud スタブ注入は `_bootstrap`（tests/harness/scripts 共通）へ集約した。
本 conftest はそれを読み込み、pytest マーカーを登録する。
"""
import os
import sys

import pytest

# _bootstrap（同ディレクトリ）を解決できるよう tests/ を path に載せてから読み込む。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)


def pytest_configure(config):
    """マーカー登録。`slow` = 極端に重くルーチンから除外する重テスト（手動実行前提）。
    `cpu_infra` = 探索/自己対戦/学習パイプラインの内部機構の健全性のみを見るテスト
    （ゲームプレイの正しさ自体は必須/標準テストが別途担保。分類基準は docs/TEST_SPEC.md
    §重要度分類）。`legacy` = Python エンジンを直に叩くテスト。**本体は
    `legacy/python_engine/tests/` へ退避した**（2026-09-07・計画 §16.3-6）ので、この下に
    残るのは 1 本だけ（`test_api_rs_errors.py` は legacy 側）。マーカーは
    「まだ Python エンジンを触るものがあれば `make test` から外す」という保険として残す。
    `make test` は `-m "not slow and not legacy"`（cargo test を併走）、`make test-fast` は
    `-m "not slow and not cpu_infra"`。legacy 側は tag `py-engine-final` を checkout して
    回す（`docs/TEST_SPEC.md`）。
    """
    config.addinivalue_line(
        "markers",
        "slow: 実行が極端に長くルーチンから除外する重テスト（手動実行前提・例 test_journal の parked_resume ~245s）",
    )
    config.addinivalue_line(
        "markers",
        "cpu_infra: 探索/自己対戦/学習パイプラインの内部機構の健全性のみを見る基盤健全性テスト（make test-fast で除外）",
    )
    config.addinivalue_line(
        "markers",
        "legacy: Python エンジン直叩きのテスト（Rust 化後は golden 2 本が一次防衛線・"
        "make test-legacy でのみ実行・docs/rust_engine_plan.md §16.1）",
    )


@pytest.fixture(autouse=True)
def _theory_option_off():
    """**`ν` の潜在価値（T46・分布で足す項）はテストでは既定で切る**。

    多くのテストは攻撃項の**閉じた代数**（`lead × R`・リーダー未満は 0 等）を固定している。
    潜在価値は同梱の分布（`tests/fixtures/opp_boards.json`）に依る実測の項なので、
    その代数とは別に T46 のテストが**明示的に入れて**固定する。
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
        import theory_order as _T
    except Exception:
        yield
        return
    before = _T.OPTION_MODE
    _T.set_option_mode("off")
    # **`w(状態)`（T49）もテストでは平均の傾き（`flat`・`κ = 1`）に固定する**——橋の符号や
    # 「助言どおりなら 0」の算術は `κ` を掛けても変わらないが、値を固定したテストは動く。
    # 時計の形そのものは T49 のテストが**明示的に `clock` にして**固定する。
    before_w = _T.W_MODE
    _T.set_w_mode("flat")
    try:
        yield
    finally:
        _T.set_option_mode(before)
        _T.set_w_mode(before_w)
