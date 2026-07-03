"""pytest 共通セットアップ。

sys.path 設定と google.cloud スタブ注入は `_bootstrap`（tests/harness/scripts 共通）へ集約した。
本 conftest はそれを読み込み、pytest マーカーを登録する。
"""
import os
import sys

# _bootstrap（同ディレクトリ）を解決できるよう tests/ を path に載せてから読み込む。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)


def pytest_configure(config):
    """マーカー登録。`slow` = CI から除外する重テスト（手動実行前提）。
    CI は `-m "not slow"` で実行し、`-m slow` で重テストだけを手動実行できる。
    """
    config.addinivalue_line(
        "markers",
        "slow: 実行が極端に長くCIから除外する重テスト（手動実行前提・例 test_journal の parked_resume ~245s）",
    )
