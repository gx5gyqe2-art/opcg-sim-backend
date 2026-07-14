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
    """マーカー登録。`slow` = 極端に重くルーチンから除外する重テスト（手動実行前提）。
    `cpu_infra` = 探索/自己対戦/学習パイプラインの内部機構の健全性のみを見るテスト
    （ゲームプレイの正しさ自体は必須/標準テストが別途担保。分類基準は docs/TEST_SPEC.md
    §重要度分類）。`make test` は `-m "not slow"`、`make test-fast` は
    `-m "not slow and not cpu_infra"` で実行する。
    """
    config.addinivalue_line(
        "markers",
        "slow: 実行が極端に長くルーチンから除外する重テスト（手動実行前提・例 test_journal の parked_resume ~245s）",
    )
    config.addinivalue_line(
        "markers",
        "cpu_infra: 探索/自己対戦/学習パイプラインの内部機構の健全性のみを見る基盤健全性テスト（make test-fast で除外）",
    )
