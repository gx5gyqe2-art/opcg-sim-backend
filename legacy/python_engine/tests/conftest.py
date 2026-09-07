"""凍結した旧テスト群の conftest（`tests/conftest.py` の legacy 版）。"""
import _bootstrap  # noqa: F401  (sys.path 設定＋google スタブ)

import legacy.python_engine as _LE

# 符号化のうちエンジン実測が要る 3 か所を差す（`opcg_sim.learned.hooks`）。
_LE.install_hooks()


def pytest_configure(config):
    """マーカー登録（`tests/conftest.py` と同じ 3 つ）。"""
    for name, desc in (
        ("slow", "実行が極端に長くルーチンから除外する重テスト（手動実行前提）"),
        ("cpu_infra", "探索/自己対戦/学習パイプラインの内部機構の健全性のみを見る基盤健全性テスト"),
        ("legacy", "Python エンジン直叩きのテスト（凍結・tag py-engine-final で回す）"),
    ):
        config.addinivalue_line("markers", f"{name}: {desc}")
