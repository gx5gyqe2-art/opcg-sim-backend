"""テスト/ハーネス/スクリプト共通のブートストラップ（sys.path 設定＋google スタブ）。

従来 `tests/conftest.py` が一手に担っていたブート処理を単一モジュールへ集約する。
`tests/harness/` と `tests/scripts/` へ移設した基盤ライブラリ・実験スクリプトは、
先頭で本モジュールを import してパス解決と google スタブを確定する（pytest 経由・単体実行の双方）。

配置規約（docs/refactoring_tests_and_errors.md）:
  - `tests/test_*.py`      : pytest テスト（本モジュールは conftest 経由で読み込む）
  - `tests/harness/`       : テストが import する基盤ライブラリ
  - `tests/scripts/`       : 単体実行の実験/計測/監査 CLI
  - `tests/fixtures/`      : テストが読み込むデータ資産
"""
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))   # tests/
ROOT = os.path.dirname(_HERE)                         # リポジトリルート
TESTS_DIR = _HERE
HARNESS_DIR = os.path.join(_HERE, "harness")
SCRIPTS_DIR = os.path.join(_HERE, "scripts")
DATA_DIR = os.path.join(ROOT, "opcg_sim", "data")
FIXTURES_DIR = os.path.join(_HERE, "fixtures")

# ルート（opcg_sim パッケージ）・tests（golden 等）・harness/scripts（ベア import）を解決可能にする。
for _p in (ROOT, TESTS_DIR, HARNESS_DIR, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _install_google_stub():
    """google.cloud が import できない環境向けの最小スタブを注入する。

    名前空間パッケージ google.cloud 自体は import できても、深い import
    (storage/firestore) が cffi/cryptography 不在で失敗する場合がある。
    pyo3 の PanicException は BaseException なので BaseException で捕捉する。
    """
    try:
        from google.cloud import storage  # noqa: F401
        from google.cloud import firestore  # noqa: F401
        return  # 本物が使えるなら何もしない
    except BaseException:
        pass

    class _FakeClient:
        def __init__(self, *a, **k):
            raise RuntimeError("google.cloud stubbed for tests")

    class _Query:
        ASCENDING = "ASCENDING"
        DESCENDING = "DESCENDING"

    google = types.ModuleType("google")
    cloud = types.ModuleType("google.cloud")
    storage = types.ModuleType("google.cloud.storage")
    firestore = types.ModuleType("google.cloud.firestore")
    storage.Client = _FakeClient
    firestore.Client = _FakeClient
    # 本番コードが参照する定数・列挙の最小スタブ（テストで fake db を差し込めるように）。
    firestore.SERVER_TIMESTAMP = "SERVER_TIMESTAMP"
    firestore.Query = _Query
    cloud.storage = storage
    cloud.firestore = firestore
    google.cloud = cloud
    sys.modules.setdefault("google", google)
    sys.modules["google.cloud"] = cloud
    sys.modules["google.cloud.storage"] = storage
    sys.modules["google.cloud.firestore"] = firestore


_install_google_stub()
