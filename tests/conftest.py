"""
テスト共通セットアップ。

ローカル/CI 環境では google-cloud (cffi/cryptography) が利用できない場合があるため、
本番コードを変更せずに google.cloud.{storage,firestore} をスタブ化してから
プロジェクトモジュールを import 可能にする。
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


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
