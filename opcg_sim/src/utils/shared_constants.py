"""共有定数（shared_constants.json）の単一ローダ。

フロント／バックエンドで共有する API キー・アクション種別の**正本**を読み込む。
従来 `models.py` / `api/schemas.py` / `api/app.py` に3実装が重複していたのを本モジュールへ一本化する。

本モジュールは**葉**（`models`/`enums` 等に依存しない）ため、`models.py` からの import でも
循環しない。`utils/__init__.py` は空なので、本 submodule を直接 import しても
`utils/loader.py`（models 依存）を巻き込まない。
"""
import os
import json
import hashlib
import logging

logger = logging.getLogger("opcg.const")

_HERE = os.path.dirname(os.path.abspath(__file__))
# 探索パス: リポジトリルート（開発/CI）と Docker WORKDIR(/app)。
_CANDIDATES = [
    os.path.abspath(os.path.join(_HERE, "..", "..", "..", "shared_constants.json")),
    "/app/shared_constants.json",
]

# 読込失敗時のフォールバック（最小限）。従来 models.py がインラインで持っていた辞書を集約する。
# 参照側（schemas/app）は `CONST.get(..., 既定値)` で防御しているため、これは models 用の保険。
FALLBACK_CONSTANTS = {
    "CARD_PROPERTIES": {
        "UUID": "uuid",
        "CARD_ID": "card_id",
        "NAME": "name",
        "POWER": "power",
        "COUNTER": "counter",
        "ATTRIBUTE": "attribute",
        "ATTACHED_DON": "attached_don",
        "IS_REST": "is_rest",
        "OWNER_ID": "owner_id",
    }
}


def load_shared_constants() -> dict:
    """shared_constants.json を読み込んで返す。

    見つからない/壊れている場合は空 dict を返す（従来の3実装と同一挙動＝呼び出し側の
    フォールバック／既定値解決を壊さない）。失敗は従来の沈黙をやめ warning でログする。
    """
    for path in _CANDIDATES:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                logger.warning("shared_constants.json の読込に失敗: %s", path, exc_info=True)
                continue
    logger.warning("shared_constants.json が見つかりません（探索: %s）", _CANDIDATES)
    return {}


def constants_hash() -> str:
    """現在ロードされる定数の内容ハッシュ（sha256 先頭12桁）。

    `/health` の契約照合用。フロントが埋め込みハッシュと突き合わせて、
    定数の乖離（同期漏れ deploy）を検出できるようにする。
    """
    data = load_shared_constants()
    canon = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()[:12]
