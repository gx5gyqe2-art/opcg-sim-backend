"""プロセス常駐リソース（Firestore クライアント／カードDB／ETag）。

従来 `app.py` が直接生成していた外部リソースを集約する。初期化失敗の沈黙（裸 except）を
`logging.warning` へ改め、`db=None` での継続挙動は不変に保つ（デッキ CRUD が無効になるだけ）。
"""
import logging

from opcg_sim.src.utils.loader import CardLoader
from .config import CARD_DB_PATH

try:
    from google.cloud import firestore
except Exception:
    firestore = None

logger = logging.getLogger("opcg.api")


def _init_firestore():
    """Firestore クライアントを生成する。未導入/失敗時は None を返し警告ログを残す（従来は沈黙）。"""
    if firestore is None:
        logger.warning("google-cloud-firestore 未導入: デッキ CRUD は無効（db=None で継続）")
        return None
    try:
        return firestore.Client()
    except Exception:
        logger.warning("Firestore 初期化失敗: デッキ CRUD は無効（db=None で継続）", exc_info=True)
        return None


db = _init_firestore()

# カードDB。ビルド時のパース済みキャッシュがあれば採用し、コールドスタートの全件パース(~1.8s)を回避。
card_db = CardLoader(CARD_DB_PATH)
card_db.load()
card_db.load_cache()

# /api/cards の条件付きGET(ETag)用。カードDBの内容が変わると変化する。
CARDS_ETAG = f'"{card_db.db_hash()}"'


def materialize_all_cards() -> None:
    """遅延パースのカードを全件実体化する（従来 app.py に3回コピペされていた処理を集約）。"""
    if len(card_db.cards) < len(card_db.raw_db):
        for card_id in card_db.raw_db.keys():
            card_db.get_card(card_id)
