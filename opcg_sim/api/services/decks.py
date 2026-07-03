"""デッキ読込サービス（Firestore デッキドキュメント→CardInstance／ロビー用プレビュー）。

Firestore 依存はこの1モジュールに閉じる。ルート（routers.py）は `deck_svc.load_deck_mixed(...)` の
ように**このモジュール属性経由**で呼ぶ。DB 非依存にしたいテスト・スタブは
`opcg_sim.api.services.decks` の `load_deck_mixed` / `_deck_preview` を monkeypatch する
（app モジュール属性を差し替えても効かない：C-5 でルートを本モジュール経由に移行済み）。
"""
from typing import Any, Dict

from opcg_sim.src.models.models import CardInstance
from ..resources import db, card_db


def _load_deck_doc(source_str: str) -> Dict[str, Any]:
    """`db:<id>` 形式のデッキIDから Firestore のデッキドキュメント(dict)を取得する。"""
    if not source_str.startswith("db:"):
        raise ValueError(f"Unknown deck id: {source_str}")
    if not db: raise ValueError("Firestore is not initialized.")
    deck_id = source_str[3:]; doc = db.collection("decks").document(deck_id).get()
    if not doc.exists: raise ValueError(f"Deck ID not found: {deck_id}")
    return doc.to_dict()


def load_deck_mixed(source_str: str, owner_id: str):
    deck_id = source_str[3:] if source_str.startswith("db:") else source_str
    data = _load_deck_doc(source_str); leader_id = data.get("leader_id"); card_uuids = data.get("card_uuids", [])
    leader_inst = None
    if leader_id:
        master = card_db.get_card(leader_id)
        if master: leader_inst = CardInstance(master, owner_id)
    cards_inst = [CardInstance(m, owner_id) for cid in card_uuids if (m := card_db.get_card(cid))]
    return leader_inst, cards_inst


def _deck_preview(deck_id: str, owner_id: str) -> Dict[str, Any]:
    """ロビー表示用にデッキのリーダー情報のみ抽出する。

    デッキ全カードはロードせず、Firestore のメタデータ(leader_id)から
    リーダー1枚だけを解決する（一覧表示のたびに50枚パースしない）。
    """
    try:
        leader_id = _load_deck_doc(deck_id).get("leader_id")
        master = card_db.get_card(leader_id) if leader_id else None
        if master:
            return {"leader_id": master.card_id, "leader_name": master.name}
    except Exception as e:
        pass
    return None
