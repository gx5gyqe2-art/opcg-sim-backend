"""held-out 実デッキ集合（heldout_decks.json）の凍結検証。

汎化ゲートの前提が壊れていないことを CI で保証する: 実在ID・50枚・4枚制限・リーダー色一致・
実ゲームに投入可能。リスト内容の変更検知（凍結ハッシュ）も行う＝うっかり書き換えたら CI が落ちる。
"""
import hashlib
import json

import conftest  # noqa: F401
import heldout_decks as HD
from cpu_selfplay import _load_db
from opcg_sim.src.core.gamestate import GameManager, Player

# 凍結ハッシュ（2026-07-01 freeze・変更する場合は新しい日付で freeze し直すこと）
FROZEN_SHA256 = "b3c026d7d58511a0cbf94b12bd0af98cd615a26e8ec91824618fa051f802d5dd"


def test_frozen_content_unchanged():
    with open(HD.PATH, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    assert h == FROZEN_SHA256, (
        "heldout_decks.json が凍結時から変更されている。"
        "変更が意図的なら新しい日付で freeze し直しハッシュを更新する")


def test_decks_are_legal():
    db = _load_db()
    spec = HD.load_spec()
    assert len(spec["decks"]) >= 3
    for d in spec["decks"]:
        lm = db.get_card(d["leader"])
        assert lm is not None and lm.type.name == "LEADER", d["id"]
        lcol = {getattr(x, "value", x) for x in (lm.colors or [])}
        total = 0
        for cid, n in d["cards"].items():
            m = db.get_card(cid)
            assert m is not None, f"{d['id']}: {cid} が DB に無い"
            assert 1 <= int(n) <= 4, f"{d['id']}: {cid} x{n}"
            ccol = {getattr(x, "value", x) for x in (m.colors or [])}
            assert lcol & ccol, f"{d['id']}: {cid} 色不一致"
            total += int(n)
        assert total == 50, f"{d['id']}: {total}枚"


def test_decks_playable_in_engine():
    db = _load_db()
    ids = HD.deck_ids()
    l1, c1 = HD.build(db, ids[0], "p1")
    l2, c2 = HD.build(db, ids[1], "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    pa = m.pending_actor_action()
    assert pa is not None
    actor = m.p1 if m.p1.name == pa[0] else m.p2
    assert m.get_legal_actions(actor), "held-out デッキで合法手が生成できない"
