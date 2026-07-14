"""EFFECT イベントの行き先（dest）記録の検証。

eventLog の MOVE_CARD が行き先を持たないと、フロントで効果の意味が読めない
（実対局で OP16-119 のライフ追加が素の MOVE_CARD 表示になり「発動していない」ように
見えた誤解の直接原因）。移動系アクションの EFFECT イベントに dest（"LIFE" 等）が
additive に載ることを、実カード OP16-119 の【登場時】（LOOK→1枚までライフへ→残り底）で固定する。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_effect_event_dest.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401

from engine_helpers import make_game, make_instance, make_master
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.utils.loader import CardLoader


def _teach_life_add_game():
    """OP16-119 の登場時効果を発動し、ライフ追加の選択対話まで進めた盤面を返す。"""
    db = CardLoader("opcg_sim/data/opcg_cards.json")
    db.load()
    teach = db.get_card("OP16-119")
    onplay = [ab for ab in teach.abilities if ab.trigger.name == "ON_PLAY"][0]
    gm, p1, _ = make_game()
    for cid in ["OP15-105", "OP06-104", "EB03-053"]:
        p1.deck.insert(0, make_instance(db.get_card(cid), owner=p1.name))
    for i in range(4):
        p1.deck.append(make_instance(make_master(card_id=f"D-{i}", cost=i + 1), owner=p1.name))
    src = make_instance(teach, owner=p1.name)
    p1.field.append(src)   # 発生源は場に無いと再開対話が silent drop される
    gm.resolve_ability(p1, onplay, source_card=src)
    return gm, p1


def test_life_add_move_card_event_carries_dest():
    """OP16-119 のライフ追加（MOVE_CARD dest=LIFE）が eventLog に dest 付きで残る。"""
    gm, p1 = _teach_life_add_game()
    pend = gm.get_pending_request()
    assert pend and pend.get("action") == "SEARCH_AND_SELECT"
    uid = [u for u in pend["selectable_uuids"] if cpu_ai._card_label(gm, u) == "EB03-053"][0]
    payload = gm.default_interaction_payload(pend)
    payload["selected_uuids"] = [uid]
    gm.resolve_interaction(p1, payload)

    assert [c.master.card_id for c in p1.life] == ["EB03-053"], "ライフ追加が実行されていない"
    moves = [e for e in gm.action_events
             if e.get("type") == "EFFECT" and e.get("action") == "MOVE_CARD"]
    assert moves, "MOVE_CARD の EFFECT イベントが記録されていない"
    assert moves[-1].get("dest") == "LIFE", f"dest が載っていない: {moves[-1]}"


def test_non_move_effect_event_has_no_dest():
    """行き先を持たないアクション（LOOK 等）のイベントには dest を載せない（additive の逆保証）。"""
    gm, _ = _teach_life_add_game()
    looks = [e for e in gm.action_events
             if e.get("type") == "EFFECT" and e.get("action") == "LOOK"]
    assert looks, "LOOK イベントが記録されていない"
    assert "dest" not in looks[-1], f"dest が不要な action に載っている: {looks[-1]}"
