"""デッキ配置の並び替え/上下選択(ARRANGE_DECK)とライフ並べ替えの対話テスト。

課題2(2a/2b): 「好きな順番でデッキの上か下に置く」「ライフすべてを見て好きな順番で置く」を、
従来の現状順・デッキ下固定から、プレイヤーが順序/位置を選べる対話(ARRANGE_DECK)へ。
ヘッドレス(drain 相当の既定 payload)では従来挙動（現状順・デッキ下）を保つ。
"""
import conftest  # noqa: F401
import pytest

from engine_helpers import make_game, make_master, make_instance
from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
from opcg_sim.src.models.effect_types import GameAction, TargetQuery, Ability, Sequence, Branch, Choice
from opcg_sim.src.models.enums import ActionType, TriggerType, Zone, Player, Phase

v2 = EffectParserV2()


def _collect(node):
    if node is None:
        return
    if isinstance(node, GameAction):
        yield node
    elif isinstance(node, Sequence):
        for a in node.actions:
            yield from _collect(a)
    elif isinstance(node, Branch):
        yield from _collect(node.if_true)
        yield from _collect(node.if_false)
    elif isinstance(node, Choice):
        for o in node.options:
            yield from _collect(o)


def _deck_actions(text):
    out = []
    for ab in v2.parse_card_text(text):
        for ga in _collect(ab.effect):
            if ga.type == ActionType.DECK_BOTTOM:
                out.append(ga)
    return out


def _setup_temp(n=3):
    gm, p1, p2 = make_game()
    gm.phase = Phase.MAIN; gm.turn_player = p1; gm.opponent = p2
    src = make_instance(make_master(card_id="SRC"), owner="P1")
    p1.field.append(src)
    cards = [make_instance(make_master(card_id=c), owner="P1") for c in "ABCDE"[:n]]
    p1.temp_zone.extend(cards)
    return gm, p1, p2, src, cards


def _run(gm, p1, src, action):
    gm.resolve_ability(p1, Ability(trigger=TriggerType.ON_PLAY, effect=action), source_card=src)


# --- パーサ: status/dest_position の付与 -------------------------------------

def test_parse_arrange_choose_position():
    acts = _deck_actions("【登場時】自分のデッキの上から5枚を見て、好きな順番に並び替え、デッキの上か下に置く。")
    assert acts, "DECK_BOTTOM not parsed"
    ga = acts[-1]
    assert ga.status == "ARRANGE"
    assert ga.dest_position == "CHOOSE"


def test_parse_remaining_bottom_arrange():
    acts = _deck_actions("【登場時】自分のデッキの上から3枚を見て、コスト2のキャラ1枚を手札に加える。その後、残りを好きな順番でデッキの下に置く。")
    ga = next((a for a in acts if a.status == "ARRANGE"), None)
    assert ga is not None
    assert ga.dest_position == "BOTTOM"


# --- エンジン: ARRANGE_DECK 中断と適用 --------------------------------------

def test_arrange_deck_reorder_and_top():
    gm, p1, p2, src, cards = _setup_temp(3)
    act = GameAction(type=ActionType.DECK_BOTTOM,
                     target=TargetQuery(player=Player.SELF, zone=Zone.TEMP, select_mode="REMAINING", count=-1),
                     status="ARRANGE", dest_position="CHOOSE")
    _run(gm, p1, src, act)
    ia = gm.active_interaction
    assert ia and ia["action_type"] == "ARRANGE_DECK"
    assert ia["allow_reorder"] is True and ia["allow_position"] is True
    # 並び替え C,A,B を上(TOP)へ → デッキ最上面が C
    gm.resolve_interaction(p1, {"selected_uuids": [cards[2].uuid, cards[0].uuid, cards[1].uuid], "position": "TOP"})
    assert gm.active_interaction is None
    assert [c.master.card_id for c in p1.deck] == ["C", "A", "B"]
    assert p1.temp_zone == []


def test_arrange_deck_bottom_preserves_order():
    gm, p1, p2, src, cards = _setup_temp(3)
    act = GameAction(type=ActionType.DECK_BOTTOM,
                     target=TargetQuery(player=Player.SELF, zone=Zone.TEMP, select_mode="REMAINING", count=-1),
                     status="ARRANGE", dest_position="BOTTOM")
    _run(gm, p1, src, act)
    # 下固定（CHOOSE でない）→ 並び替えのみ
    assert gm.active_interaction["allow_position"] is False
    gm.resolve_interaction(p1, {"selected_uuids": [cards[1].uuid, cards[2].uuid, cards[0].uuid], "position": "BOTTOM"})
    assert [c.master.card_id for c in p1.deck] == ["B", "C", "A"]


def test_headless_default_keeps_current_order_bottom():
    """drain 相当の既定 payload(selected_uuids=[], position なし)で現状順・デッキ下になる。"""
    gm, p1, p2, src, cards = _setup_temp(3)
    act = GameAction(type=ActionType.DECK_BOTTOM,
                     target=TargetQuery(player=Player.SELF, zone=Zone.TEMP, select_mode="REMAINING", count=-1),
                     status="ARRANGE", dest_position="CHOOSE")
    _run(gm, p1, src, act)
    gm.resolve_interaction(p1, {"selected_uuids": [], "index": 0})
    assert gm.active_interaction is None
    assert [c.master.card_id for c in p1.deck] == ["A", "B", "C"]  # 現状順・下


def test_order_life_reorder():
    gm, p1, p2 = make_game()
    gm.phase = Phase.MAIN; gm.turn_player = p1; gm.opponent = p2
    src = make_instance(make_master(card_id="SRC"), owner="P1"); p1.field.append(src)
    life = [make_instance(make_master(card_id=c), owner="P1") for c in "XYZ"]
    p1.life = list(life)
    act = GameAction(type=ActionType.ORDER_LIFE)
    _run(gm, p1, src, act)
    ia = gm.active_interaction
    assert ia and ia["action_type"] == "ARRANGE_DECK" and ia["allow_position"] is False
    # ライフを Z,X,Y の順へ（life[0]=一番上）
    gm.resolve_interaction(p1, {"selected_uuids": [life[2].uuid, life[0].uuid, life[1].uuid]})
    assert [c.master.card_id for c in p1.life] == ["Z", "X", "Y"]


def test_order_life_single_card_no_suspend():
    gm, p1, p2 = make_game()
    src = make_instance(make_master(card_id="SRC"), owner="P1"); p1.field.append(src)
    p1.life = [make_instance(make_master(card_id="solo"), owner="P1")]
    _run(gm, p1, src, GameAction(type=ActionType.ORDER_LIFE))
    assert gm.active_interaction is None  # 1枚は並べ替え不要


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider", "-p", "no:capture"]))
