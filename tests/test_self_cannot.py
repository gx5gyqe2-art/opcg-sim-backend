"""自己制限（self_cannot / 「自分は、このターン中、…できない」）の enforce テスト。

課題3(3a): 従来 RULE_PROCESSING の no-op だった自己制限を、parser が制限キーへ写像し、
エンジンが player.restrictions に記録して各アクション地点で enforce する。
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

from engine_helpers import make_game, make_master, make_instance, action
from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
from opcg_sim.src.models.effect_types import GameAction, Sequence, Branch, Choice, ValueSource
from opcg_sim.src.models.enums import ActionType, CardType, Phase

v2 = EffectParserV2()


def _collect(node):
    """効果ツリーを歩いて GameAction を全て列挙する。"""
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


def _restriction_status(text):
    """カードテキストを V2 で解析し、自己制限(RULE_PROCESSING)の status を返す。"""
    found = []
    for ab in v2.parse_card_text(text):
        for ga in _collect(ab.effect):
            if ga.type == ActionType.RULE_PROCESSING and ga.status:
                found.append(ga)
    return found


# --- パーサ: 述語 → 制限キーの写像 -----------------------------------------

@pytest.mark.parametrize("text,key", [
    ("【メイン】その後、自分は、このターン中、自分の場にキャラカードを登場できない。", "CANNOT_PLAY_CHARACTER"),
    ("【メイン】その後、自分は、このターン中、手札からカードをプレイできない。", "CANNOT_PLAY_FROM_HAND"),
    ("【メイン】その後、自分は、このターン中、自分の効果でカードを引くことができない。", "CANNOT_DRAW_BY_EFFECT"),
    ("【メイン】その後、自分は、このターン中、自分の効果でライフを手札に加えられない。", "CANNOT_LIFE_TO_HAND"),
    ("【メイン】その後、自分は、このターン中、リーダーにアタックできない。", "CANNOT_ATTACK_LEADER"),
    ("【メイン】その後、自分は、このターン中、キャラの効果でドン‼をアクティブにできない。", "CANNOT_ACTIVATE_DON"),
])
def test_parse_self_cannot_status(text, key):
    found = _restriction_status(text)
    assert any(ga.status == key for ga in found), f"{key} not parsed from: {text}"


def test_parse_cost_filtered_play():
    found = _restriction_status("【メイン】その後、自分は、このターン中、元々のコスト7以上のキャラカードを登場できない。")
    ga = next((g for g in found if g.status == "CANNOT_PLAY_CHARACTER"), None)
    assert ga is not None
    assert ga.value is not None and ga.value.base == 7


def test_parse_deckbuild_rule_is_noop():
    """「デッキに入れることができない」は構築ルールなので status なし（従来どおり no-op）。"""
    abilities = v2.parse_card_text("ルール上、自分はコスト5以上のカードをデッキに入れることができない。")
    for ab in abilities:
        for ga in _collect(ab.effect):
            if ga.type == ActionType.RULE_PROCESSING:
                assert ga.status is None


# --- エンジン: 記録と enforce -----------------------------------------------

def _restrict(gm, player, key, min_cost=None):
    """RULE_PROCESSING+status を解決して player へ制限を記録する（早期ハンドラ経由）。"""
    val = ValueSource(base=min_cost) if min_cost else None
    ok = gm.apply_action_to_engine(
        player, GameAction(type=ActionType.RULE_PROCESSING, status=key, value=val), [], 0)
    assert ok
    assert key in player.restrictions


def _enter_main(gm, player):
    gm.phase = Phase.MAIN
    gm.turn_player = player
    gm.opponent = gm.p2 if player is gm.p1 else gm.p1


def test_cannot_play_character_blocks_play():
    gm, p1, p2 = make_game()
    _enter_main(gm, p1)
    _restrict(gm, p1, "CANNOT_PLAY_CHARACTER")
    card = make_instance(make_master(card_id="C-1", type=CardType.CHARACTER, cost=3), owner="P1")
    p1.hand.append(card)
    with pytest.raises(ValueError):
        gm.play_card_action(p1, card)
    assert card in p1.hand  # 登場していない


def test_cannot_play_character_cost_filter():
    gm, p1, p2 = make_game()
    _enter_main(gm, p1)
    _restrict(gm, p1, "CANNOT_PLAY_CHARACTER", min_cost=7)
    cheap = make_instance(make_master(card_id="C-lo", type=CardType.CHARACTER, cost=3), owner="P1")
    big = make_instance(make_master(card_id="C-hi", type=CardType.CHARACTER, cost=8), owner="P1")
    p1.hand.extend([cheap, big])
    # コスト3は制限外 → 登場できる
    gm.play_card_action(p1, cheap)
    assert cheap in p1.field
    # コスト8(>=7)は制限対象 → 弾かれる
    with pytest.raises(ValueError):
        gm.play_card_action(p1, big)
    assert big in p1.hand


def test_cannot_play_from_hand_blocks_event_too():
    gm, p1, p2 = make_game()
    _enter_main(gm, p1)
    _restrict(gm, p1, "CANNOT_PLAY_FROM_HAND")
    ev = make_instance(make_master(card_id="E-1", type=CardType.EVENT, cost=1), owner="P1")
    p1.hand.append(ev)
    with pytest.raises(ValueError):
        gm.play_card_action(p1, ev)


def test_cannot_attack_leader():
    gm, p1, p2 = make_game()
    _enter_main(gm, p1)
    _restrict(gm, p1, "CANNOT_ATTACK_LEADER")
    attacker = make_instance(make_master(card_id="A-1", type=CardType.CHARACTER), owner="P1")
    attacker.is_rest = False
    p1.field.append(attacker)
    with pytest.raises(ValueError):
        gm.declare_attack(attacker, p2.leader)
    assert gm.active_battle is None


def test_cannot_draw_by_effect():
    gm, p1, p2 = make_game()
    p1.deck = [make_instance(make_master(card_id=f"D-{i}"), owner="P1") for i in range(5)]
    before = len(p1.hand)
    _restrict(gm, p1, "CANNOT_DRAW_BY_EFFECT")
    gm.apply_action_to_engine(p1, action(ActionType.DRAW, value=2), [], 2)
    assert len(p1.hand) == before  # ドローされない


def test_cannot_activate_don_by_effect():
    gm, p1, p2 = make_game()
    rested = p1.don_deck[:2]
    p1.don_rested = list(rested)
    _restrict(gm, p1, "CANNOT_ACTIVATE_DON")
    gm.apply_action_to_engine(p1, action(ActionType.ACTIVE_DON, value=2), [], 2)
    assert len(p1.don_rested) == 2  # アクティブ化されない


def test_cannot_life_to_hand_by_effect():
    gm, p1, p2 = make_game()
    life = make_instance(make_master(card_id="L-x"), owner="P1")
    p1.life.append(life)
    _restrict(gm, p1, "CANNOT_LIFE_TO_HAND")
    gm.apply_action_to_engine(
        p1, GameAction(type=ActionType.MOVE_CARD, destination=None), [life], 0)
    # destination 未指定は HAND 既定。制限により移動しない。
    assert life in p1.life and life not in p1.hand


def test_restriction_expires_next_turn():
    gm, p1, p2 = make_game()
    _enter_main(gm, p1)
    _restrict(gm, p1, "CANNOT_PLAY_CHARACTER")
    # 同ターン中は有効
    assert gm._active_restriction(p1, "CANNOT_PLAY_CHARACTER") is not None
    # 次ターン（turn_count 進行）で失効
    gm.turn_count += 1
    assert gm._active_restriction(p1, "CANNOT_PLAY_CHARACTER") is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider", "-p", "no:capture"]))
