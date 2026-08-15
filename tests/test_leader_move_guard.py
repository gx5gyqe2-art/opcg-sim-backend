"""リーダーのゾーン移動ガード（bb0 発見の欠陥A・2026-08-11）。

**欠陥**: `move_card` はリーダー（リーダー枠＝リスト外の専用スロット）に届くと、
元リストからの remove が素通りしたまま宛先リストへ append し、**カードが複製される**
（bb0 合成世界の実測: INCLUDE_LEADER 選択→条件KO の経路・seed880007 ほか計3局で
カード総数 +1）。実カードでも「リーダーとキャラを選ぶ」系（INCLUDE_LEADER・OP07-059 等）
の選択群へ後続の KO/移動が及べば同じ経路に入り得る。

**修正**: ルール上リーダーはゾーン移動しないため、`move_card` の先頭で中央 no-op ガード。
発生源（KO/バウンス/トラッシュ/デッキ送り等の per_target ハンドラ）に依らず全経路を塞ぐ。
"""
import conftest  # noqa: F401

from engine_helpers import make_game, make_master
from opcg_sim.src.models.models import CardInstance, CardType
from opcg_sim.src.models.enums import Zone


def _totals(p):
    return (len(p.deck) + len(p.hand) + len(p.field) + len(p.trash)
            + len(p.life) + len(p.temp_zone) + (1 if p.leader else 0)
            + (1 if p.stage else 0))


def _with_leader(gm, p, cid="T-L01"):
    p.leader = CardInstance(make_master(card_id=cid, type=CardType.LEADER,
                                        power=5000, life=5), p.name)
    return p.leader


def test_move_leader_to_trash_is_noop():
    gm, p1, p2 = make_game()
    ldr = _with_leader(gm, p2)
    before = _totals(p2)
    gm.move_card(ldr, Zone.TRASH, p2)
    assert p2.leader is ldr, "リーダーが枠から消えた"
    assert len(p2.trash) == 0, "リーダーの複製がトラッシュへ落ちた（欠陥Aの再発）"
    assert _totals(p2) == before


def test_move_leader_all_zones_conserve_cards():
    gm, p1, p2 = make_game()
    ldr = _with_leader(gm, p1)
    before = (_totals(p1), _totals(p2))
    for zone in (Zone.HAND, Zone.DECK, Zone.LIFE, Zone.TEMP, Zone.FIELD):
        gm.move_card(ldr, zone, p1)
        assert p1.leader is ldr
    assert (_totals(p1), _totals(p2)) == before


def test_ko_handler_on_leader_is_noop():
    """KO ハンドラ経由（bb0 の実経路）でも保存則が守られる。"""
    from opcg_sim.src.core.actions import per_target as PT
    gm, p1, p2 = make_game()
    ldr = _with_leader(gm, p2)
    before = _totals(p2)
    PT.ko(gm, p1, None, ldr, p2, None, 0, ldr)
    assert p2.leader is ldr and len(p2.trash) == 0 and _totals(p2) == before


def test_normal_character_move_still_works():
    """ガードが通常キャラの移動を壊していないこと（回帰対照）。"""
    gm, p1, p2 = make_game()
    _with_leader(gm, p1)
    c = CardInstance(make_master(card_id="T-C01", cost=2, power=3000), p1.name)
    p1.field.append(c)
    gm.move_card(c, Zone.TRASH, p1)
    assert c in p1.trash and c not in p1.field
