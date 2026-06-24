"""「このX…し、パワー/コスト±N」連用接続の主語継承 回帰テスト。

OP12-063 ヴィンスモーク・レイジュ
  「自分のトラッシュにイベントが4枚以上ある場合、このキャラのパワー+2000し、コスト+5。/【ブロッカー】」

バグ: 後半句「コスト+5」が主語「このキャラ」を失い、parse_target が
「自分のキャラ1枚を選ぶ(CHOOSE)」へ誤フォールバックしていた。PASSIVE 自己バフが
盤面に自軍キャラ2体以上いると再計算中に対象選択へ中断＝盤面が固まる。
修正後は後半句も SOURCE（自身）に解決し、中断せず自身へ +5 が乗る。

同型: EB04-048 ロブ・ルッチ（…パワー+1000し、コスト+2）も同じ取りこぼし。
非対象: OP14-086（…し、自分の『B・W』…キャラすべてを、コスト+2）は独自主語を持つため不変。
"""
import os

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.models.models import CardInstance
from opcg_sim.src.utils.loader import CardLoader

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "opcg_sim", "data", "opcg_cards.json")
_DB = None


def db():
    global _DB
    if _DB is None:
        _DB = CardLoader(DATA)
        _DB.load()
    return _DB


def inst(cid, owner="P1"):
    return CardInstance(db().get_card(cid), owner)


def _cost_action(master):
    """対象の COST_REDUCTION BUFF アクションを返す（PASSIVE 能力の効果列から）。"""
    for ab in master.abilities:
        eff = ab.effect
        actions = getattr(eff, "actions", [eff]) if eff else []
        for act in actions:
            if getattr(act, "status", None) == "COST_REDUCTION":
                return act
    return None


def test_op12_063_cost_clause_targets_self_not_choose():
    """コスト+5 句の対象は自身(SOURCE)であり、CHOOSE（要選択）ではない。"""
    act = _cost_action(db().get_card("OP12-063"))
    assert act is not None, "コスト+5 が COST_REDUCTION として解析されていない"
    assert act.target.select_mode == "SOURCE"


def test_eb04_048_cost_clause_targets_self_not_choose():
    """同型 EB04-048 のコスト+2 句も自身(SOURCE)に解決される。"""
    act = _cost_action(db().get_card("EB04-048"))
    assert act is not None
    assert act.target.select_mode == "SOURCE"


def test_op14_086_cost_clause_unaffected():
    """OP14-086 は独自主語『B・W』…すべてを持つため SOURCE 化しない（ALL のまま）。"""
    act = _cost_action(db().get_card("OP14-086"))
    assert act is not None
    assert act.target.select_mode != "SOURCE"


def test_op12_063_passive_applies_to_self_without_suspending_with_two_chars():
    """自軍キャラ2体（レイジュ＋他）でも PASSIVE 再計算が中断せず、+2000/+5 が
    レイジュ自身に乗り、もう1体は不変であること（=フリーズ再現の回帰）。"""
    reiju = inst("OP12-063", "P1")
    other = inst("EB01-022", "P1")
    p1 = Player(name="P1", deck=[], leader=inst("OP01-001", "P1"))
    p2 = Player(name="P2", deck=[], leader=inst("OP01-001", "P2"))
    p1.field = [reiju, other]
    # 条件成立: トラッシュにイベント4枚
    p1.trash = [inst(c, "P1") for c in ("EB01-028", "EB01-029", "EB02-030", "EB02-031")]

    gm = GameManager(player1=p1, player2=p2)
    gm.turn_player = p1
    gm.turn_count = 2

    base_other_power = other.get_power(True)
    base_other_cost = other.current_cost

    gm.refresh_passive_state()

    # 中断していない（キャラ選択画面で固まっていない）
    assert not gm.active_interaction

    # レイジュ自身に +2000 / +5
    assert reiju.get_power(True) == (reiju.master.power or 0) + 2000
    assert reiju.current_cost == (reiju.master.cost or 0) + 5

    # もう1体は影響を受けない
    assert other.get_power(True) == base_other_power
    assert other.current_cost == base_other_cost


def test_op12_063_inactive_below_trash_threshold():
    """トラッシュのイベントが4枚未満なら発動せず、パワー/コストは素のまま。"""
    reiju = inst("OP12-063", "P1")
    p1 = Player(name="P1", deck=[], leader=inst("OP01-001", "P1"))
    p2 = Player(name="P2", deck=[], leader=inst("OP01-001", "P2"))
    p1.field = [reiju]
    p1.trash = [inst(c, "P1") for c in ("EB01-028", "EB01-029", "EB02-030")]  # 3枚

    gm = GameManager(player1=p1, player2=p2)
    gm.turn_player = p1
    gm.turn_count = 2

    gm.refresh_passive_state()

    assert not gm.active_interaction
    assert reiju.get_power(True) == (reiju.master.power or 0)
    assert reiju.current_cost == (reiju.master.cost or 0)
