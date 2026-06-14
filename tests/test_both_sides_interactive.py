"""「お互いの〜」両側効果で、各プレイヤーが自分側の選択を個別に行う（Phase 項目1）回帰テスト。

従来は両側とも既定選択（候補先頭）で非中断確定していた（人間が選べない）。選択の余地が
あるサイド（候補>必要枚数の手札捨て等）は、そのサイドのプレイヤーに**相手→自分の順**で
個別に選ばせる。選択の余地が無いサイド（位置確定の隠しゾーン・候補≤必要数）は非中断。

対象: OP05-058「その後、お互いは手札が5枚になるように、自身の手札を捨てる」。
"""
import os

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.models.models import CardInstance
from opcg_sim.src.models.enums import TriggerType
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


def _game(p1_hand, p2_hand):
    p1 = Player(name="P1", deck=[], leader=inst("OP01-001", "P1"))
    p2 = Player(name="P2", deck=[], leader=inst("OP01-001", "P2"))
    gm = GameManager(player1=p1, player2=p2)
    gm.turn_player = p1
    gm.turn_count = 3
    p1.hand = [inst("OP01-016", "P1") for _ in range(p1_hand)]
    p2.hand = [inst("OP02-013", "P2") for _ in range(p2_hand)]
    src = inst("OP05-058", "P1")
    p1.trash = [src]            # source をゾーンに置く（中断・再開で解決される）
    return gm, p1, p2, src


def _activate(gm, p1, src):
    ab = next(a for a in src.master.abilities if a.trigger == TriggerType.ACTIVATE_MAIN)
    gm.resolve_ability(p1, ab, source_card=src)


def test_each_player_chooses_own_discard():
    """相手→自分の順に SELECT_TARGET を提示し、各自が選んだ札が実際に捨てられる。"""
    gm, p1, p2, src = _game(p1_hand=7, p2_hand=6)
    _activate(gm, p1, src)

    # 1つ目: 相手(P2)が自分の手札6枚から1枚（7→… いや 6-5=1枚）捨てる
    ai = gm.active_interaction
    assert ai["action_type"] == "SELECT_TARGET" and ai["player_id"] == "P2"
    assert ai["constraints"] == {"min": 1, "max": 1}
    p2_pick = p2.hand[3]
    gm.resolve_interaction(p2, {"selected_uuids": [p2_pick.uuid]})

    # 2つ目: 自分(P1)が手札7枚から2枚捨てる
    ai = gm.active_interaction
    assert ai["action_type"] == "SELECT_TARGET" and ai["player_id"] == "P1"
    assert ai["constraints"] == {"min": 2, "max": 2}
    p1_picks = [p1.hand[1], p1.hand[5]]
    gm.resolve_interaction(p1, {"selected_uuids": [c.uuid for c in p1_picks]})

    assert gm.active_interaction is None
    assert len(p1.hand) == 5 and len(p2.hand) == 5
    # 選んだ札が実際に捨てられている（先頭自動ではなく選択が反映）
    assert p2_pick not in p2.hand and p2_pick in p2.trash
    for c in p1_picks:
        assert c not in p1.hand and c in p1.trash


def test_no_interaction_when_within_limit():
    """両者とも手札5枚以下なら捨てる必要が無く、中断は起きない。"""
    gm, p1, p2, src = _game(p1_hand=4, p2_hand=5)
    _activate(gm, p1, src)
    assert gm.active_interaction is None
    assert len(p1.hand) == 4 and len(p2.hand) == 5


def test_only_side_over_limit_chooses():
    """超過している側だけが選択する（自分のみ超過 → 自分の選択のみ）。"""
    gm, p1, p2, src = _game(p1_hand=7, p2_hand=5)
    _activate(gm, p1, src)
    # 相手(P2)は5枚＝捨て不要 → 自分(P1)の選択のみが提示される
    ai = gm.active_interaction
    assert ai is not None and ai["player_id"] == "P1"
    assert ai["constraints"] == {"min": 2, "max": 2}
    gm.resolve_interaction(p1, {"selected_uuids": [c.uuid for c in p1.hand[:2]]})
    assert gm.active_interaction is None
    assert len(p1.hand) == 5 and len(p2.hand) == 5
