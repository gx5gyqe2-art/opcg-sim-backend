"""ON_REST 誘発の主語・要因フィルタ（任意主語＋「自分の/相手の効果で」）の回帰テスト。

item 7（「このキャラがレストになった時」）に続き、任意主語・要因修飾付きの on-rest を
ON_REST トリガーとして発火させる（従来 PASSIVE/ACTIVATE_MAIN/YOUR_TURN へ誤写像され不発）。

- OP10-036: 「【自分のターン中】【ターン1回】キャラが自分の効果でレストになった時、
  自分のドン!!1枚までを、アクティブにする。」（任意主語＋自分の効果で）
- PRB02-009: 「このキャラが相手の効果でレストになった時、発動できる。…2枚引く。」
  （このキャラ＋相手の効果で・任意）
"""
import os

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.models.models import CardInstance, DonInstance
from opcg_sim.src.models.effect_types import GameAction
from opcg_sim.src.models.enums import ActionType
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


def _game(turn="P1"):
    p1 = Player(name="P1", deck=[], leader=inst("OP01-001", "P1"))
    p2 = Player(name="P2", deck=[], leader=inst("OP01-001", "P2"))
    gm = GameManager(player1=p1, player2=p2)
    gm.turn_player = p1 if turn == "P1" else p2
    gm.turn_count = 2 if turn == "P1" else 3
    return gm, p1, p2


def _rest_by(gm, actor, card):
    gm.apply_action_to_engine(actor, GameAction(type=ActionType.REST, raw_text="rest"), [card], 1)


def _drain(gm, responder):
    guard = 0
    while gm.active_interaction and guard < 6:
        guard += 1
        gm.resolve_interaction(responder, {"selected_uuids": [], "accepted": True, "confirm": True})


# --- OP10-036: 任意主語 ＋ 自分の効果で -----------------------------------

def test_any_subject_self_effect_activates_don():
    """自分の効果でキャラ（相手キャラ可）をレスト→自分のドン!!がアクティブになる。"""
    gm, p1, p2 = _game(turn="P1")
    p1.field = [inst("OP10-036", "P1")]
    p2.field = [inst("OP01-016", "P2")]
    rd = DonInstance(owner_id="P1", is_rest=True)
    p1.don_rested = [rd]
    p1.don_active = []
    _rest_by(gm, p1, p2.field[0])      # 自分(P1)の効果で相手キャラをレスト
    assert len(p1.don_active) == 1     # ドン!!がアクティブ化
    assert len(p1.don_rested) == 0


def test_self_effect_not_fired_by_opponent():
    """相手(P2)の効果でレストにしても「自分の効果で」条件で不発。"""
    gm, p1, p2 = _game(turn="P2")
    p1.field = [inst("OP10-036", "P1")]
    p2.field = [inst("OP01-016", "P2")]
    p1.don_rested = [DonInstance(owner_id="P1", is_rest=True)]
    p1.don_active = []
    _rest_by(gm, p2, p1.field[0])      # 相手(P2)の効果でレスト
    assert len(p1.don_active) == 0     # 発火しない


# --- PRB02-009: このキャラ ＋ 相手の効果で（任意） -----------------------

def test_this_card_opponent_effect_fires():
    """相手の効果でこのキャラがレスト→トラッシュへ置き2枚引く。"""
    gm, p1, p2 = _game(turn="P2")
    host = inst("PRB02-009", "P1")
    p1.field = [host]
    p1.hand = []
    p1.trash = []
    p1.deck = [inst("OP01-016", "P1") for _ in range(4)]
    _rest_by(gm, p2, host)             # 相手(P2)の効果でレスト
    _drain(gm, p1)
    assert len(p1.hand) == 2
    assert host in p1.trash
    assert host not in p1.field


def test_this_card_not_fired_by_own_effect():
    """自分の効果でこのキャラをレストにしても「相手の効果で」条件で不発。"""
    gm, p1, _p2 = _game(turn="P1")
    host = inst("PRB02-009", "P1")
    p1.field = [host]
    p1.hand = []
    p1.deck = [inst("OP01-016", "P1") for _ in range(4)]
    _rest_by(gm, p1, host)             # 自分の効果でレスト
    assert gm.active_interaction is None
    assert len(p1.hand) == 0
    assert host in p1.field


def test_attack_rest_does_not_match_effect_qualified():
    """アタック（by_attack=True）は「効果で」限定の ON_REST 主語フィルタに一致しない。"""
    gm, p1, _p2 = _game(turn="P1")
    host = inst("OP10-036", "P1")       # 「キャラが自分の効果でレストになった時」
    ab = next(a for a in host.master.abilities
              if a.trigger.name == "ON_REST")
    # アタック由来は不一致（効果でないため）
    assert gm._rest_subject_matches(ab, host, host, p1, by_attack=True) is False
    # 自分の効果（effect_controller=host_owner）なら一致
    assert gm._rest_subject_matches(ab, host, host, p1, by_attack=False,
                                    effect_controller=p1) is True
    # 相手の効果（effect_controller≠host_owner）は「自分の効果で」に不一致
    assert gm._rest_subject_matches(ab, host, host, p1, by_attack=False,
                                    effect_controller=_p2) is False
