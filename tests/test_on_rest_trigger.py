"""「（このキャラが）レストになった時」誘発を**効果によるレスト**でも発火させる回帰テスト。

従来、この誘発はアタック宣言（declare_attack）でのみ発火し、効果（REST アクション）で
レストになった場合は不発だった（TEST_SPEC §8.2「効果でのレストは未対応」）。
REST アクションのアクティブ→レスト遷移で同じ誘発を積むようにした。

対象: OP14-027 ジュラキュール・シャンクス
  「【自分のターン中】このキャラがレストになった時、相手の元々のパワー7000以下の
   キャラ1枚までを、レストにする。」
"""
import os

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.models.models import CardInstance
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


def _setup(turn_player_is_p1=True):
    p1 = Player(name="P1", deck=[], leader=inst("OP01-001", "P1"))
    p2 = Player(name="P2", deck=[], leader=inst("OP01-001", "P2"))
    gm = GameManager(player1=p1, player2=p2)
    gm.turn_player = p1 if turn_player_is_p1 else p2
    gm.turn_count = 2
    src = inst("OP14-027", "P1")
    p1.field = [src]
    victim = inst("OP01-016", "P2")  # コスト2/パワー2000（7000以下）
    p2.field = [victim]
    return gm, p1, p2, src, victim


def _rest(gm, player, card):
    gm.apply_action_to_engine(player, GameAction(type=ActionType.REST, raw_text="rest"), [card], 1)


def test_effect_rest_fires_on_rest_trigger():
    """自分のターン中に効果でレストになった OP14-027 が on-rest を誘発する。"""
    gm, p1, p2, src, victim = _setup(turn_player_is_p1=True)
    _rest(gm, p1, src)
    assert src.is_rest is True
    # on-rest 効果（相手のパワー7000以下を1枚レスト）が対象選択で中断している。
    assert gm.active_interaction is not None
    gm.resolve_interaction(p1, {"selected_uuids": [victim.uuid]})
    assert victim.is_rest is True
    assert gm.active_interaction is None


def test_opponent_turn_does_not_fire():
    """相手ターン中の効果レストでは【自分のターン中】条件で不発（誤発火防止）。"""
    gm, p1, p2, src, victim = _setup(turn_player_is_p1=False)
    _rest(gm, p1, src)
    assert src.is_rest is True
    assert gm.active_interaction is None   # 条件不成立で誘発しない
    assert victim.is_rest is False


def test_already_rested_does_not_refire():
    """既にレストのキャラを再度レストにしても遷移が無いので誘発しない。"""
    gm, p1, p2, src, victim = _setup(turn_player_is_p1=True)
    src.is_rest = True
    _rest(gm, p1, src)
    assert gm.active_interaction is None
    assert victim.is_rest is False


def test_plain_card_rest_is_noop():
    """on-rest を持たないキャラの効果レストは誘発を起こさず素通りする。"""
    gm, p1, p2, _src, _victim = _setup(turn_player_is_p1=True)
    plain = inst("OP01-016", "P1")
    p1.field.append(plain)
    _rest(gm, p1, plain)
    assert plain.is_rest is True
    assert gm.active_interaction is None
