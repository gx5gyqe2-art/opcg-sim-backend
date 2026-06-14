"""FREEZE_DON（OP07-026 ドン側）の回帰テスト。

「相手の、レストのキャラかドン‼1枚までは、次の相手のリフレッシュフェイズでアクティブに
ならない」は、従来キャラ側のみ FREEZE され**ドン側が脱落**していた（TEST_SPEC §8.2 残課題）。
パーサが Choice[FREEZE(キャラ), FREEZE_DON] を生成し、エンジンがレストのドン!!を1回だけ
アクティブ化しないことを固定する。
"""
import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
from opcg_sim.src.models.models import CardInstance, DonInstance
from opcg_sim.src.models.effect_types import GameAction, Choice
from opcg_sim.src.models.enums import ActionType

v2 = EffectParserV2()

OP07_026 = "【登場時】相手の、レストのキャラかドン‼1枚までは、次の相手のリフレッシュフェイズでアクティブにならない。"


def _make_game():
    p1 = Player(name="P1", deck=[], leader=None)
    p2 = Player(name="P2", deck=[], leader=None)
    gm = GameManager(player1=p1, player2=p2)
    gm.turn_player = p1
    gm.turn_count = 2
    return gm, p1, p2


def test_parse_char_or_don_freeze_is_choice():
    """「キャラかドン」は Choice[FREEZE, FREEZE_DON] になる（ドン側が脱落しない）。"""
    abilities = v2.parse_card_text(OP07_026)
    choice = abilities[0].effect
    assert isinstance(choice, Choice)
    types = {o.type for o in choice.options if isinstance(o, GameAction)}
    assert types == {ActionType.FREEZE, ActionType.FREEZE_DON}
    don = next(o for o in choice.options if o.type == ActionType.FREEZE_DON)
    assert don.status == "OPPONENT"
    assert don.value.base == 1


def test_parse_single_char_freeze_stays_plain():
    """「キャラ」のみ（ドン無し）は従来どおり単一 FREEZE（回帰防止）。"""
    t = "相手のレストのキャラ1枚までは、次の相手のリフレッシュフェイズでアクティブにならない。"
    eff = v2.parse_card_text(t)[0].effect
    assert isinstance(eff, GameAction) and eff.type == ActionType.FREEZE


def test_freeze_don_skips_one_refresh():
    """FREEZE_DON は相手のレストのドン!!を value 枚だけ次のリフレッシュで据え置く。"""
    gm, p1, p2 = _make_game()
    p2.don_active = []
    p2.don_rested = [DonInstance(owner_id="P2", is_rest=True) for _ in range(3)]

    action = GameAction(type=ActionType.FREEZE_DON, status="OPPONENT")
    assert gm.apply_action_to_engine(p1, action, [], 1) is True

    frozen = [d for d in p2.don_rested if d.is_frozen]
    assert len(frozen) == 1

    # 相手（ドン!!の持ち主）のリフレッシュ: フリーズ1枚は据え置き、残り2枚はアクティブ化。
    gm.refresh_all(p2)
    assert len(p2.don_active) == 2
    assert len(p2.don_rested) == 1
    assert p2.don_rested[0].is_frozen is False  # 1回限り（フラグは下りる）

    # 次のリフレッシュではフリーズが無く、据え置かれていた1枚もアクティブ化する。
    gm.refresh_all(p2)
    assert len(p2.don_active) == 3
    assert len(p2.don_rested) == 0


def test_freeze_don_caps_at_available():
    """据え置きは利用可能なレストのドン!!枚数まで（不足してもエラーにならない）。"""
    gm, p1, p2 = _make_game()
    p2.don_active = []
    p2.don_rested = [DonInstance(owner_id="P2", is_rest=True)]
    action = GameAction(type=ActionType.FREEZE_DON, status="OPPONENT")
    gm.apply_action_to_engine(p1, action, [], 3)
    assert sum(1 for d in p2.don_rested if d.is_frozen) == 1
