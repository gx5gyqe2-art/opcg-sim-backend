"""「キャラかドン!!合計N枚」の混在選択（1キャラ+1ドン 等）の回帰テスト。

従来は Choice[REST(キャラ最大N), REST_DON(N)] で近似され、キャラとドン!!を混ぜて
合計N枚にできなかった（TEST_SPEC §8.2 残課題）。合計N(≥2) は単一 REST に CHAR_OR_DON
フラグの混在候補を持たせ、最大N枚を自由に選べるようにした。対象: OP06-035 / OP12-037。
"""
import os

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core.effects.matcher import get_target_cards
from opcg_sim.src.models.models import CardInstance, DonInstance
from opcg_sim.src.models.effect_types import Sequence
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


def _setup():
    p1 = Player(name="P1", deck=[inst("OP01-016") for _ in range(5)],
                leader=inst("OP01-001", "P1"))
    p2 = Player(name="P2", deck=[], leader=inst("OP01-001", "P2"))
    gm = GameManager(player1=p1, player2=p2)
    gm.turn_player = p1
    gm.turn_count = 2
    p1.life = [inst("OP01-016") for _ in range(3)]
    p1.hand = []
    ch1 = inst("OP01-016", "P2")
    ch2 = inst("OP03-001", "P2")
    p2.field = [ch1, ch2]
    dons = [DonInstance(owner_id="P2") for _ in range(3)]
    p2.don_active = dons
    p2.don_rested = []
    return gm, p1, p2, ch1, ch2, dons


def _on_play(src):
    return next(a for a in src.master.abilities if a.trigger == TriggerType.ON_PLAY)


def test_candidate_pool_is_chars_plus_dons():
    """CHAR_OR_DON の候補プールは相手のキャラ＋ドン!!（リーダー/ステージは含まない）。"""
    gm, p1, p2, ch1, ch2, dons = _setup()
    src = inst("OP06-035", "P1")
    p1.field = [src]
    rest = _on_play(src).effect.actions[0]   # Sequence の先頭 = REST(CHAR_OR_DON)
    assert "CHAR_OR_DON" in rest.target.flags
    pool = get_target_cards(gm, rest.target, src)
    assert len(pool) == 5                      # 2 キャラ + 3 ドン
    assert p2.leader not in pool


def test_mixed_one_char_one_don():
    """1キャラ+1ドンの混在選択で、その両方がレストになる。"""
    gm, p1, p2, ch1, ch2, dons = _setup()
    src = inst("OP06-035", "P1")
    p1.field = [src]
    gm.resolve_ability(p1, _on_play(src), source_card=src)
    gm.resolve_interaction(p1, {"selected_uuids": [ch1.uuid, dons[0].uuid]})
    assert ch1.is_rest is True
    assert ch2.is_rest is False
    assert len(p2.don_active) == 2 and len(p2.don_rested) == 1


def test_two_dons_only():
    """ドン!!2枚だけの選択も可能（混在を強制しない）。"""
    gm, p1, p2, ch1, ch2, dons = _setup()
    src = inst("OP06-035", "P1")
    p1.field = [src]
    gm.resolve_ability(p1, _on_play(src), source_card=src)
    gm.resolve_interaction(p1, {"selected_uuids": [dons[0].uuid, dons[1].uuid]})
    assert len(p2.don_rested) == 2 and len(p2.don_active) == 1
    assert ch1.is_rest is False and ch2.is_rest is False


def test_two_chars_only():
    """キャラ2枚だけの選択も可能（混在を強制しない）。"""
    gm, p1, p2, ch1, ch2, dons = _setup()
    src = inst("OP06-035", "P1")
    p1.field = [src]
    gm.resolve_ability(p1, _on_play(src), source_card=src)
    gm.resolve_interaction(p1, {"selected_uuids": [ch1.uuid, ch2.uuid]})
    assert ch1.is_rest is True and ch2.is_rest is True
    assert len(p2.don_active) == 3   # ドンは無傷


def test_up_to_select_zero():
    """「合計2枚まで」なので 0 枚選択も可能で、何もレストにならない。"""
    gm, p1, p2, ch1, ch2, dons = _setup()
    src = inst("OP06-035", "P1")
    p1.field = [src]
    gm.resolve_ability(p1, _on_play(src), source_card=src)
    gm.resolve_interaction(p1, {"selected_uuids": []})
    assert ch1.is_rest is False and ch2.is_rest is False
    assert len(p2.don_active) == 3
