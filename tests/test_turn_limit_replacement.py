"""置換/保護系【ターン1回】の per-turn enforce の回帰テスト。

効果検証イテレーション1で確定したバグ:
  置換効果（「(このキャラが)KOされる場合、代わりに〜」）の【ターン1回】が enforce されず、
  同一ターンに何度でも発動していた（docs/effect_verification/REPORT.md §2.1）。

根本原因（二重の取りこぼし）:
  1. parser が自己置換（「このキャラ」）の final_condition を None にし TURN_LIMIT を捨てる。
  2. _active_replacement / _active_protection が resolve_ability を通らず ability_used_this_turn を
     参照/加算しない。
"""
import os

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

from effect_coverage import _build_test_state
from opcg_sim.src.utils.loader import CardLoader
from opcg_sim.src.models.enums import ActionType, TriggerType

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opcg_sim", "data")


@pytest.fixture(scope="module")
def db():
    d = CardLoader(os.path.join(DATA, "opcg_cards.json"))
    d.load()
    for cid in list(d.raw_db.keys()):
        d.get_card(cid)
    return d


def _setup_replacement(db, cid):
    """cid のカードを場に置き、その REPLACE_EFFECT の status を返す。"""
    m = db.get_card(cid)
    gm, p1, p2, _ = _build_test_state(m)
    card = next((c for c in p1.field if c.master.card_id == cid), None)
    status = None
    for ab in m.abilities:
        if ab.trigger != TriggerType.PASSIVE:
            continue
        eff = gm._find_action(ab.effect, ActionType.REPLACE_EFFECT)
        if eff is not None:
            status = eff.status
            break
    return gm, card, status


# 自己置換【ターン1回】の代表カード（REPORT §2.1 で同ターン2回発動を確認したもの）
@pytest.mark.parametrize("cid", ["OP10-034", "ST09-010", "OP10-074"])
def test_replacement_turn_limit_enforced(db, cid):
    gm, card, status = _setup_replacement(db, cid)
    assert card is not None and status is not None, f"{cid}: REPLACE_EFFECT 能力が見つからない"

    first = gm._active_replacement(card, (status,))
    second = gm._active_replacement(card, (status,))
    assert first is True, f"{cid}: 1回目の置換が発動しない"
    assert second is False, f"{cid}: 【ターン1回】置換が同一ターンに2回発動した（未 enforce）"

    # ターン境界（refresh）相当で使用回数がリセットされれば再び発動できる。
    card.reset_turn_status(clear_usage=True)
    assert gm._active_replacement(card, (status,)) is True, f"{cid}: 次ターンで再発動できない"


def test_protection_turn_limit_enforced(db):
    """保護系【ターン1回】（OP10-118「このキャラはターンに1回、相手の効果でKOされない」）も
    1ターンに1回まで（inline 表記は parser が TURN_LIMIT を載せないため raw_text 併用で enforce）。"""
    m = db.get_card("OP10-118")
    gm, p1, p2, _ = _build_test_state(m)
    card = next((c for c in p1.field if c.master.card_id == "OP10-118"), None)
    status = None
    for ab in m.abilities:
        if ab.trigger != TriggerType.PASSIVE:
            continue
        eff = gm._find_action(ab.effect, ActionType.PREVENT_LEAVE)
        if eff is not None:
            status = eff.status
            break
    assert card is not None and status is not None

    assert gm._active_protection(card, (status,), p2) is True
    assert gm._active_protection(card, (status,), p2) is False, "保護がターン2回発動した（未 enforce）"
    card.reset_turn_status(clear_usage=True)
    assert gm._active_protection(card, (status,), p2) is True
