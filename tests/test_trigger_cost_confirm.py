"""自動誘発のコスト使用確認（CONFIRM_OPTIONAL）— コスト句の支払いは常に任意。

OPCG ルールではコスト句（「X：Y」の X）の支払いは「できる/してもよい」表記の有無に
依らず任意（払わなければ効果が発生しないだけ）。自動誘発（登場時/KO時/ターン終了時/
アタック時 等）はコストが有る限り発動前に使用確認を挟む（resolver §2.5）。
確認を挟まない例外＝発動自体が意思表示のもの: ACTIVATE_MAIN／【トリガー】(CONFIRM_TRIGGER
で確認済み)／【カウンター】／イベントカード。

回帰の発端: ボルサリーノ OP16-073【自分のターン終了時】ドン!!-2 が確認なしで強制発動し、
ドン!!返却の選択が押し付けられていた（「できる」表記が無く cost_optional=False だったため）。

実行:
  OPCG_LOG_SILENT=1 python -m pytest tests/test_trigger_cost_confirm.py -q -s -p no:cacheprovider
"""
import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google スタブ)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "harness"))

from leader_test_helpers import build, db, get_ability
from opcg_sim.src.models.models import CardInstance


def _make_borsalino(gm, p1, rested=True):
    bor = CardInstance(db().get_card("OP16-073"), p1.name)
    bor.is_rest = rested
    p1.field.append(bor)
    return bor


def test_turn_end_cost_ability_asks_confirmation():
    """OP16-073: 【自分のターン終了時】ドン!!-2 はターン終了で CONFIRM_OPTIONAL を立てる（強制発動しない）。"""
    gm, p1, p2, L = build("OP16-060")
    bor = _make_borsalino(gm, p1)
    don_before = len(p1.don_active) + len(p1.don_rested)
    gm.end_turn()
    ia = gm.active_interaction
    assert ia is not None and ia["action_type"] == "CONFIRM_OPTIONAL"
    assert "ボルサリーノ" in ia.get("message", "")
    # 確認が立った時点ではコスト未払い・効果未解決
    assert bor.is_rest is True
    assert len(p1.don_active) + len(p1.don_rested) == don_before


def test_turn_end_cost_ability_decline_pays_nothing():
    """OP16-073: 使用を拒否したらコストを払わず、キャラはレストのまま。"""
    gm, p1, p2, L = build("OP16-060")
    bor = _make_borsalino(gm, p1)
    don_before = len(p1.don_active) + len(p1.don_rested)
    gm.end_turn()
    gm.resolve_interaction(p1, {"accepted": False})
    assert gm.active_interaction is None
    assert bor.is_rest is True
    assert len(p1.don_active) + len(p1.don_rested) == don_before


def test_turn_end_cost_ability_accept_pays_and_resolves():
    """OP16-073: 受諾したらドン!!-2 を払ってアクティブになり【ブロッカー】を得る。"""
    gm, p1, p2, L = build("OP16-060")
    bor = _make_borsalino(gm, p1)
    don_before = len(p1.don_active) + len(p1.don_rested)
    gm.end_turn()
    gm.resolve_interaction(p1, {"accepted": True})
    ia = gm.active_interaction
    assert ia is not None and ia["action_type"] == "SELECT_RESOURCE"  # 返すドン!!の選択
    cands = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
    gm.resolve_interaction(p1, {"selected_uuids": cands[:2]})
    assert bor.is_rest is False
    assert "ブロッカー" in (bor.current_keywords | bor.timed_keywords)
    assert (len(p1.don_active) + len(p1.don_rested)) - don_before == -2


def test_turn_end_multiple_cost_abilities_both_confirmed():
    """同名2枚のターン終了時コスト能力: 1枚目の確認中でも2枚目は待ち行列に積まれ、両方に確認が来る。"""
    gm, p1, p2, L = build("OP16-060")
    b1 = _make_borsalino(gm, p1)
    b2 = _make_borsalino(gm, p1)
    gm.end_turn()
    # 1枚目: 受諾して支払い
    assert gm.active_interaction["action_type"] == "CONFIRM_OPTIONAL"
    gm.resolve_interaction(p1, {"accepted": True})
    ia = gm.active_interaction
    assert ia["action_type"] == "SELECT_RESOURCE"
    cands = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
    gm.resolve_interaction(p1, {"selected_uuids": cands[:2]})
    # 2枚目: 消えずに確認が来る（従来は先行の中断で無言消失していた系）
    ia = gm.active_interaction
    assert ia is not None and ia["action_type"] == "CONFIRM_OPTIONAL"
    gm.resolve_interaction(p1, {"accepted": False})
    assert b1.is_rest is False
    assert b2.is_rest is True


def test_on_play_cost_ability_asks_confirmation():
    """OP16-065 サカズキ: 【登場時】ドン!!-1 も登場時に CONFIRM_OPTIONAL を挟む（拒否可能）。"""
    gm, p1, p2, L = build("OP16-060")
    saka = CardInstance(db().get_card("OP16-065"), p1.name)
    p1.hand.append(saka)
    don_before = len(p1.don_active) + len(p1.don_rested)
    gm.play_card_action(p1, saka)
    ia = gm.active_interaction
    assert ia is not None and ia["action_type"] == "CONFIRM_OPTIONAL"
    gm.resolve_interaction(p1, {"accepted": False})
    assert gm.active_interaction is None
    assert len(p1.don_active) + len(p1.don_rested) == don_before  # 何も払っていない


def test_activate_main_cost_needs_no_confirmation():
    """起動メイン（例: OP16-065 の【起動メイン】ドン1レスト）は起動自体が意思表示＝確認を挟まない。"""
    gm, p1, p2, L = build("OP16-060")
    saka = CardInstance(db().get_card("OP16-065"), p1.name)
    p1.field.append(saka)
    ab = get_ability(saka.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, saka)
    ia = gm.active_interaction
    # CONFIRM_OPTIONAL ではなく、直接コスト/効果の解決（対話が立つ場合も確認ではない）
    assert ia is None or ia["action_type"] != "CONFIRM_OPTIONAL"
