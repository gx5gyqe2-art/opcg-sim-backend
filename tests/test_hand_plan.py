"""`hand_plan.py`（T66・手札の価値＝計画の価値 `H`・札の価値＝計画の増分 `ΔH`）の算術を固める。

**厳密 DP**（各札は 1 回・ターンごとのドンの上限・t ターン目は `s^t` で割引）・**ΔH ≥ 0**・**出せない札は 0**・
**穴を埋める札は高い**（次のターンに出せる札が無い手札に来た低コストの札）。
"""
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import hand_plan as HP  # noqa: E402
from theory_order import KO_P  # noqa: E402

S = 1.0 - KO_P


def test_caps_are_active_now_then_stock_plus_one_per_turn_capped_at_ten():
    assert HP.caps_of(2, 4) == [2, 5, 6, 10]                    # 今・次・その次・それより後（上限 10）
    assert HP.caps_of(0, 9) == [0, 10, 10, 10]                  # 上限 10
    assert HP.caps_of(3.4, 3.6, turns=3) == [3, 5, 10]
    # **T68**: 「それより後」の枠は残りターン `R` に応じて 10 × max(1, R − 3)——R ≤ 3 は 1 ターンぶんのまま
    assert HP.caps_of(2, 4, r_turns=5) == [2, 5, 6, 20]
    assert HP.caps_of(2, 4, r_turns=4) == [2, 5, 6, 10]
    assert HP.caps_of(2, 4, r_turns=2) == [2, 5, 6, 10]
    assert HP.caps_of(2, 4, turns=3, r_turns=5) == [2, 5, 30]


def test_the_plan_is_an_exact_knapsack_over_turns_with_discount():
    items = [(2, 0.05), (5, 0.15), (3, 0.08)]
    caps = [2, 5, 6]
    # 今 2 コスト・次 5 コスト・その次 3 コスト（合計を最大にする並び）
    assert HP.plan_value(items, caps, S) == pytest.approx(0.05 + S * 0.15 + S * S * 0.08, abs=1e-9)
    # 同じターンに 2 枚も可: caps [6, 0, 0] なら割引なしで 6 以内の最良＝5 コスト 1 枚（0.15）> 2+3 コスト（0.13）
    assert HP.plan_value(items, [6, 0, 0], S) == pytest.approx(0.15, abs=1e-9)
    assert HP.plan_value([(2, 0.09), (3, 0.08), (5, 0.15)], [6, 0, 0], S) == pytest.approx(0.17, abs=1e-9)   # 2 枚の方が高ければ 2 枚
    assert HP.plan_value([(7, 0.3)], caps, S) == 0.0                                            # 出せない札は 0
    assert HP.plan_value([(1, None), (1, 0.0)], caps, S) == 0.0                                 # 価値の無い札は計画に入らない
    assert HP.plan_value([], caps, S) == 0.0


def test_delta_h_rewards_the_card_that_fills_the_hole():
    caps = [2, 5, 6]
    hand = [(5, 0.15), (3, 0.08)]
    assert HP.delta_h(hand, (2, 0.05), caps, S) == pytest.approx(0.05, abs=1e-9)                 # 今出せる穴を埋める → 満額
    assert HP.delta_h(hand, (7, 0.30), caps, S) == 0.0                                          # 出せない → 0
    # もう 1 枚の 5 コストは t=2（上限 6）にしか入らず 3 コストを押し出す＝増分は s²·(0.10 − 0.08)
    assert HP.delta_h(hand, (5, 0.10), caps, S) == pytest.approx(S * S * (0.10 - 0.08), abs=1e-9)
    # 「それより後」の枠（4 枠目・割引 s³）が在れば、新しい 5 コストを t=2 に置いて 3 コストを後ろへ回す方が高い
    assert HP.delta_h(hand, (5, 0.10), caps + [10], S) == pytest.approx(S * S * 0.10 - S * S * 0.08 + S ** 3 * 0.08, abs=1e-9)
    assert HP.delta_h([], (2, 0.05), caps, S) == pytest.approx(0.05, abs=1e-9)
    assert HP.playable_next([(7, 0.3)], caps) is False and HP.playable_next([(4, 0.1)], caps) is True
    assert HP.playable_next([(4, 0.1)], [2]) is None


def test_summarise_compares_searched_and_drawn_delta_h():
    draws = [{"cid": "D", "v": 0.05, "dh": 0.02, "dg": 0.00, "dtotal": 0.02, "counter_card": False, "turn": 2, "hand_n": 5, "playable_next_before": True},
             {"cid": "E", "v": 0.07, "dh": 0.04, "dg": 0.08, "dtotal": 0.08, "counter_card": True, "turn": 3, "hand_n": 5, "playable_next_before": False}]
    ss = [{"cid": "A", "k": 5.0, "turn": 3, "found": 1,
           "got": [{"cid": "X", "v": 0.06, "dh": 0.06, "dg": 0.01, "dtotal": 0.06, "counter_card": False, "playable_next_before": False}]},
          {"cid": "A", "k": 5.0, "turn": 4, "found": 0, "got": []}]
    out = HP.summarise(ss, draws, min_card=1)
    assert out["draws"]["dh_mean"] == pytest.approx(0.03) and out["draws"]["hole_before"] == pytest.approx(0.5)
    assert out["draws"]["dg_mean"] == pytest.approx(0.04) and out["draws"]["dtotal_mean"] == pytest.approx(0.05)
    assert out["draws"]["counter_card_share"] == pytest.approx(0.5)
    b = out["all"]
    assert b["found"] == 0.5 and b["dh_searched"] == pytest.approx(0.06)
    assert b["premium_dh"] == pytest.approx(0.06 - 0.03) and b["premium_v"] == pytest.approx(0.06 - 0.06)
    assert b["dg_searched"] == pytest.approx(0.01) and b["premium_dg"] == pytest.approx(0.01 - 0.04)
    assert b["dtotal_searched"] == pytest.approx(0.06) and b["premium_total"] == pytest.approx(0.06 - 0.05)
    assert b["guard_motivated_share"] == 0.0 and b["counter_card_share"] == 0.0
    assert b["hole_before"] == 1.0 and b["dh_zero_share"] == 0.0
    assert "A" in out["by_card"] and out["k=5"]["n"] == 2


def test_added_card_gains_reads_each_entering_card_in_the_hand_it_landed_in(monkeypatch):
    """**T69**: 窓の中で手札に入った札（after − before の多重集合）を、入った先の手札の残りに対して `card_deltas` で読む。
    同じ札が 2 枚入れば別の枠を当て、手札に見つからない札は落とす。"""
    hands = {"B": ["X"], "A": ["X", "S", "S", "Y"]}                          # S が 2 枚・Y が 1 枚入った
    monkeypatch.setattr(HP, "hand_ids", lambda ci, idx2cid: list(hands[ci]))
    monkeypatch.setattr(HP, "hand_items", lambda tok, ci, idx2cid, cards, olp, r: [
        {"cid": c, "cost": 1.0, "v": 0.01 * (k + 1), "counter": 0.0, "event": False} for k, c in enumerate(hands[ci])])
    monkeypatch.setattr(HP, "caps_of", lambda a, b, turns=4, r_turns=None: [2, 3, 4, 10])
    monkeypatch.setattr(HP, "don_stock", lambda sc, tok, side="me": 2.0)
    monkeypatch.setattr(HP.HG, "incoming", lambda tok: [])
    monkeypatch.setattr(HP.HG, "take_cost_of", lambda life: 0.087)
    seen = []

    def _deltas(rest, card, caps, xs, take):
        seen.append((card["cid"], sorted(it["cid"] for it in rest), tuple(caps), tuple(xs), take))
        return {"dtotal": card["v"], "dh": card["v"], "dg": 0.0, "counter": 0.0, "counter_card": False}
    monkeypatch.setattr(HP, "card_deltas", _deltas)
    sc = [0.0] * 24; sc[HP.SC_OPP_LIFE] = 4.0; sc[HP.SC_MY_DON] = 2.0; sc[HP.SC_MY_LIFE] = 3.0
    got = HP.added_card_gains(sc, None, "B", "A", {}, None)
    assert got == [("S", pytest.approx(0.02)), ("S", pytest.approx(0.03)), ("Y", pytest.approx(0.04))]
    assert [s[0] for s in seen] == ["S", "S", "Y"]
    assert seen[0][1] == ["S", "X", "Y"] and seen[1][1] == ["S", "X", "Y"] and seen[2][1] == ["S", "S", "X"]   # その札だけ除く
    assert seen[0][2] == (2, 3, 4, 10) and seen[0][4] == 0.087
    assert HP.added_card_gains(sc, None, "A", "B", {}, None) == []                                          # 出ただけなら空
