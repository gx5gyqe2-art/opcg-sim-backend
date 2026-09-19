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


def test_caps_are_active_now_then_stock_plus_two_per_turn_capped_at_ten():
    """ドン!!フェイズは 2 枚（ユーザ指摘 2026-09-17・T66〜T70 は +1 で誤っていた）。"""
    assert HP.caps_of(2, 4) == [2, 6, 8, 10]                    # 今・次・その次・それより後（上限 10）
    assert HP.caps_of(0, 9) == [0, 10, 10, 10]                  # 上限 10
    assert HP.caps_of(3.4, 3.6, turns=3) == [3, 6, 10]
    # **T68**: 「それより後」の枠は残りターン `R` に応じて 10 × max(1, R − 3)——R ≤ 3 は 1 ターンぶんのまま
    assert HP.caps_of(2, 4, r_turns=5) == [2, 6, 8, 20]
    assert HP.caps_of(2, 4, r_turns=4) == [2, 6, 8, 10]
    assert HP.caps_of(2, 4, r_turns=2) == [2, 6, 8, 10]
    assert HP.caps_of(2, 4, turns=3, r_turns=5) == [2, 6, 30]
    assert HP.DON_PER_TURN == 2


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
    monkeypatch.setattr(HP, "own_field_ids", lambda ci, idx2cid: [])
    monkeypatch.setattr(HP, "state_of_row", lambda *a, **k: {})                    # T72 の状態もここでは見ない
    monkeypatch.setattr(HP, "apply_inflow", lambda items, *a, **k: list(items))   # T70 の相方待ちはここでは見ない
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


def test_plan_value_accepts_a_per_turn_value_for_partner_waiting_cards():
    """**T70**: `v` がターンごとの並びなら、その t の値で計画に入る（相方が来るまで待つ札は後のターンほど高い）。"""
    caps = [4, 4, 4, 10]
    waiting = (2, [0.02, 0.05, 0.08, 0.08])                  # 今出せば 0.02・2 ターン待てば 0.08
    assert HP.plan_value([waiting], caps, S) == pytest.approx(max(0.02, S * 0.05, S * S * 0.08, S ** 3 * 0.08), abs=1e-9)
    assert HP.plan_value([(2, [0.0, 0.0, 0.0, 0.0])], caps, S) == 0.0
    assert HP.v_at([0.1, 0.2], 5) == 0.2 and HP.v_at(0.3, 2) == 0.3 and HP.v_at(None, 0) == 0.0 and HP.v_at([], 1) == 0.0
    assert HP.v_scalar([0.02, 0.05, 0.08], S) == pytest.approx(max(0.02, S * 0.05, S * S * 0.08))
    assert HP.v_scalar(0.04) == 0.04 and HP.v_scalar(None) == 0.0
    assert HP.playable_next([(2, [0.0, 0.05, 0.0])], caps) is True and HP.playable_next([(2, [0.05, 0.0, 0.0])], caps) is False


def test_arrival_probability_and_the_inflow_from_draws_life_and_searches(monkeypatch):
    """**T70**: `P = 1 − (1 − p)^n`・受けるライフ＝守る規則で止めない攻撃の本数・サーチの当たりはデッキの割合から。"""
    assert HP.arrival_prob(0.0, 3) == 0.0 and HP.arrival_prob(1.0, 1) == 1.0 and HP.arrival_prob(1.0, 0) == 0.0
    assert HP.arrival_prob(0.36, 1.5) == pytest.approx(1.0 - 0.64 ** 1.5)
    items = [{"cid": "A", "cost": 2.0, "v": 0.01, "counter": 2000.0, "event": False}]
    take = 0.087
    # 攻撃 2 本（x=1000, x=0）: 2000 カウンター 1 枚で 1 本は止める（節約 0.077）・残り 1 本は受ける → 1 枚入る
    assert HP.expected_taken(items, [1000.0, 0.0], take) == 1.0
    assert HP.expected_taken([], [1000.0, 0.0], take) == 2.0
    assert HP.expected_taken([{"cid": "B", "cost": 5.0, "v": 0.20, "counter": 2000.0, "event": False}], [1000.0], take) == 1.0   # 高い札は切らず受ける
    assert HP.expected_taken(items, [], take) == 0.0
    monkeypatch.setattr(HP, "expected_search_hits", lambda items, deck, cards: 0.5)
    assert HP.inflow_per_turn(items, [1000.0, 0.0], take, ["X"], None) == pytest.approx(1.0 + 1.0 + 0.5)


def test_inflow_item_turns_an_enabler_into_a_per_turn_value(monkeypatch):
    """**T70**: 相方待ちの札の `v_t = base + P(t までに来る) × E[相方 − μ]`・相方が今在れば全部の t で満額・相方待ちでなければそのまま。"""
    import search_price as SP
    from theory_order import MU

    class _Cards:
        def info(self, cid):
            return {"E": {"cost": 4}, "P": {"cost": 2}, "Q": {"cost": 2}, "Z": {"cost": 5}}.get(cid)
    cards = _Cards()
    monkeypatch.setattr(SP, "enabler_target", lambda cid, cards_json=None: {"cost_max": 2} if cid == "E" else None)
    monkeypatch.setattr(SP, "eligible_hand_cards", lambda target, items, cards, skip_cid=None: [it["cid"] for it in items if it["cid"] in ("P", "Q")])
    monkeypatch.setattr(SP, "eligible_deck_cards", lambda target, deck, cards: [c for c in deck if c in ("P", "Q")])
    monkeypatch.setattr(HP, "_base_value", lambda cid, info, cards, olp, r, field=(), st_base=None: 0.10)
    # 相方が手札に在るときの札の値（効果のコスト・条件込み）＝base + 相方の取り分
    monkeypatch.setattr(HP, "_value_with_partner", lambda cid, info, cards, olp, r, partner, field=(), st_base=None: 0.10 + {"P": 0.12, "Q": 0.10}.get(partner, 0.0) - MU)
    monkeypatch.setattr(HP, "inflow_per_turn", lambda items, xs, take, deck, cards: 2.0)
    e = {"cid": "E", "cost": 4.0, "v": 0.2, "counter": 0.0, "event": False}
    z = {"cid": "Z", "cost": 5.0, "v": 0.3, "counter": 0.0, "event": False}
    deck = ["P", "Q", "Z", "Z"]                                   # 合う札の割合 p = 0.5
    got = HP.inflow_item(e, [z], deck, [], 0.087, cards, 5000.0, 4.0, turns=4)
    p_t = [HP.arrival_prob(0.5, 2.0 * t) for t in range(4)]
    gain = ((0.12 - MU) + (0.10 - MU)) / 2
    assert got["v"] == pytest.approx([0.10 + p * gain for p in p_t]) and got["v"][0] == pytest.approx(0.10)
    assert got["v_static"] == 0.2 and got["p_partner"] == 0.5 and got["inflow_per_turn"] == 2.0
    # 相方が今の手札に在れば全部の t で満額（一番良い相方）
    p_item = {"cid": "P", "cost": 2.0, "v": 0.12, "counter": 0.0, "event": False}
    got2 = HP.inflow_item(e, [p_item, z], deck, [], 0.087, cards, 5000.0, 4.0, turns=4)
    assert got2["v"] == pytest.approx([0.10 + (0.12 - MU)] * 4) and got2["p_partner"] == 1.0
    # 相方待ちでない札はそのまま・デッキに合う札が無ければ base だけ
    assert HP.inflow_item(z, [e], deck, [], 0.087, cards, 5000.0, 4.0) is z
    assert HP.inflow_item(e, [z], ["Z", "Z"], [], 0.087, cards, 5000.0, 4.0)["v"] == [0.10] * 4
    HP.set_inflow_mode("off")
    try:
        assert HP.apply_inflow([e], deck, [], 0.087, cards, 5000.0, 4.0) == [e]
    finally:
        HP.set_inflow_mode("on")
    with pytest.raises(ValueError):
        HP.set_inflow_mode("guess")


def test_project_state_advances_the_clocks_from_rules_and_the_theory(monkeypatch):
    """**T73**: ドンは +2/ターン（上限 10・ターン開始は全部アクティブ）・自分のライフは受ける本数・相手のライフは受ける規則が受けろと言う本数・
    トラッシュはカウンター＋KO＋イベント・ターンは +2t。`t = 0` と `off` はそのまま。"""
    st = {"my_don_total": 7.0, "my_don": 7, "my_don_active": 3, "opp_don_total": 9.0, "opp_don": 9, "opp_don_active": 9,
          "my_life": 4, "opp_life": 3, "my_trash": 2, "turn": 5, "my_field_ids": ["A", "B"],
          "my_attack_xs": [3000.0, -1000.0, 0.0]}
    items = [{"cid": "C", "cost": 1.0, "v": 0.01, "counter": 2000.0, "event": False},
             {"cid": "E", "cost": 1.0, "v": 0.03, "counter": 0.0, "event": True}]
    xs = [1000.0, 0.0]; take = 0.087
    monkeypatch.setattr(HP.TO, "c_of", lambda x, mode=None: 3.0 if x >= 2000 else 1.0)     # 3000 超過だけ受けろ（c > Θ）
    monkeypatch.setattr(HP.TO, "THETA", 1.58)
    assert HP.project_state(st, 0, items, xs, take) is st
    out = HP.project_state(st, 1, items, xs, take)
    assert out["my_don_total"] == 9.0 and out["my_don"] == 9 and out["my_don_active"] == 9      # +2・全部アクティブ
    assert out["opp_don_total"] == 10.0 and out["opp_don_active"] == 10                          # 上限 10
    assert out["my_life"] == pytest.approx(4 - 1.0)                                              # 2 本のうち 1 本を 2000 カウンターで止め、1 本受ける
    assert out["opp_life"] == pytest.approx(3 - 1.0)                                              # 3000 超過の 1 本だけ受けろ
    assert out["my_trash"] == pytest.approx(2 + 1.0 + HP.KO_P * 2 + 1.0 / HP.PLAN_TURNS)        # 切った 1 枚 + KO 期待 + イベント
    assert out["turn"] == 7
    out2 = HP.project_state(st, 2, items, xs, take)
    assert out2["my_don_total"] == 10.0 and out2["my_life"] == pytest.approx(2.0) and out2["turn"] == 9
    assert HP.counters_cut(items, xs, take) == 1.0 and HP.opp_life_loss_per_turn(st) == 1.0
    HP.set_cond_clock_mode("off")
    try:
        assert HP.project_state(st, 2, items, xs, take) is st
    finally:
        HP.set_cond_clock_mode("on")
    with pytest.raises(ValueError):
        HP.set_cond_clock_mode("guess")


def test_a_conditional_card_gets_a_per_turn_value_from_the_projected_state(monkeypatch):
    """**T73**: 登場時能力に条件が在る札（相方待ちではない）は、t ターン後の状態で読んだ `v_t` の並びになる。"""
    import search_price as SP
    monkeypatch.setattr(SP, "enabler_target", lambda cid, cards_json=None: None)
    monkeypatch.setattr(HP, "has_on_play_condition", lambda cid: cid == "K")
    # v は「ドンの総在庫が 10 なら 0.2・それ未満なら 0.05」（ベン・ベックマン型）
    monkeypatch.setattr(HP, "use_value", lambda cid, info, olp, r, cards=None, st=None: 0.2 if (st or {}).get("my_don_total", 0) >= 10 else 0.05)

    class _Cards:
        def info(self, cid):
            return {"cost": 5}
    st = {"my_don_total": 6.0, "my_don": 6, "my_don_active": 6, "my_life": 4, "opp_life": 4, "my_trash": 0, "turn": 3,
          "my_field_ids": [], "my_attack_xs": []}
    k = {"cid": "K", "cost": 5.0, "v": 0.05, "counter": 0.0, "event": False}
    got = HP.inflow_item(k, [], [], [], 0.087, _Cards(), 5000.0, 4.0, turns=4, st_base=st)
    assert got["v"] == pytest.approx([0.05, 0.05, 0.2, 0.2]) and got["v_static"] == 0.05      # 6 → 8 → 10 で条件が立つ
    z = {"cid": "Z", "cost": 5.0, "v": 0.05, "counter": 0.0, "event": False}
    assert HP.inflow_item(z, [], [], [], 0.087, _Cards(), 5000.0, 4.0, turns=4, st_base=st) is z   # 条件も相方も無ければそのまま
    HP.set_cond_clock_mode("off")
    try:
        assert HP.inflow_item(k, [], [], [], 0.087, _Cards(), 5000.0, 4.0, turns=4, st_base=st) is k
    finally:
        HP.set_cond_clock_mode("on")
