"""`lethal_rule.py`（T136・決着の定義）の算術を固める。

**決着 ⇔ 相手が最善で守っても通る本数 ≥ 相手の残りライフ ＋ 1**（規則だけ・完全情報）。
**「＋1」は規則**——ライフ 0 で損害を受けたら負け（最後のライフ札を取られても負けではない・`rules/battle.rs`）。
押さえるのは 6 つ:

1. **守り手は切れるだけ切る**（`max`）——1 本ごとに使うカウンター値が最小の組を選び、安い攻撃から止める＝止まる本数が最大。
2. **ブロッカーは安い攻撃から横取りする**・**通らない攻撃（相手リーダー未満）は数えない**。
3. **ドンは安い攻撃から 1 体 4 枚まで**（規則）——付ければ止めるのに要るカウンターが増える。
4. **`actual` は守り手の手札が渡らなければ落ちる**（黙って `share` に落とさない）。
5. **6 指標の算術**（宣言した勝者の精度・そのターンに終わった精度・終局での再現率・局の再現率・先読み・誤宣言）。
6. **受けたライフの札は全部手札に入る**（規則）——ライフ L なら L 枚がカウンターになりうる。

**基盤健全性ではない**——器の誤りは決着の定義の誤判断に直結する（T117 の 0.23 は器の読み方の差だった）ので必須側。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import lethal_rule as LR  # noqa: E402
import theory_order as TO  # noqa: E402
from theory_order import (S_CAN_ATTACK, S_IS_BLOCKER, S_IS_CHAR, S_IS_REST, S_POWER,  # noqa: E402
                          SC_MY_DON, SC_OPP_HAND, SC_OPP_LEADER_POWER, SC_OPP_LIFE, SLOT_OPP_FIELD,
                          SLOT_OWN_FIELD)


@pytest.fixture(autouse=True)
def _defaults():
    LR.set_lethal_hand_mode("actual")
    LR.set_lethal_stop_mode("max")
    LR.set_lethal_life_mode("off")          # 既定は `draw` だが、規則 1〜5 のテストはライフの札なしで読む
    LR.set_avg_counter_mode("rules")
    yield
    LR.set_lethal_hand_mode("actual")
    LR.set_lethal_stop_mode("max")
    LR.set_lethal_life_mode("draw")
    LR.set_avg_counter_mode("rules")


# ---- 1. 切れるだけ切る ---------------------------------------------------------------------------

def test_min_counter_set_is_chosen_per_attack():
    # 超過 1000 を止めるには 2000 以上要る: {2000} が {1000, 1000}（同じ和）と並ぶが、{3000} は選ばない
    idx = LR.stop_min_counter([3000.0, 2000.0, 1000.0, 1000.0], 1000.0)
    assert sum([3000.0, 2000.0, 1000.0, 1000.0][i] for i in idx) == 2000.0


def test_max_stops_counts_the_most_attacks_the_hand_can_stop():
    # 手札 {1000, 1000, 2000}・攻撃の超過 {0, 0, 3000}: 0 は 1000 で止まる ×2・3000 には 4000 要る＝残り 2000 では無理
    assert LR.max_stops([1000.0, 1000.0, 2000.0], [0.0, 0.0, 3000.0]) == 2
    # 同じ手札で {0, 2000}: 0 を 1000 で・2000 を 1000+2000 で＝2 本
    assert LR.max_stops([1000.0, 1000.0, 2000.0], [0.0, 2000.0]) == 2


def test_max_stops_ignores_attacks_that_do_not_reach():
    assert LR.max_stops([1000.0], [-2000.0, -1000.0]) == 0     # 通らない攻撃は止める必要が無いし数えもしない


def test_zero_counter_cards_cannot_stop_anything():
    assert LR.max_stops([0.0, 0.0, 0.0], [0.0]) == 0


def test_max_stops_never_exceeds_the_attacks_or_the_cards():
    rng = np.random.default_rng(3)
    for _ in range(50):
        cs = list(rng.choice([0.0, 1000.0, 2000.0], size=int(rng.integers(0, 7))))
        xs = list(rng.choice([-1000.0, 0.0, 1000.0, 2000.0, 3000.0], size=int(rng.integers(0, 5))))
        n = LR.max_stops(cs, xs)
        assert 0 <= n <= min(len([x for x in xs if x >= 0]), len([c for c in cs if c > 0]))


# ---- 2. ブロッカー・通らない攻撃 / 3. ドン --------------------------------------------------------

def _row(my_life=3.0, opp_life=1.0, opp_hand=0, don=0.0, leader_power=5000.0, olp=5000.0,
         own=(), opp=()):
    """`(sc, tok)` を組む。`own`/`opp` は `(power, is_blocker, is_rest, can_attack)` の列。"""
    sc = np.zeros(64)
    sc[SC_OPP_LIFE] = opp_life; sc[SC_OPP_HAND] = opp_hand; sc[SC_MY_DON] = don
    sc[SC_OPP_LEADER_POWER] = olp / 1e4
    tok = np.zeros((24, 24))
    tok[0, S_POWER] = leader_power / 1e4
    for s, (p, b, r, a) in zip(range(SLOT_OWN_FIELD.start, SLOT_OWN_FIELD.stop), own):
        tok[s, S_IS_CHAR] = 1.0; tok[s, S_POWER] = p / 1e4
        tok[s, S_IS_BLOCKER] = float(b); tok[s, S_IS_REST] = float(r); tok[s, S_CAN_ATTACK] = float(a)
    for s, (p, b, r, a) in zip(range(SLOT_OPP_FIELD.start, SLOT_OPP_FIELD.stop), opp):
        tok[s, S_IS_CHAR] = 1.0; tok[s, S_POWER] = p / 1e4
        tok[s, S_IS_BLOCKER] = float(b); tok[s, S_IS_REST] = float(r); tok[s, S_CAN_ATTACK] = float(a)
    return sc, tok


def test_leader_alone_declares_at_life_zero_with_an_empty_hand():
    sc, tok = _row(opp_life=0.0)
    ok, d = LR.lethal_of_row(sc, tok, with_don=False, defender_counters=[])
    assert ok and d["through"] == 1 and d["stops"] == 0 and d["hits"] == 1


def test_one_hit_at_life_one_is_not_lethal_by_the_rules():
    # 規則: 最後のライフ札を取られても負けではない（ライフ 0 で損害を受けたら負け）＝ライフ 1 には 2 本要る
    sc, tok = _row(opp_life=1.0)
    ok, d = LR.lethal_of_row(sc, tok, with_don=False, defender_counters=[])
    assert (not ok) and d["hits"] == 1
    sc2, tok2 = _row(opp_life=1.0, own=[(5000.0, 0, 0, 1)])
    ok2, d2 = LR.lethal_of_row(sc2, tok2, with_don=False, defender_counters=[])
    assert ok2 and d2["hits"] == 2


def test_one_counter_card_stops_the_leader():
    sc, tok = _row(opp_life=0.0)
    ok, d = LR.lethal_of_row(sc, tok, with_don=False, defender_counters=[1000.0])
    assert not ok and d["stops"] == 1


def test_active_blocker_takes_the_cheapest_attack():
    # リーダー 5000 ＋ 体 7000・相手ライフ 0・アクティブなブロッカー 1: 安い方（リーダー）が横取りされ 7000 が通る
    sc, tok = _row(opp_life=0.0, own=[(7000.0, 0, 0, 1)], opp=[(3000.0, 1, 0, 0)])
    ok, d = LR.lethal_of_row(sc, tok, with_don=False, defender_counters=[])
    assert ok and d["blockers"] == 1 and d["through"] == 1


def test_rested_blocker_does_not_count():
    sc, tok = _row(opp_life=1.0, own=[(7000.0, 0, 0, 1)], opp=[(3000.0, 1, 1, 0)])
    ok, d = LR.lethal_of_row(sc, tok, with_don=False, defender_counters=[])
    assert ok and d["blockers"] == 0 and d["through"] == 2


def test_attacks_below_the_leader_do_not_reach():
    sc, tok = _row(opp_life=0.0, leader_power=3000.0, olp=5000.0)
    ok, d = LR.lethal_of_row(sc, tok, with_don=False, defender_counters=[])
    assert not ok and d["through"] == 0


def test_don_goes_to_the_cheapest_attack_four_per_body():
    assert LR.attach_don([0.0, 2000.0], 5) == [4000.0, 3000.0]
    assert LR.attach_don([0.0], 10) == [4000.0]                 # 1 体 4 枚まで・余りは捨てる


def test_don_raises_the_counter_the_defender_needs():
    # リーダー超過 0 は 1000 で止まる。ドン 1 枚で超過 1000 → 2000 要る → 1000 の札では止まらない
    sc, tok = _row(opp_life=0.0, don=1.0)
    off, _ = LR.lethal_of_row(sc, tok, with_don=False, defender_counters=[1000.0])
    on, d = LR.lethal_of_row(sc, tok, with_don=True, defender_counters=[1000.0])
    assert (not off) and on and d["stops"] == 0


def test_life_zero_is_alive_and_one_hit_decides():
    # ライフ 0 は生きている（T134 で 84 行実在）＝1 本通れば決着。ライフが負なら（記録の不整合）宣言しない
    sc, tok = _row(opp_life=0.0)
    ok, _ = LR.lethal_of_row(sc, tok, with_don=False, defender_counters=[])
    assert ok
    sc[SC_OPP_LIFE] = -1.0
    ok_neg, _ = LR.lethal_of_row(sc, tok, with_don=False, defender_counters=[])
    assert not ok_neg


# ---- 4. 落ちるべきところで落ちる / share の再現 --------------------------------------------------

def test_actual_mode_refuses_to_run_without_the_defenders_hand():
    sc, tok = _row()
    with pytest.raises(ValueError):
        LR.lethal_of_row(sc, tok, with_don=False, defender_counters=None)


def test_share_mode_reads_the_count_times_the_deck_share():
    LR.set_lethal_hand_mode("share")
    sc, tok = _row(opp_life=0.0, opp_hand=4)
    ok_pess, d = LR.lethal_of_row(sc, tok, with_don=False, cut_share=None)   # 割合が無ければ 1.0（悲観側）
    assert (not ok_pess) and d["hand_read"] == 4.0
    ok_zero, _ = LR.lethal_of_row(sc, tok, with_don=False, cut_share=0.0)
    assert ok_zero


def test_econ_mode_needs_items_and_take_cost():
    LR.set_lethal_stop_mode("econ")
    sc, tok = _row()
    with pytest.raises(ValueError):
        LR.lethal_of_row(sc, tok, with_don=False, defender_counters=[1000.0])


# ---- 5. 6 指標の算術 ---------------------------------------------------------------------------

def test_metrics_on_a_hand_built_ledger():
    games = [
        # 勝者 0・終局 t=5。0 は t=3 で宣言（先読み 2）・t=5 でも宣言。1 は宣言なし。
        (0, 5, [(0, 1, False, False), (0, 3, True, False), (0, 5, True, True), (1, 2, False, False), (1, 4, False, False)]),
        # 勝者 1・終局 t=4。1 は t=4 だけ宣言（先読み 0）。0 が t=3 で誤って宣言。
        (1, 4, [(0, 1, False, False), (0, 3, True, False), (1, 2, False, False), (1, 4, True, True)]),
        # 勝者 0・終局 t=3。宣言なし。
        (0, 3, [(0, 1, False, False), (0, 3, False, True), (1, 2, False, False)]),
    ]
    m = LR.declare_metrics(games)
    assert m["declared"] == 4
    assert m["precision_winner"] == pytest.approx(3 / 4)
    assert m["precision_kill"] == pytest.approx(2 / 4)
    assert m["recall_end"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall_game"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["false_declared"] == 1 and m["games_both_declared"] == 1
    assert m["lead"]["n"] == 2 and m["lead"]["mean"] == pytest.approx(1.0) and m["lead"]["hist"] == {"0": 1, "2": 1}


def test_metrics_are_empty_safe():
    m = LR.declare_metrics([])
    assert m["declared"] == 0 and m["precision_winner"] == 0.0 and m["lead"]["n"] == 0


def test_cli_exposes_all_switches_and_they_reach_the_module(monkeypatch):
    monkeypatch.setattr(LR, "collect", lambda *a, **k: {})
    LR.main(["--in", "x", "--hand", "share", "--stop", "econ", "--life", "off", "--avg-counter", "rules"])
    assert (LR.LETHAL_HAND_MODE, LR.LETHAL_STOP_MODE, LR.LETHAL_LIFE_MODE, LR.AVG_COUNTER_MODE) == \
        ("share", "econ", "off", "rules")


# ---- 6. 受けたライフの札は手札に入る（規則・`rules/battle.rs`） -----------------------------------

def test_all_life_cards_become_counters():
    assert LR.life_cards_as_counters(4.0, 1000.0) == [1000.0] * 4
    assert LR.life_cards_as_counters(0.0, 1000.0) == []          # ライフ 0＝入る札は無い
    assert LR.life_cards_as_counters(3.0, 0.0) == []             # カウンターの無いデッキなら増えても止まらない


def test_life_cards_raise_the_bar_at_high_life_but_not_at_life_zero():
    LR.set_lethal_life_mode("draw")
    # リーダー ＋ 体 4（全部超過 0）・相手ライフ 4（5 本要る）・手札なし: ライフの札 4 枚（各 1000）で 4 本止まる → 1 本
    sc, tok = _row(opp_life=4.0, own=[(5000.0, 0, 0, 1)] * 4)
    ok_draw, d = LR.lethal_of_row(sc, tok, with_don=False, defender_counters=[], life_counter=1000.0)
    LR.set_lethal_life_mode("off")
    ok_off, _ = LR.lethal_of_row(sc, tok, with_don=False, defender_counters=[], life_counter=1000.0)
    assert (not ok_draw) and d["stops"] == 4 and ok_off
    # ライフ 0 では何も変わらない
    LR.set_lethal_life_mode("draw")
    sc1, tok1 = _row(opp_life=0.0)
    ok1, d1 = LR.lethal_of_row(sc1, tok1, with_don=False, defender_counters=[], life_counter=1000.0)
    assert ok1 and d1["stops"] == 0


def test_draw_mode_refuses_to_run_without_the_deck_average():
    LR.set_lethal_life_mode("draw")
    sc, tok = _row(opp_life=2.0)
    with pytest.raises(ValueError):
        LR.lethal_of_row(sc, tok, with_don=False, defender_counters=[], life_counter=None)


def test_the_plus_one_is_exact_at_every_life():
    # ライフ L には L+1 本要る（手札なし・ライフの札なし）
    LR.set_lethal_life_mode("off")
    for L in range(0, 5):
        sc_a, tok_a = _row(opp_life=float(L), own=[(5000.0, 0, 0, 1)] * L)          # L+1 本（リーダー込み）
        assert LR.lethal_of_row(sc_a, tok_a, with_don=False, defender_counters=[])[0]
        if L >= 1:                                                                  # L 本（ライフ 0 では作れない）
            sc_b, tok_b = _row(opp_life=float(L), own=[(5000.0, 0, 0, 1)] * (L - 1))
            assert not LR.lethal_of_row(sc_b, tok_b, with_don=False, defender_counters=[])[0]


# ---- 7. カウンターのイベントはドンを払う（規則） ---------------------------------------------------

def test_event_counter_needs_the_defenders_active_don():
    # 札: キャラ 1000・イベント 4000（費用 1）。超過 2000 には 3000 要る＝イベントが要る
    counters, costs = [1000.0, 4000.0], [0.0, 1.0]
    assert LR.max_stops(counters, [2000.0], costs, budget=1.0) == 1
    assert LR.max_stops(counters, [2000.0], costs, budget=0.0) == 0
    # 予算は使った分だけ減る: イベント 2 枚（各費用 1）・予算 1 → 1 本しか止まらない
    assert LR.max_stops([4000.0, 4000.0], [2000.0, 2000.0], [1.0, 1.0], budget=1.0) == 1
    assert LR.max_stops([4000.0, 4000.0], [2000.0, 2000.0], [1.0, 1.0], budget=2.0) == 2


def test_row_uses_the_opponents_active_don_as_the_event_budget():
    sc, tok = _row(opp_life=0.0)
    sc[LR.SC_OPP_DON_ACTIVE] = 0.0
    ok0, d0 = LR.lethal_of_row(sc, tok, with_don=False, defender_counters=[4000.0], defender_costs=[1.0])
    sc[LR.SC_OPP_DON_ACTIVE] = 1.0
    ok1, d1 = LR.lethal_of_row(sc, tok, with_don=False, defender_counters=[4000.0], defender_costs=[1.0])
    assert ok0 and d0["stops"] == 0 and (not ok1) and d1["stops"] == 1


# ---- 8. 決着フラグの切り出し（T138a・settled_map） --------------------------------------------------
#
# `collect` と `settled_map` は同じ下請け `_iter_declared_games` を読む（判定の式は `lethal_of_row` 1 か所）。
# 実記録での**同値性**（旧 `collect` と全く同じ JSON を出す）は pytest では確かめられない
# （このスイートはどの器も real の記録ディレクトリを読まない・`tests/_bootstrap.py` の方針どおり）ので、
# `docs/reports/2026-09-23_...` の A/B 比較（w41／w39+w42 で `json.load` した辞書が完全一致）で確認済み。
# ここでは**下請けを差し替えて**、2 つの入口が同じ行から同じ答えを作ることを固定する。

def _fake_generator(rows_by_game):
    def gen(dirs, limit_games, with_don):
        return iter(rows_by_game)
    return gen


def test_settled_map_keys_are_seed_w_t_and_values_are_bool(monkeypatch):
    rows_by_game = [
        (101, 0, 5, [(0, 1, False, False, {}, False, None),
                     (0, 3, True, False, {"life": 2.0}, False, 900.0),
                     (1, 2, False, False, {}, True, None)]),          # hand_missing 行も乗る（False のまま）
    ]
    monkeypatch.setattr(LR, "_iter_declared_games", _fake_generator(rows_by_game))
    sm = LR.settled_map(["x"])
    assert sm == {(101, 0, 1): False, (101, 0, 3): True, (101, 1, 2): False}
    assert all(isinstance(v, bool) for v in sm.values())


def test_settled_map_merges_multiple_games_without_key_collision(monkeypatch):
    rows_by_game = [
        (1, 0, 1, [(0, 1, True, True, {"life": 0.0}, False, 0.0)]),
        (2, 0, 1, [(0, 1, False, True, {"life": 0.0}, False, 0.0)]),   # 別局の同じ (w, t) は別 key（seed が違う）
    ]
    monkeypatch.setattr(LR, "_iter_declared_games", _fake_generator(rows_by_game))
    sm = LR.settled_map(["x"])
    assert sm == {(1, 0, 1): True, (2, 0, 1): False}


def test_collect_and_settled_map_agree_on_the_same_underlying_rows(monkeypatch):
    """**同じ下請けを読む 2 つの入口が食い違わない**——`collect` の宣言総数と `settled_map` の
    True の数が一致する（判定の式が 1 か所にしか無いことの検算）。"""
    rows_by_game = [
        (202, 0, 4, [(0, 1, False, False, {}, False, None),
                     (0, 3, True, False, {"life": 1.0}, False, 800.0),
                     (1, 2, True, False, {"life": 0.0}, False, 800.0),
                     (1, 4, False, True, {"life": 3.0}, False, 800.0)]),
    ]
    monkeypatch.setattr(LR, "_iter_declared_games", _fake_generator(rows_by_game))
    sm = LR.settled_map(["x"])
    out = LR.collect(["x"])
    assert sum(sm.values()) == out["declared"] == 2


def test_collect_dump_still_carries_life_counter_alongside_the_row_details(monkeypatch):
    """**回帰止め**: `_iter_declared_games` への分離で `life_counter` が `dd` に紛れ込み、`false_rows`
    に漏れて出力が変わる事故を一度踏んだ（本 T で発見・修正）。`dump` には引き続き乗り、
    `false_rows`（敗者側の宣言）には乗らないことを固定する。"""
    rows_by_game = [
        (303, 0, 2, [(0, 1, True, False, {"life": 1.0}, False, 777.0),          # 勝者側の宣言（dump のみ）
                     (1, 1, True, False, {"life": 5.0}, False, 555.0),          # 敗者側の宣言（false_rows にも乗る）
                     (0, 2, False, True, {"life": 0.0}, False, 333.0)]),
    ]
    monkeypatch.setattr(LR, "_iter_declared_games", _fake_generator(rows_by_game))
    dump = []
    out = LR.collect(["x"], dump=dump)
    winner_row = next(r for r in dump if r["w"] == 0 and r["t"] == 1)
    assert winner_row["life_counter"] == 777.0
    assert out["false_rows_sample"] == [{"seed": 303, "w": 1, "t": 1, "t_end": 2, "life": 5.0}]
    assert "life_counter" not in out["false_rows_sample"][0]


# ---- 9. 局ごとの最初の宣言ターン（T151-3・first_declared_turn） -----------------------------------
#
# `--pre-settle game` は「どちらかの席が最初に宣言したターン以降を**両席とも**落とす」。
# `on` は宣言した席の行だけを落とすので、優勢側の行が消えても劣勢側の鏡の行が残る（標本が席で非対称）。

def test_first_declared_turn_takes_the_min_over_both_seats():
    settled = {(7, 0, 1): False, (7, 1, 2): False, (7, 0, 5): True, (7, 1, 4): True, (7, 0, 7): True}
    assert LR.first_declared_turn(settled) == {7: 4}           # 席 1 の 4 が席 0 の 5 より早い


def test_first_declared_turn_omits_games_without_a_declaration():
    settled = {(1, 0, 1): False, (1, 1, 2): False, (2, 0, 3): True}
    out = LR.first_declared_turn(settled)
    assert out == {2: 3} and 1 not in out                        # 宣言の無い局は載らない＝1 行も落ちない


def test_first_declared_turn_is_a_pure_function_of_the_map():
    assert LR.first_declared_turn({}) == {}


# ---- 10. `avg_counter` の数え方（P8-7(b) 候補(c)・2026-09-25） ------------------------------------
#
# `EB01-028`（ゴムゴムのチャンピオン回転弾）は印字カウンター0だが【カウンター】能力でパワー+2000。
# `EB01-022`（イナズマ）は印字カウンター1000でイベント能力は無い。
# `printed`（既定）は前者を0として数え、`rules`（P8-7(b) 候補(c)）は【カウンター】の上げ幅も数える。

def test_avg_counter_printed_mode_ignores_the_counter_event_boost():
    from opcg_sim.learned.train import plan_labels as PL
    cards = PL.Cards()
    LR.set_avg_counter_mode("printed")
    assert LR.avg_counter(["EB01-028", "EB01-022"], cards) == pytest.approx(500.0)   # (0 + 1000) / 2


def test_avg_counter_rules_mode_counts_the_counter_event_boost():
    from opcg_sim.learned.train import plan_labels as PL
    cards = PL.Cards()
    LR.set_avg_counter_mode("rules")
    assert LR.avg_counter(["EB01-028", "EB01-022"], cards) == pytest.approx(1500.0)  # (2000 + 1000) / 2


def test_avg_counter_rules_mode_takes_the_larger_of_printed_and_event():
    from opcg_sim.learned.train import plan_labels as PL
    cards = PL.Cards()
    LR.set_avg_counter_mode("rules")
    # イナズマは印字1000・イベント無し(0) → max(1000, 0) = 1000 のまま(printedと同じ)
    assert LR.avg_counter(["EB01-022"], cards) == pytest.approx(1000.0)


def test_avg_counter_skips_unknown_card_ids_in_both_modes():
    from opcg_sim.learned.train import plan_labels as PL
    cards = PL.Cards()
    LR.set_avg_counter_mode("printed")
    assert LR.avg_counter(["__no_such_card__"], cards) == 0.0
    LR.set_avg_counter_mode("rules")
    assert LR.avg_counter(["__no_such_card__"], cards) == 0.0


def test_avg_counter_returns_zero_for_an_empty_deck():
    from opcg_sim.learned.train import plan_labels as PL
    cards = PL.Cards()
    for mode in LR.AVG_COUNTER_MODES:
        LR.set_avg_counter_mode(mode)
        assert LR.avg_counter([], cards) == 0.0
    LR.set_avg_counter_mode("printed")


def test_set_avg_counter_mode_rejects_unknown_modes():
    with pytest.raises(ValueError):
        LR.set_avg_counter_mode("bogus")
