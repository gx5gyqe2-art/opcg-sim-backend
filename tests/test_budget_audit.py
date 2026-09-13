"""`tests/scripts/budget_audit.py` の予算の算術と**窓の切り方**。

基盤健全性（`cpu_infra`）。要は 2 つ:
1. **窓を攻撃 1 回ごとに切れているか**（`kind==1` が入口・`kind==2` が同じ窓の追加・`PASS` で閉じる）。
   ここが崩れると「1 ターンで守り 1 回・受け 1 回」が潰れて `life_budget.md` の式と単位が合わない。
2. **`c(x)` の閾値が殴ってきた枠のパワーで測られているか**（最大パワーで測ると過大＝
   実測で `c_actual < c_min` が多発した。2026-09-13 にこれで計器を直した）。
"""
import json
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import budget_audit as B  # noqa: E402
from race_state import _T_PWR, _PWR_SCALE  # noqa: E402


class FakeCards:
    def __init__(self, table):
        self.table = table

    def info(self, cid):
        return self.table.get(cid)


def test_c_min_takes_the_largest_first():
    assert B.c_min([2000, 1000, 1000], 0) == 0
    assert B.c_min([2000, 1000, 1000], 1500) == 1
    assert B.c_min([1000, 1000, 1000], 2500) == 3
    assert B.c_min([2000, 1000], 4000) is None       # 止められない
    assert B.c_min([], 1000) is None
    assert B.c_min([], None) == 0


def test_budget_and_regimes():
    # L=3・H=5・A=2・T=3・B=0 → N=6・G=3・S=5+3+3−3·2=5 → 領域 2
    N, G, S = B.budget(L=3, H=5, A=2, T=3, B=0, cbar=2.0)
    assert (N, G) == (6.0, 3.0)
    assert S == pytest.approx(5.0)
    assert B.regime_of(G, S) == 2
    # 全部受けても生き残る（L が大きい）→ 領域 1
    _n, g1, s1 = B.budget(L=9, H=5, A=2, T=3, B=0, cbar=2.0)
    assert g1 == 0.0 and B.regime_of(g1, s1) == 1
    # 単価が高くて足りない → 領域 3
    _n, g3, s3 = B.budget(L=1, H=1, A=3, T=4, B=0, cbar=3.0)
    assert g3 > 0 and s3 < 0 and B.regime_of(g3, s3) == 3


def test_optimal_action_is_the_unit_price_comparison():
    assert B.optimal_action(1, 1, 2.0) == "take"          # 領域 1 は守るのが無駄
    assert B.optimal_action(2, 1, 2.0) == "guard"         # 安い＝守る
    assert B.optimal_action(2, 3, 2.0) == "take"          # 高い＝受ける
    assert B.optimal_action(2, None, 2.0) == "take"       # 止められない＝受けるしかない
    assert B.optimal_action(3, 1, 2.0) == "other"


def test_pressure_band_is_one_counter_card_per_step():
    assert B.pressure_band(None) is None
    assert B.pressure_band(-1000) == "p<=0"
    assert B.pressure_band(0) == "p<=0"
    assert B.pressure_band(1) == "p1k"
    assert B.pressure_band(2000) == "p2k"
    assert B.pressure_band(4001) == "p5k+"


def _tk(own_leader, opp_leader, chars=()):
    """race_state の 12 枠 × 5 列のうち、パワーと is_char だけ埋めた行。"""
    tk = np.zeros((12, 5), np.float32)
    tk[0, _T_PWR] = own_leader / _PWR_SCALE
    tk[1, _T_PWR] = opp_leader / _PWR_SCALE
    for j, p in enumerate(chars):
        tk[7 + j, _T_PWR] = p / _PWR_SCALE
        tk[7 + j, 4] = 1.0                                 # _T_CHAR
    return tk


def test_incoming_uses_the_attacker_that_actually_attacked():
    """最大パワー（7000 のキャラ）ではなく、殴ってきたリーダー（5000）で測る。"""
    tk = _tk(own_leader=5000, opp_leader=5000, chars=(7000,))
    assert B.incoming(tk, attacker_is_leader=True) == pytest.approx(0.0)
    assert B.incoming(tk, attacker_is_leader=False) == pytest.approx(2000.0)
    assert B.incoming(tk, None) == pytest.approx(2000.0)    # 分からないときは最大＝従来
    # キャラが居ないのに「キャラが殴った」なら測れない
    assert B.incoming(_tk(5000, 5000), attacker_is_leader=False) is None


def test_attackers_and_blockers_read_the_encoding():
    tok = np.zeros((22, 22), np.float32)
    tok[7, B.S_IS_CHAR] = 1.0
    tok[8, B.S_IS_CHAR] = 1.0
    assert B.attackers(tok) == 3                           # リーダー＋キャラ 2
    tok[2, B.S_BLOCKER] = 1.0
    tok[3, B.S_BLOCKER] = 1.0
    assert B.blockers(tok) == 2


def _sig(at, uuid=None, targets=()):
    return json.dumps([at, uuid, list(targets), []])


def _fake_game():
    """1 ターンで 2 回殴られ、1 回守って 1 回受けた対局（**ターン単位では take に潰れる形**）。"""
    #  p1（who=0）のターン 3 で 2 回攻撃・守るのは p2（who=1）
    recs = [
        # (who, turn, kind, sig)
        (0, 3, 0, _sig("DON_BOX", "atk1", ["ldrU"])),      # 1 回目の攻撃（リーダーへ）
        (1, 3, 1, _sig("SELECT_COUNTER", "c1")),           # 窓の入口＝守った
        (1, 3, 2, _sig("SELECT_COUNTER", "c2")),           # 同じ窓で 2 枚目
        (1, 3, 2, _sig("PASS")),                           # 窓を閉じる
        (0, 3, 0, _sig("DON_BOX", "atk2", ["ldrU"])),      # 2 回目の攻撃
        (1, 3, 1, _sig("PASS")),                           # 窓の入口で受けた
        (0, 3, 0, _sig("TURN_END")),
    ]
    n = len(recs)
    rows = {
        "who": np.array([r[0] for r in recs], np.int8),
        "turn": np.array([r[1] for r in recs], np.int16),
        "kind": np.array([r[2] for r in recs], np.int8),
        "sig": np.array([r[3] for r in recs]),
        "seed": np.zeros(n, np.int64),
        "z": np.array([1.0 if r[0] == 1 else -1.0 for r in recs], np.float32),
        "step": np.arange(n, dtype=np.int32),
        "pol_len": np.zeros(n, np.int32),
        "deck_kinds": np.array(['{"colors":["RED"],"templates":["t"],"rate":10}'] * n),
    }
    # 候補は攻撃の 1 本だけ（uuid → カード ID を作るのに要る）
    rows["pol_len"][0] = 1
    pol = {"pol_sig": np.array([_sig("DON_BOX", "atk1", ["ldrU"])]),
           "pol_cid": np.array(["CHR1"]), "pol_tcid": np.array(["LDR1"])}
    L = rows["pol_len"].astype(np.int64)
    ptr = np.concatenate([[0], np.cumsum(L)]).astype(np.int64)
    tok = np.zeros((n, 22, 22), np.float32)
    tok[:, 12, B.S_COUNTER] = 2000.0 / B.COUNTER_SCALE      # 手札に 2000 カウンター 1 枚
    tok[:, 13, B.S_COUNTER] = 1000.0 / B.COUNTER_SCALE
    tok[:, 12, 0] = tok[:, 13, 0] = 0.5                     # 枠が埋まっている印
    tok[:, 7, B.S_IS_CHAR] = 1.0                            # 相手の場にキャラ 1 体（A=2）
    sc = np.zeros((n, 127), np.float32)
    sc[:, B.SC_L] = 3.0
    sc[:, B.SC_H] = 5.0
    ex = {"tok": tok, "sc": sc,
          "ci": np.zeros((n, 22), np.int64),
          "race_tk": np.stack([_tk(5000, 5000, (7000,))] * n),
          "lives": np.full((n, 2), 3.0, np.float32)}
    idx = list(range(n))
    cards = FakeCards({"CHR1": {"leader": False, "removal": False, "blocker": False,
                                "counter": 0, "event": False, "cost": 2},
                       "LDR1": {"leader": True, "removal": False, "blocker": False,
                                "counter": 0, "event": False, "cost": 0}})
    return rows, pol, ex, L, ptr, idx, cards


def test_windows_are_cut_per_attack_not_per_turn():
    rows, pol, ex, L, ptr, idx, cards = _fake_game()
    wins = B.windows_of(rows, pol, ex, L, ptr, idx, cards, {}, with_events=False)
    assert len(wins) == 2, wins                            # **2 回の攻撃＝2 つの窓**
    first, second = wins
    assert first["c_actual"] == 2 and not first["blocked"]
    assert second["c_actual"] == 0
    assert first["target"] == second["target"] == "face"
    assert first["atk_leader"] is False                    # 殴ったのはキャラ（CHR1）
    # 殴ったのはキャラ（7000）なので閾値は 7000−5000＝2000 ＝ 2000 カウンター 1 枚で足りる
    assert first["x"] == pytest.approx(2000.0)
    assert first["c_min"] == 1
    assert first["L"] == 3.0 and first["H"] == 5.0
    assert first["A"] == 2 and first["B"] == 0             # リーダー＋キャラ 1・ブロッカー無し


def test_annotate_marks_the_played_action_and_the_greedy_gap():
    rows, pol, ex, L, ptr, idx, cards = _fake_game()
    wins = B.windows_of(rows, pol, ex, L, ptr, idx, cards, {}, with_events=False)
    table = {"_global": 2.0}
    B.annotate(wins, table)
    first, second = wins
    assert first["played"] == "guard" and second["played"] == "take"
    # 2000 を止めるのに 1 枚で足りたのに 2 枚払った＝払い過ぎ +1
    assert first["greedy_gap"] == 1
    assert second["greedy_gap"] is None                    # 守っていない窓は対象外
    assert first["optimal"] in ("take", "guard")
    assert first["agree"] == (first["played"] == first["optimal"])


def test_free_values_uses_the_printed_counter_when_the_card_is_known():
    """語彙を引ければ印字カウンターだけ・引けなければ符号化の値に落ちる（`cant_stop` の水増し防止）。"""
    tok = np.zeros((22, 22), np.float32)
    tok[12, 0] = 0.5
    tok[12, B.S_COUNTER] = 4000.0 / B.COUNTER_SCALE        # 符号化は 4000（イベントの上げ幅）
    ci = np.zeros(22, np.int64)
    ci[12] = 9
    cards = FakeCards({"EVT": {"leader": False, "removal": False, "blocker": False,
                               "counter": 0, "event": True, "cost": 2}})
    # 引ける＝印字 0 なので無料では払えない
    assert B._free_values(tok, ci, {9: "EVT"}, cards) == []
    # 引けない＝符号化の値に落ちる
    assert B._free_values(tok, ci, {}, cards) == [pytest.approx(4000.0)]


def test_cbar_table_falls_back_to_the_global_mean():
    wins = [{"turn": 3, "L": 3.0, "c_min": 2}, {"turn": 3, "L": 3.0, "c_min": 4}]
    table = B.cbar_table(wins)
    assert table["_global"] == pytest.approx(3.0)
    # 帯の n が 20 未満なので帯の値は作らず、global に落ちる
    assert B.cbar_of(table, wins[0]) == pytest.approx(3.0)


def test_theta_curve_and_by_deck_need_a_minimum_n():
    wins = [{"L": 3.0, "theta": 3.0, "played": "guard", "z": 1.0,
             "deck": {"colors": "RED", "templates": "t", "rate": 10},
             "cbar": 2.0, "c_actual": 1, "agree": True}] * 5
    assert B.theta_curve(wins) == {}                       # n<20 は出さない
    assert B.by_deck(wins, "colors") == {}                 # min_n 200
    assert B.by_deck(wins * 60, "colors")["RED"]["n"] == 300
