"""理論と探索の食い違いを解剖する算術（`tests/scripts/theory_vs_search.py`）。

基盤健全性（`cpu_infra`）。記録もネットも要らない純関数だけを固める。要は 4 つで、
**どれも 2026-09-13 に実際に間違えた箇所**である:

1. **攻撃は `DON_BOX`（対象付き）の形で来る**——記録に `ATTACK` という行動型は無い。
   `ATTACK` だけ見ていると攻撃を 1 件も数えられず、予言 `p3` が常に 0 になる。
2. **パワーは枠の現在値**——印字ではドンが付いたキャラを測れず、超過 `x` が狂う。
3. **理論の最良が同値の行は判定に使えない**（`th_tied`）——`argmax` は最小 index を
   返すだけなので、そのまま数えると**器の癖を「探索の食い違い」と誤読する**。
4. **探索の結論は訪問最大**（`i_n`）。`Q` 最大（`i_q`）を基準にした損は
   **勝者の呪いで上振れする**ので、本命の物差しと分けて出す。
"""
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

import theory_order as T  # noqa: E402
import theory_vs_search as V  # noqa: E402


class _Cards:
    def info(self, cid):
        return {"C_ATK": {"power": 7000, "cost": 3, "leader": False, "event": False},
                "C_LEAD": {"power": 5000, "cost": 0, "leader": True, "event": False}}.get(cid)


CTX = {"theta": 1.15, "mu": 0.05, "opp_leader_power": 5000.0,
       "my_leader_power": 5000.0, "r_turns": 3.0, "don_k": 1}


def _tok(**slots):
    t = np.zeros((22, 22), np.float32)
    for s, pw in slots.items():
        t[int(s), V.S_POWER] = pw / 1e4
    return t


def test_slot_power_reads_the_current_power_and_rounds_the_f32_dust():
    tok = _tok(**{"3": 7000.0})
    assert V.slot_power(tok, 3) == pytest.approx(7000.0)   # 0.7×1e4 = 6999.999… を吸う
    assert V.slot_power(tok, -1) is None                   # 枠が無い
    assert V.slot_power(tok, 999) is None                  # 範囲外


def test_a_don_box_with_a_target_is_an_attack():
    """**記録に `ATTACK` は出てこない**＝対象付きの `DON_BOX` を攻撃と読まないと数えられない。"""
    box = V.cand_detail(["DON_BOX", "u", ["t"], [], None], "C_ATK", "C_LEAD", 2, 8,
                        CTX, _Cards(), _tok(**{"2": 5000.0, "8": 5000.0}))
    assert box["is_attack"] is True and box["kind"] == "attack"
    # 枠の 5000 ＋ ドン 1 枚 − 相手 5000 ＝ 超過 1000（印字 7000 は使わない）
    assert box["x"] == pytest.approx(1000.0) and box["x_src"] == "slot"


def test_a_don_box_without_a_target_is_an_attach():
    bare = V.cand_detail(["DON_BOX", "u", [], [], None], "C_ATK", None, 2, -1,
                         CTX, _Cards(), _tok(**{"2": 6000.0}))
    assert bare["is_attack"] is False and bare["kind"] == "attach"
    assert bare["x"] == pytest.approx(1000.0)              # 付与は増分を見るので k を足さない


def test_printed_power_is_the_fallback_only():
    d = V.cand_detail(["DON_BOX", "u", [], [], None], "C_ATK", None, -1, -1,
                      CTX, _Cards(), _tok())
    assert d["x_src"] == "printed" and d["x"] == pytest.approx(2000.0)   # 7000 − 5000


def test_the_three_predictions_fire_where_the_theory_says_they_should():
    sat = T.saturation_x(1.15)
    assert sat == 2000.0
    zero = V.predictions_of({"at": "DON_BOX", "x": 0.0, "is_attack": False}, 1.15, 1)
    assert zero["p1_don_on_zero"] is True                  # c(0) = c(1000) ＝ 段を跨げない
    step = V.predictions_of({"at": "DON_BOX", "x": 1500.0, "is_attack": False}, 1.15, 1)
    assert step["p1_don_on_zero"] is False                 # 1500 → 2500 は段を跨ぐ
    over = V.predictions_of({"at": "DON_BOX", "x": sat + 1000.0, "is_attack": True}, 1.15, 1)
    assert over["p2_over_sat"] is True
    # **攻撃は DON_BOX で来る**＝`is_attack` で見る（`at == "ATTACK"` だと 0 件になる）
    dead = V.predictions_of({"at": "DON_BOX", "x": -1000.0, "is_attack": True}, 1.15, 1)
    assert dead["p3_cannot_connect"] is True
    attach = V.predictions_of({"at": "DON_BOX", "x": -1000.0, "is_attack": False}, 1.15, 1)
    assert attach["p3_cannot_connect"] is False            # 付与は「通る／通らない」を持たない
    assert V.predictions_of({"at": "PLAY", "x": None}, 1.15, 1)["p2_over_sat"] is False


def test_a_tied_theory_best_is_flagged_not_silently_argmaxed():
    """同値の最良を `argmax` で 1 本に潰すと**器の癖が食い違いとして数えられる**。"""
    n = [100.0, 100.0, 100.0]
    q = [0.1, 0.5, 0.3]
    tied = V.row_compare(n, q, [1.0, 1.0, 0.5], chosen=0, n_min=1, q_eps=0.0, n_min_frac=0.0)
    assert tied["th_tied"] is True
    clear = V.row_compare(n, q, [2.0, 1.0, 0.5], chosen=0, n_min=1, q_eps=0.0, n_min_frac=0.0)
    assert clear["th_tied"] is False and clear["i_th"] == 0


def test_visits_and_qmax_are_reported_as_separate_yardsticks():
    """選択は訪問で行う＝**探索の結論は訪問最大**。`Q` 最大の損は勝者の呪いで上振れする。"""
    n = [90.0, 10.0, 100.0]      # 訪問最大は index 2
    q = [0.2, 0.9, 0.3]          # Q 最大は index 1（訪問 10 の楽観）
    out = V.row_compare(n, q, [5.0, 1.0, 2.0], chosen=2, n_min=1, q_eps=0.0, n_min_frac=0.0)
    assert out["i_th"] == 0 and out["i_q"] == 1 and out["i_n"] == 2
    assert out["agree"] is False and out["agree_n"] is False
    assert out["q_loss"] == pytest.approx(0.7)             # 0.9 − 0.2（上振れした基準）
    assert out["q_loss_vs_n"] == pytest.approx(0.1)        # 0.3 − 0.2（訪問最大の基準）
    assert out["q_loss_vs_n"] < out["q_loss"]
    assert out["i_played"] == 2


def test_rows_without_two_scorable_candidates_are_refused():
    thin = V.row_compare([100.0, 100.0], [0.1, 0.2], [1.0, None], chosen=0,
                         n_min=1, q_eps=0.0, n_min_frac=0.0)
    assert "i_th" not in thin and thin["k_kept"] == 1


def _rec(seed, agree_n, tie, th_tied, at_th="attack", at_n="attach", z=1.0):
    return {"seed": seed, "band": "close", "turn": 3, "z": z, "agree": agree_n,
            "agree_n": agree_n, "tie": tie, "th_tied": th_tied, "at_th": at_th, "at_n": at_n,
            "i_q": 0, "i_n": 0 if agree_n else 1, "i_th": 0, "i_played": 0,
            "played_is_th": True, "q_loss": 0.05, "q_loss_vs_n": 0.04,
            "th_loss": 0.03, "th_loss_vs_n": 0.02,
            "q_loss_rand": 0.08, "th_loss_rand": 0.08}


def test_summary_excludes_tied_rows_from_every_judgement():
    from collections import Counter
    recs = [_rec(1, False, False, True), _rec(2, False, False, True),
            _rec(3, True, False, False), _rec(4, False, False, False)]
    out = V.summarise(recs, Counter({("attack", "attach"): 1}), {}, None, 0.02)
    assert out["n"] == 4 and out["n_judgeable"] == 2
    assert out["th_tied_share"] == pytest.approx(0.5)
    assert out["top1_th_vs_visits"]["mean"] == pytest.approx(0.5)   # 判定できる 2 行のうち 1 行
    # 分類の母数は**本物の食い違い**（同値の行・雑音床以下の行を除いた数）と揃う
    assert out["real_disagreement"] == 1
    assert out["cross_top"][0]["share"] == pytest.approx(1.0)


def test_summary_survives_an_all_tied_input():
    from collections import Counter
    out = V.summarise([_rec(1, False, False, True)], Counter(), {}, None, 0.02)
    assert out["n_judgeable"] == 0 and out["th_tied_share"] == 1.0
    assert V.summarise([], Counter(), {}, None, 0.02) == {"n": 0}


def test_the_loss_is_reported_against_a_random_candidate_baseline():
    """**損の数字は無作為な候補の損と比べて初めて読める**。

    理論が `Q` と無相関なだけでも `q_loss` は「最良 − 平均」まで大きく出るので、
    **平均からどれだけ最良側に寄れたか**（`recovered`）に直して読む。
    """
    n = [100.0, 50.0, 50.0]
    q = [0.5, 0.1, 0.0]           # 最良 0.5・平均 0.2
    out = V.row_compare(n, q, [1.0, 2.0, 0.5], chosen=0, n_min=1, q_eps=0.0, n_min_frac=0.0)
    assert out["i_th"] == 1 and out["i_n"] == 0
    assert out["q_loss_vs_n"] == pytest.approx(0.4)          # 0.5 − 0.1
    assert out["q_loss_rand"] == pytest.approx(0.3)          # 0.5 − 0.2（無作為の期待値）
    # 理論は無作為より**悪い**手を選んだ＝recovered は負になる
    from collections import Counter
    rec = {"seed": 1, "band": "close", "turn": 1, "z": 1.0, "agree": False, "agree_n": False,
           "tie": False, "th_tied": False, "at_th": "attack", "at_n": "attach",
           "i_q": 0, "i_n": 0, "i_th": 1, "i_played": 0, "played_is_th": False,
           "q_loss": 0.4, "q_loss_vs_n": 0.4, "th_loss": 0.0, "th_loss_vs_n": 0.0,
           "q_loss_rand": 0.3, "th_loss_rand": 0.3}
    s = V.summarise([rec, dict(rec, seed=2)], Counter(), {}, None, 0.02)
    assert s["q_recovered"] == pytest.approx(1.0 - 0.4 / 0.3, abs=1e-4)   # 出力は 4 桁で丸める
    assert s["q_recovered"] < 0
