"""ゲーム理論の 3 計器（零和の反対称・PIMC の融合・読まれやすさ）の算術。

基盤健全性（`cpu_infra`）。エンジンも記録も要らない純関数だけを固める。要は 3 つ:

1. **零和**: 判定は `sum` の**平均**（`E[V_p1]+E[V_p2]=0`）で、**1 局面ごとのずれは正常**
   （2 つの値は違う情報に条件付けた期待値）。標準誤差は**対局でクラスタ**する。
2. **融合**: 「束ねた選択がどの世界の最善でもない」が strategy fusion の直接の観測。
   世界の最善は**訪問最大**で、訪問 0 の世界は数に入れない。
3. **読まれやすさ**: 帯が細かいと `I` が上振れし予測器は帯を外す＝`bands_per_row` と
   `covered` を必ず併記し、`lift` は**検証側**で測る。
"""
import math
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import zero_sum_check as Z  # noqa: E402
import pimc_diag as P  # noqa: E402
import policy_entropy as Q  # noqa: E402
import numpy as np  # noqa: E402


# --- 零和の反対称 ---------------------------------------------------------------

def _zs(seed, sums):
    return [{"seed": seed, "sum": s, "v_p1": s / 2, "v_p2": s / 2,
             "turn": 3, "band": "T<=4", "to_move": "p1"} for s in sums]


def test_cluster_se_uses_games_not_rows():
    """同じ局の行は独立でない＝SE は**局数**で決まる（行数で割ってはいけない）。"""
    rows = _zs(1, [0.4] * 50) + _zs(2, [-0.4] * 50)
    out = Z.cluster_se(rows)
    assert out["games"] == 2
    # 局ごとの平均は +0.4 と −0.4 → 平均 0・sd(ddof=1)=0.5657 → SE=0.4
    assert out["sum_mean_se"] == pytest.approx(0.4, abs=1e-3)
    assert out["sum_mean_ci95"][0] < 0 < out["sum_mean_ci95"][1]


def test_cluster_se_needs_two_games():
    out = Z.cluster_se(_zs(1, [0.1, 0.2]))
    assert out["games"] == 1 and out["sum_mean_se"] is None


def test_verdict_is_about_the_mean_not_the_spread():
    """**ばらつきが大きくても平均が 0 なら零和は破れていない**（情報差は正常）。"""
    wide = _zs(1, [1.0] * 10) + _zs(2, [-1.0] * 10)     # |sum| は大きいが平均 0
    b = Z.block(wide)
    assert b["abs_mean"] == pytest.approx(1.0)
    assert Z.verdict(b) == "antisymmetric"
    biased = _zs(1, [-0.3] * 10) + _zs(2, [-0.28] * 10) + _zs(3, [-0.32] * 10)
    assert Z.verdict(Z.block(biased)) == "biased"
    assert Z.verdict(Z.block(_zs(1, [0.0]))) == "not_enough_games"
    assert Z.verdict(None) is None


def test_blind_hand_zeroes_only_the_hand_slots():
    s_dim = 22
    enc = {"tok": [1.0] * (22 * s_dim), "scalars": [3.0] * 14}
    out = Z.blind_hand(enc)
    tok = out["tok"]
    # 手札は枠 12〜21・それ以外は触らない
    assert all(tok[slot * s_dim + k] == 0.0
               for slot in range(12, 22) for k in range(s_dim))
    assert all(tok[slot * s_dim] == 1.0 for slot in range(0, 12))
    assert out["scalars"][Z.SC_MY_HAND] == 0.0
    assert enc["tok"][12 * s_dim] == 1.0                 # 入力は書き換えない


def test_to_net_enc_renames_tokens_to_tok():
    """`encode_state` は `tokens`・`net_eval` は `tok`（黙って 0 要素になる穴）。"""
    d = Z.to_net_enc('{"tokens": [1, 2], "scalars": [0]}')
    assert d["tok"] == [1, 2] and "tokens" not in d
    keep = Z.to_net_enc('{"tok": [3], "scalars": [0]}')
    assert keep["tok"] == [3]


# --- PIMC の融合 ---------------------------------------------------------------

def _out(per_world, groups):
    return {"per_world": per_world, "groups": groups,
            "stats": {"legal": [{}] * 3}}


def test_world_best_ignores_worlds_with_no_visits():
    assert P.world_best({"N": [0.0, 0.0]}) is None
    assert P.world_best({"N": [1.0, 5.0, 2.0]}) == 1
    assert P.world_best({}) is None


def test_fusion_detects_the_merged_choice_that_no_world_liked():
    """世界 0 は手 0・世界 1 は手 1 が最善だが、束ねた選択は手 2＝**融合の直接の観測**。"""
    pw = [{"N": [10.0, 1.0, 6.0], "Q": [0.5, 0.1, 0.2]},
          {"N": [1.0, 10.0, 6.0], "Q": [0.1, 0.5, 0.3]}]
    groups = [{"rep": 0, "n": 11.0, "q": 0.3}, {"rep": 1, "n": 11.0, "q": 0.3},
              {"rep": 2, "n": 12.0, "q": 0.25}]
    row = P.fusion_of(_out(pw, groups))
    assert row["disagree"] is True and row["n_distinct_best"] == 2
    assert row["merged_novel"] is True
    assert row["q_spread"] == pytest.approx(0.1)          # 手 2 の Q は 0.2 と 0.3
    assert row["worlds"] == 2 and row["unmapped"] == 0


def test_fusion_when_every_world_agrees():
    pw = [{"N": [9.0, 1.0], "Q": [0.4, 0.1]}, {"N": [8.0, 2.0], "Q": [0.42, 0.1]}]
    groups = [{"rep": 0, "n": 17.0, "q": 0.41}, {"rep": 1, "n": 3.0, "q": 0.1}]
    row = P.fusion_of(_out(pw, groups))
    assert row["disagree"] is False and row["merged_novel"] is False
    assert row["q_spread"] == pytest.approx(0.02)


def test_fusion_needs_two_usable_worlds():
    assert P.fusion_of(_out([{"N": [1.0]}], [{"rep": 0, "n": 1.0}])) is None
    assert P.fusion_of(_out([{"N": [0.0]}, {"N": [0.0]}], [{"rep": 0, "n": 1.0}])) is None


def test_block_pools_the_rates():
    rows = [{"disagree": True, "merged_novel": False, "n_distinct_best": 2, "q_spread": 0.2,
             "q_sd": 0.1, "k_legal": 5, "n_cands": 5, "unmapped": 0},
            {"disagree": False, "merged_novel": False, "n_distinct_best": 1, "q_spread": 0.0,
             "q_sd": 0.0, "k_legal": 3, "n_cands": 3, "unmapped": 1}]
    b = P.block(rows)
    assert b["n"] == 2 and b["disagree"] == 0.5 and b["merged_novel"] == 0.0
    assert b["q_spread_mean"] == pytest.approx(0.1)
    assert b["unmapped_total"] == 1
    assert P.block([]) is None


def test_v0_band_edges():
    assert P.v0_band(0.0) == "close" and P.v0_band(-0.2) == "close"
    assert P.v0_band(0.5) == "mid" and P.v0_band(0.61) == "decided"


# --- 読まれやすさ ---------------------------------------------------------------

def test_entropy_and_mutual_information():
    import collections
    assert Q.entropy(collections.Counter()) == 0.0
    assert Q.entropy(collections.Counter({"a": 1, "b": 1})) == pytest.approx(1.0)
    assert Q.entropy(collections.Counter({"a": 4})) == 0.0
    # 帯が行動を完全に決める → H_cond = 0・I = H
    rows = ([{"band": "A", "action": "guard", "seed": 0}] * 10
            + [{"band": "B", "action": "take", "seed": 1}] * 10)
    d = Q.conditional_entropy(rows)
    assert d["H"] == pytest.approx(1.0)
    assert d["H_cond"] == pytest.approx(0.0)
    assert d["I_bits"] == pytest.approx(1.0)
    # 補正は帯の数だけ引く（2 帯 × 1 自由度 / (2·20·ln2)）
    assert d["I_bits_corrected"] == pytest.approx(1.0 - 2 / (2 * 20 * math.log(2)), abs=1e-4)
    assert d["bands_per_row"] == pytest.approx(0.1)


def test_mutual_information_is_zero_when_the_band_says_nothing():
    rows = []
    for b in ("A", "B"):
        rows += [{"band": b, "action": "guard", "seed": 0}] * 5
        rows += [{"band": b, "action": "take", "seed": 1}] * 5
    d = Q.conditional_entropy(rows)
    assert d["I_bits"] == pytest.approx(0.0)


def test_majority_predictor_is_scored_on_held_out_games():
    train = [{"band": "A", "action": "guard", "seed": 0}] * 8 + \
            [{"band": "A", "action": "take", "seed": 2}] * 2 + \
            [{"band": "B", "action": "take", "seed": 4}] * 10
    test = [{"band": "A", "action": "guard", "seed": 1}] * 5 + \
           [{"band": "B", "action": "take", "seed": 3}] * 5
    p = Q.majority_predictor(train, test)
    assert p["acc"] == pytest.approx(1.0)
    assert p["base_rate"] == pytest.approx(0.5)
    assert p["lift"] == pytest.approx(0.5)
    assert p["covered"] == pytest.approx(1.0)
    # 訓練に無い帯は全体の多数決に落ちる＝covered が下がる
    p2 = Q.majority_predictor(train, [{"band": "Z", "action": "guard", "seed": 5}])
    assert p2["covered"] == 0.0 and p2["acc"] == 0.0      # 全体の多数決は take
    assert Q.majority_predictor([], test) is None


def test_public_band_levels_are_nested_and_public_only():
    sc = np.zeros(14, np.float32)
    sc[Q.SC_L], sc[Q.SC_OPP_L], sc[Q.SC_H] = 3, 5, 6
    sc[Q.SC_OPP_H], sc[Q.SC_FIELD], sc[Q.SC_OPP_FIELD], sc[Q.SC_TURN] = 4, 2, 1, 7
    b_min = Q.public_band(sc, 2000.0, "min")
    b_mid = Q.public_band(sc, 2000.0, "mid")
    b_full = Q.public_band(sc, 2000.0, "full")
    assert b_min == "3|x2k"
    assert b_mid.startswith(b_min) and "h6+" in b_mid and "T5-8" in b_mid
    assert b_full.startswith(b_mid)
    # 攻めの帯（超過パワーなし）は相手ライフが 2 番目に来る
    assert Q.public_band(sc, None, "min") == "3|5"


def test_x_band_tolerates_f16_rounding():
    assert Q.x_band(None) == "none"
    assert Q.x_band(2000.0002) == "x2k"
    assert Q.x_band(1.0) == "x<=0"


def test_verdict_thresholds():
    assert Q.verdict({"predictor": {"lift": 0.2}}) == "readable"
    assert Q.verdict({"predictor": {"lift": 0.01}}) == "opaque"
    assert Q.verdict({"predictor": {"lift": 0.09}}) == "partly"
    assert Q.verdict({"predictor": {"lift": None}}) is None
    assert Q.verdict(None) is None
