"""攻撃順序の option value の算術（`tests/scripts/order_option.py`）。

基盤健全性（`cpu_infra`）。エンジンを呼ばない純関数だけを固める。要は 3 つ:

1. **先頭原始手で腕を畳む**——同じ枠へ 1 枚／3 枚付与する 2 つの箱はどちらも
   `ATTACH_DON(枠)` から始まる＝別の腕として数えると「腕の差 0」を自分で作ってしまう。
2. **`max − played` は勝者の呪いで上振れする**——腕が全部同じ強さでも max は played より上。
   判定は **A で選び B で測る** `opt_gap_honest`。
3. **雑音の下限**（同じ腕の |A−B|）に対して腕の差が小さければ判定を出さない。
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

import order_option as O  # noqa: E402


def _box(uuid, don_k, targets=()):
    return {"kind": "game", "action_type": "DON_BOX",
            "payload": {"uuid": uuid, "don_k": don_k, "target_ids": list(targets)}}


def _out(legal, groups, move=None):
    return {"stats": {"legal": legal}, "groups": groups, "move": move}


def test_arms_are_folded_by_the_first_primitive():
    """同じ枠への 1 枚／3 枚付与は**同じ入口**＝1 腕に畳み、訪問は和を取る。"""
    legal = [_box("A", 1), _box("A", 3), _box("B", 2)]
    groups = [{"rep": 0, "n": 10.0, "q": 0.1}, {"rep": 1, "n": 5.0, "q": 0.3},
              {"rep": 2, "n": 4.0, "q": 0.2}]
    cands = O.attack_candidates(_out(legal, groups))
    assert len(cands) == 2
    assert cands[0]["n"] == 15.0 and cands[0]["boxes"] == 2      # 枠 A は 10+5
    assert cands[0]["q"] == 0.3                                  # 畳んだ側は良い方の Q
    assert cands[1]["n"] == 4.0 and cands[1]["boxes"] == 1


def test_only_attack_entries_become_arms():
    legal = [{"kind": "game", "action_type": "PLAY", "payload": {"uuid": "C"}},
             {"kind": "game", "action_type": "TURN_END", "payload": {}},
             _box("A", 0, ["L"])]
    groups = [{"rep": 0, "n": 9.0}, {"rep": 1, "n": 1.0}, {"rep": 2, "n": 7.0}]
    cands = O.attack_candidates(_out(legal, groups))
    assert len(cands) == 1
    # 付与 0 枚で対象が在る箱の先頭原始手は **ATTACK**（ATTACH_DON ではない）
    assert cands[0]["move"]["action_type"] == "ATTACK"


def test_max_arms_keeps_the_most_visited():
    legal = [_box(u, 1) for u in "ABCDE"]
    groups = [{"rep": i, "n": float(i)} for i in range(5)]
    cands = O.attack_candidates(_out(legal, groups), max_arms=2)
    assert [c["n"] for c in cands] == [4.0, 3.0]


def test_played_sig_matches_the_folded_arm():
    """打った手（既に原始手）の鍵が、畳んだ腕の鍵と一致する。"""
    legal = [_box("A", 1), _box("B", 2)]
    groups = [{"rep": 0, "n": 10.0}, {"rep": 1, "n": 4.0}]
    move = {"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": "A"}}
    cands = O.attack_candidates(_out(legal, groups, move))
    assert O.played_sig(_out(legal, groups, move)) == cands[0]["sig"]
    assert O.played_sig(_out(legal, groups, None)) is None


def test_root_value_falls_back_to_visit_weighted_q():
    assert O.root_value({"v0": -0.4}) == pytest.approx(-0.4)
    out = {"groups": [{"n": 3.0, "q": 1.0}, {"n": 1.0, "q": -1.0}]}
    assert O.root_value(out) == pytest.approx(0.5)
    assert O.root_value({}) == 0.0


def test_v0_and_turn_bands():
    assert O.v0_band(0.2) == "close" and O.v0_band(-0.61) == "decided"
    assert O.turn_band(4) == "T<=4" and O.turn_band(9) == "T9+"


def _rows(n, honest, raw, rand, spread, noise, seed=None):
    return [{"opt_gap_honest": honest, "opt_gap_raw": raw, "gap_random": rand,
             "arm_spread": spread, "noise": noise, "argmax_is_played": False,
             "arms": 3, "seed": (i if seed is None else seed)} for i in range(n)]


def test_cluster_se_uses_games_not_points():
    """1 局から複数の点を採るので、**点の数で割ると SE を過小に見る**。"""
    rows = _rows(50, 0.4, 0, 0, 0.1, 0.01, seed=1) + _rows(50, -0.4, 0, 0, 0.1, 0.01, seed=2)
    cl = O.cluster_se(rows, "opt_gap_honest")
    assert cl["games"] == 2
    assert cl["se"] == pytest.approx(0.4, abs=1e-3)       # 局ごとの平均 ±0.4 → SE 0.4
    assert cl["ci95"][0] < 0 < cl["ci95"][1]
    assert O.cluster_se(_rows(3, 0.1, 0, 0, 0.1, 0.01, seed=1),
                        "opt_gap_honest")["se"] is None   # 1 局では出せない


def test_block_keeps_raw_and_honest_apart():
    b = O.block(_rows(4, 0.01, 0.08, -0.02, 0.09, 0.05))
    assert b["n"] == 4
    assert b["opt_gap_honest"] == pytest.approx(0.01)
    assert b["opt_gap_raw"] == pytest.approx(0.08)      # 上限は別に残す
    assert b["gap_random"] == pytest.approx(-0.02)
    assert b["spread_over_noise"] == pytest.approx(1.8)
    assert O.block([]) == {"n": 0}
    # played が腕に無い行は母数から外す
    assert O.block([{"arm_spread": 0.1, "noise": 0.01}]) == {"n": 0}


def test_verdict_refuses_to_judge_at_the_noise_floor():
    """腕の差が雑音と同程度なら、`opt_gap` が大きくても判定を出さない。"""
    noisy = O.block(_rows(6, 0.09, 0.2, 0.0, 0.06, 0.05))
    assert noisy["spread_over_noise"] == pytest.approx(1.2)
    assert O.verdict(noisy) == "noise_floor"


def test_verdict_needs_the_ci_to_clear_the_threshold():
    leak = O.block(_rows(30, 0.06, 0.2, 0.0, 0.30, 0.05))   # sd 0 → CI は点に潰れる
    assert O.verdict(leak) == "ordering_leaks"
    tiny = O.block(_rows(30, 0.001, 0.2, 0.0, 0.30, 0.05))
    assert O.verdict(tiny) == "no_option_value"
    flat = O.block(_rows(30, 0.0, 0.0, 0.0, 0.01, 0.001))
    assert O.verdict(flat) == "no_option_value"            # 腕の差そのものが無い
    assert O.verdict(None) is None and O.verdict({"n": 0}) is None
