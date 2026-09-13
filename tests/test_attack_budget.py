"""`tests/scripts/attack_budget.py`（攻め側の予算＝掛けた圧力とドンの使い切り）の算術。

基盤健全性（`cpu_infra`）。要は **`don_idle` の分解**——手持ちのドンから「付与した分」と
「場に出すのに払った分」を引かないと「余らせた」を大きく数え過ぎる（`don_unused` 4.9 →
`don_idle` 1.9 に変わった）。ここを取り違えると「付与が苦手」の量が水増しになる。
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

import attack_budget as A  # noqa: E402
from race_state import _T_PWR, _T_CAN, _T_CHAR, _PWR_SCALE  # noqa: E402


class FakeCards:
    def __init__(self, table):
        self.table = table

    def info(self, cid):
        return self.table.get(cid)


def _tk(own_leader=(5000, True), opp_leader=5000, own_chars=()):
    """race_state の 12 枠 × 5 列（パワー・攻撃可否・キャラ旗だけ）。"""
    tk = np.zeros((12, 5), np.float32)
    tk[0, _T_PWR] = own_leader[0] / _PWR_SCALE
    tk[0, _T_CAN] = 1.0 if own_leader[1] else 0.0
    tk[1, _T_PWR] = opp_leader / _PWR_SCALE
    for j, (p, can) in enumerate(own_chars):
        tk[2 + j, _T_PWR] = p / _PWR_SCALE
        tk[2 + j, _T_CAN] = 1.0 if can else 0.0
        tk[2 + j, _T_CHAR] = 1.0
    return tk


def test_force_is_ceil_in_1000_units():
    assert A.force(None) == 0
    assert A.force(0) == 0
    assert A.force(1000) == 1
    assert A.force(1500) == 2
    assert A.force(5000) == 5


def test_force_tolerates_f16_rounding():
    """`power/10000` を f16 で戻すと 6000 が 6000.0002 になる。**許容が無いと +1 される**
    （ちょうど 1000 の倍数が実データの大半なので、系統的な水増しになる）。"""
    assert A.force(1000.0002) == 1
    assert A.force(3000.0004) == 3
    assert A.force(1.0) == 0                     # 1 パワーの超過は実在しない
    assert A.x_band(2000.0002) == "x2k"


def test_x_band_edges():
    assert A.x_band(None) is None
    assert A.x_band(-500) == "x<=0"
    assert A.x_band(1000) == "x1k"
    assert A.x_band(4000) == "x4k"
    assert A.x_band(4500) == "x5k+"


def test_my_attackers_only_counts_slots_that_can_attack():
    tk = _tk(own_leader=(5000, True), own_chars=((6000, True), (3000, False)))
    assert sorted(A.my_attackers(tk)) == pytest.approx([5000.0, 6000.0], abs=1.0)
    # リーダーが既に殴っていれば入らない
    tk2 = _tk(own_leader=(5000, False), own_chars=((6000, True),))
    assert A.my_attackers(tk2) == pytest.approx([6000.0], abs=1.0)
    assert A.opp_leader_power(tk) == pytest.approx(5000.0, abs=1.0)


def test_max_force_adds_one_card_per_don_when_something_already_reaches():
    # 6000 vs 5000 → base 1 枚・ドン 3 個で 4 枚
    total, base, don = A.max_force([6000.0], 5000.0, 3)
    assert (total, base, don) == (4, 1, 3)


def test_max_force_wastes_don_that_only_closes_the_gap():
    """どの枠も届かないとき、届かせるまでのドンは force を増やさない。"""
    # 3000 vs 5000（−2000）→ ドン 2 個で±0・3 個目で初めて 1 枚
    total, base, _don = A.max_force([3000.0], 5000.0, 3)
    assert base == 0
    assert total == 1
    total0, _b, _d = A.max_force([3000.0], 5000.0, 1)
    assert total0 == 0
    assert A.max_force([], 5000.0, 5) == (0, 0, 0)


def _sig(at, uuid=None, targets=()):
    return json.dumps([at, uuid, list(targets), []])


def _fake_game():
    """自席ターン 3（who=0）: 4 コストを 1 枚出し・ドン 1 個付与・リーダーへ 1 回・キャラへ 1 回。"""
    recs = [
        (0, 3, 0, _sig("PLAY", "chr1")),
        (0, 3, 0, _sig("DON_BOX", "atk1", ["ldrU"])),
        (0, 3, 2, _sig("ATTACH_DON", "atk1")),
        (0, 3, 2, _sig("ATTACK", "atk1", ["ldrU"])),          # 箱の中＝同じ攻撃の続き
        (0, 3, 0, _sig("DON_BOX", "atk1", ["oppChrU"])),   # キャラへの攻撃＝board
        (0, 3, 0, _sig("TURN_END")),
    ]
    n = len(recs)
    rows = {
        "who": np.array([r[0] for r in recs], np.int8),
        "turn": np.array([r[1] for r in recs], np.int16),
        "kind": np.array([r[2] for r in recs], np.int8),
        "sig": np.array([r[3] for r in recs]),
        "seed": np.zeros(n, np.int64),
        "z": np.ones(n, np.float32),
        "step": np.arange(n, dtype=np.int32),
        "pol_len": np.zeros(n, np.int32),
        "deck_kinds": np.array(["{}"] * n),
    }
    sc = np.zeros((n, 14), np.float32)
    sc[:, A.SC_L] = 4.0
    sc[:, A.SC_OPP_L] = 3.0
    sc[:, A.SC_DON] = 6.0
    ex = {"sc": sc, "tk": np.stack([_tk((6000, True), 5000, ((5000, True),))] * n)}
    L = rows["pol_len"].astype(np.int64)
    ptr = np.concatenate([[0], np.cumsum(L)]).astype(np.int64)
    u2c = {"chr1": "C4", "atk1": "CHR", "ldrU": "LDR", "oppChrU": "CHR"}
    cards = FakeCards({
        "C4": {"leader": False, "cost": 4, "counter": 0, "event": False,
               "removal": False, "blocker": False},
        "CHR": {"leader": False, "cost": 3, "counter": 1000, "event": False,
                "removal": False, "blocker": False},
        "LDR": {"leader": True, "cost": 0, "counter": 0, "event": False,
                "removal": False, "blocker": False},
    })
    return rows, ex, L, ptr, list(range(n)), cards, u2c


def test_turns_of_separates_attach_play_and_board_attacks():
    rows, ex, L, ptr, idx, cards, u2c = _fake_game()
    ts = A.turns_of(rows, ex, L, ptr, idx, cards, u2c)
    assert len(ts) == 1
    t = ts[0]
    # 箱の main 行 1 本だけを数える（中の `ATTACK` 行は同じ攻撃の続き）
    assert t["attacks_made"] == 1
    assert t["board_attacks"] == 1                # キャラへの攻撃は別枠
    assert t["don_attached"] == 1
    assert t["plays"] == 1 and t["don_plays"] == 4
    assert t["attacks_available"] == 2            # リーダー＋キャラ 1
    # 6000 vs 5000 → 1 枚ぶん・base 1（リーダー）+1（キャラ 5000 は ±0 なので 0）
    assert t["force_actual"] == 1
    assert t["don"] == 6


def test_block_decomposes_the_unused_don():
    """`don_idle` は付与と支払いを引いた残り（引かないと水増しになる）。"""
    rows, ex, L, ptr, idx, cards, u2c = _fake_game()
    ts = A.turns_of(rows, ex, L, ptr, idx, cards, u2c)
    b = A.block(ts)
    assert b["don"] == 6.0
    assert b["don_attached"] == 1.0
    assert b["don_plays"] == 4.0
    assert b["don_idle"] == 1.0                   # 6 − 1 − 4
    assert b["don_idle_pos"] == 1.0
    assert b["force_gap"] == pytest.approx(b["force_max"] - b["force_actual"])
    assert A.block([]) is None


def test_x_dist_shares_sum_to_one():
    recs = [{"x_list": [1000.0, 3000.0, -500.0]}, {"x_list": [6000.0]}]
    d = A.x_dist(recs)
    assert d["x1k"]["n"] == 1 and d["x3k"]["n"] == 1
    assert d["x<=0"]["n"] == 1 and d["x5k+"]["n"] == 1
    assert sum(v["share"] for v in d.values()) == pytest.approx(1.0)
