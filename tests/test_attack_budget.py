"""`tests/scripts/attack_budget.py`（攻め側の予算＝掛けた圧力とドンの使い切り）の算術。

基盤健全性（`cpu_infra`）。要は **`don_idle` の出し方**——**状態から直接読む**
（自席ターンの最後の main 行のアクティブなドン）。

> **2026-09-14 にこのテストは書き直した**（`docs/reports/2026-09-14_don_accounting.md`）。
> 旧版は `don_idle == 6 − 1 − 4` という**再構成の定義を固定していた**＝**バグを守るテスト**
> だった。`ATTACH_DON` は選ばれた手として一度も現れない（付与は全部 `DON_BOX` の中）ので、
> 再構成は付与を丸ごと余りに計上し、**実データで余りを 4 倍に見せていた**（2.07 対 真値 0.51）。
> いまラチェットするのは **(1) 余りは状態から読む・(2) 箱の付与は `pol_k` で数える
> （的の無い箱も）・(3) 恒等式 `don = 付与 + コスト + 余り` が閉じる**の 3 点。
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
    """自席ターン 3（who=0）: ドン 6 で 4 コスト 1 枚・箱で 1 個付与しリーダーへ・キャラへ 1 回。

    **ドンは行ごとに減る**（6 → 出して 2 → 付与して 1）＝`don_idle` は
    **最後の main 行のアクティブなドン**を直接読むので 1。
    `pol_k` に箱の付与枚数を入れる（付与は**箱の中でしか起きない**ので、これが実際の観測）。
    """
    #                                                       don_active, pol_k
    recs = [
        (0, 3, 0, _sig("PLAY", "chr1"), 6, -1),
        (0, 3, 0, _sig("DON_BOX", "atk1", ["ldrU"]), 2, 1),   # 4 払った後・1 個乗せる
        (0, 3, 2, _sig("ATTACH_DON", "atk1"), 1, -1),          # 箱の中の付与の行
        (0, 3, 2, _sig("ATTACK", "atk1", ["ldrU"]), 1, -1),    # 箱の中＝同じ攻撃の続き
        (0, 3, 0, _sig("DON_BOX", "atk1", ["oppChrU"]), 1, 0),  # キャラへ＝board・付与なし
        (0, 3, 0, _sig("TURN_END"), 1, -1),                    # ここの 1 が余らせたドン
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
        "pol_len": np.ones(n, np.int32),
        "pol_chosen": np.zeros(n, np.int32),
        "deck_kinds": np.array(["{}"] * n),
    }
    sc = np.zeros((n, 14), np.float32)
    sc[:, A.SC_L] = 4.0
    sc[:, A.SC_OPP_L] = 3.0
    sc[:, A.SC_DON] = np.array([r[4] for r in recs], np.float32)
    ex = {"sc": sc, "tk": np.stack([_tk((6000, True), 5000, ((5000, True),))] * n)}
    L = rows["pol_len"].astype(np.int64)
    ptr = np.concatenate([[0], np.cumsum(L)]).astype(np.int64)
    pol = {"pol_k": np.array([r[5] for r in recs], np.int32)}
    u2c = {"chr1": "C4", "atk1": "CHR", "ldrU": "LDR", "oppChrU": "CHR"}
    cards = FakeCards({
        "C4": {"leader": False, "cost": 4, "counter": 0, "event": False,
               "removal": False, "blocker": False},
        "CHR": {"leader": False, "cost": 3, "counter": 1000, "event": False,
                "removal": False, "blocker": False},
        "LDR": {"leader": True, "cost": 0, "counter": 0, "event": False,
                "removal": False, "blocker": False},
    })
    return rows, ex, L, ptr, list(range(n)), cards, u2c, pol


def test_turns_of_separates_attach_play_and_board_attacks():
    rows, ex, L, ptr, idx, cards, u2c, pol = _fake_game()
    ts = A.turns_of(rows, ex, L, ptr, idx, cards, u2c, pol)
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


def test_the_box_attachment_is_counted_from_pol_k():
    """**付与は `DON_BOX` の中でしか起きない**ので `pol_k` を数えないと付与が丸ごと抜ける。

    実測で `ATTACH_DON` は**選ばれた手として一度も現れない**（2026-09-14・300 局 約 1.9 万手で
    0 件）。`ATTACH_DON` の**行**は箱の中に在るが付与の一部しか捉えない（実測 1.32 対 2.94）。
    """
    rows, ex, L, ptr, idx, cards, u2c, pol = _fake_game()
    t = A.turns_of(rows, ex, L, ptr, idx, cards, u2c, pol)[0]
    assert t["don_box_attached"] == 1             # リーダーへの箱で 1 個乗せた
    # **的の無い箱（純粋な付与）でも数える**＝`continue` より前に置いてある
    rows2, ex2, L2, ptr2, idx2, cards2, u2c2, pol2 = _fake_game()
    rows2["sig"][1] = _sig("DON_BOX", "atk1")     # 的を外す（配分だけの箱）
    t2 = A.turns_of(rows2, ex2, L2, ptr2, idx2, cards2, u2c2, pol2)[0]
    assert t2["don_box_attached"] == 1
    assert t2["attacks_made"] == 0                # 的が無いので攻撃ではない
    # `pol` を渡さなければ 0（後方互換）
    t3 = A.turns_of(rows, ex, L, ptr, idx, cards, u2c)[0]
    assert t3["don_box_attached"] == 0


def test_block_reads_the_unused_don_from_the_state_not_from_the_actions():
    """**`don_idle` は最後の main 行のアクティブなドンを直接読む**。

    2026-09-14 の訂正: 以前は `don − ATTACH_DON の行数 − PLAY のコスト` で再構成していたが、
    **付与が全部 `DON_BOX` の中で起きる**ので付与が抜け、**余りを 3 倍に見せていた**
    （実測 2.07 対 真値 0.51）。**状態が答えを持っているものを行動の再構成で出さない**。
    """
    rows, ex, L, ptr, idx, cards, u2c, pol = _fake_game()
    b = A.block(A.turns_of(rows, ex, L, ptr, idx, cards, u2c, pol))
    assert b["don"] == 6.0
    assert b["don_plays"] == 4.0
    assert b["don_box_attached"] == 1.0
    assert b["don_idle"] == 1.0                   # TURN_END 行のアクティブなドン（直接読み）
    assert b["don_idle_pos"] == 1.0
    # 会計が閉じる: 6 = 付与 1 + コスト 4 + 余り 1
    assert b["don_unaccounted"] == pytest.approx(0.0)
    # 旧い再構成も並記する（同じ罠を二度踏まないため）＝付与が抜けるので 1 個多く出る
    assert b["don_idle_recon_wrong"] == 1.0       # 6 − 1（ATTACH_DON 行）− 4
    assert b["force_gap"] == pytest.approx(b["force_max"] - b["force_actual"])
    assert A.block([]) is None


def test_the_unused_don_is_not_inflated_when_the_box_attaches_many():
    """箱が 3 個乗せたターンは**再構成だけが余りを水増しする**（直接読みは正しい）。"""
    rows, ex, L, ptr, idx, cards, u2c, pol = _fake_game()
    pol["pol_k"][1] = 3                            # 箱で 3 個乗せた
    ex["sc"][2:, A.SC_DON] = 0.0                   # 付与し切ってドンは残っていない
    b = A.block(A.turns_of(rows, ex, L, ptr, idx, cards, u2c, pol))
    assert b["don_idle"] == 0.0                    # 直接読み＝余っていない
    assert b["don_idle_recon_wrong"] == 1.0        # 再構成は 1 個余ったと言う（誤り）
    assert b["don_unaccounted"] == pytest.approx(-1.0)   # 6 − 3 − 4 − 0（払い過ぎ＝途中で得た）


def test_x_dist_shares_sum_to_one():
    recs = [{"x_list": [1000.0, 3000.0, -500.0]}, {"x_list": [6000.0]}]
    d = A.x_dist(recs)
    assert d["x1k"]["n"] == 1 and d["x3k"]["n"] == 1
    assert d["x<=0"]["n"] == 1 and d["x5k+"]["n"] == 1
    assert sum(v["share"] for v in d.values()) == pytest.approx(1.0)
