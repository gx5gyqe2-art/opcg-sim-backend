"""**手札の守る備え `G_guard`**（T67・2026-09-16）——出す計画 `H_play`（T66）と対になる、手札のもう一つの価値。

ユーザ指摘 2026-09-16「カウンター値を増やすために 2000 カウンターやイベントカウンターを加えることもある」。
手札の札は出すだけでなく**守りに切る**。守る備えは「来る攻撃を、受けるより安く止められる分」:

```
G_guard(手札) = Σ_t s^t Σ_{t に来る攻撃 x} max(0, 受ける損 − 守る費用(手札, x))
守る費用(手札, x) = カウンター合計 ≥ x + 1000 になる札の組のうち v の和が最小のもの（T64・同値は命中＝上回る合計・T61）
来る攻撃      = 相手の場の攻撃手（リーダー＋キャラ）が自分のリーダーに来る超過 x（`incoming_x`）
受ける損      = theta_take(自ライフ)·μ（T63・ライフ 0 なら勝利の価値 0.5）
```

札は 1 回しか切れないので、攻撃には大きい順に、残った札の中から最小の組を当てる（貪欲）。`ΔG(札) = G(手札 ∪ 札) − G(手札)`。
2000 カウンターやイベントカウンターは `v`（切っても惜しくない）が低くカウンター値が高いので `ΔG` が立つ。
**回帰しない・当てはめない**（新定数ゼロ）。
"""
import itertools
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import guard_afford as GA  # noqa: E402
import theory_order as TO  # noqa: E402
from theory_order import KO_P, MU, PWR_EPS, incoming_x  # noqa: E402

#: 守りの地平（相手のターン 2 回ぶん・割引は `s^t`）
GUARD_TURNS = 2
#: 手札の上限（規則）＝部分集合の列挙が 2^10 で済む
HAND_MAX = 10


def counter_of(tok_row, slot):
    """手札の枠のカウンター値（印字＋イベントの上げ幅・`guard_afford.COUNTER_SCALE`）。"""
    return float(np.asarray(tok_row)[slot, GA.S_COUNTER]) * GA.COUNTER_SCALE


def guard_cost_min_v(items, x):
    """超過 `x` を止める最小の `v` の和（`items` = [(counter, v), …]・止められなければ `None`）。
    `x < 0` は止める必要が無い（0）。同値は命中するので合計は `x + 1000` 以上が要る。"""
    x = float(x)
    if x < -PWR_EPS:
        return 0.0, ()
    need = x + 1000.0 - PWR_EPS
    best, best_idx = None, ()
    n = min(len(items), HAND_MAX)
    for r in range(1, n + 1):
        for idx in itertools.combinations(range(n), r):
            if sum(items[i][0] for i in idx) < need:
                continue
            c = sum(0.0 if items[i][1] is None else float(items[i][1]) for i in idx)
            if best is None or c < best:
                best, best_idx = c, idx
    return best, best_idx


def guard_value(items, xs, take_cost, s=1.0 - KO_P, turns=GUARD_TURNS):
    """**守る備え**＝来る攻撃（`xs`・毎ターン同じと置く）を受けるより安く止められる分の和。札は地平の中で 1 回だけ。"""
    items = list(items)
    total = 0.0
    for t in range(turns):
        for x in sorted(xs, reverse=True):                 # 大きい攻撃から当てる
            if x < -PWR_EPS:
                continue
            cost, idx = guard_cost_min_v(items, x)
            if cost is None:
                continue                                    # 止められない＝受ける（節約 0）
            saving = float(take_cost) - float(cost)
            if saving <= 0.0:
                continue                                    # 受ける方が安い
            total += (s ** t) * saving
            items = [it for i, it in enumerate(items) if i not in idx]
    return float(total)


def delta_g(items, extra, xs, take_cost, s=1.0 - KO_P, turns=GUARD_TURNS):
    """`ΔG(札)` = 札を足した備え − 元の備え（≥ 0）。"""
    return guard_value(list(items) + [extra], xs, take_cost, s, turns) - guard_value(items, xs, take_cost, s, turns)


def incoming(tok_row):
    """来る攻撃の超過 `x` の並び（通らない攻撃は落とす）。"""
    return [x for x in incoming_x(tok_row) if x >= -PWR_EPS]


def take_cost_of(my_life, mu=MU):
    return float(TO.theta_take(my_life)) * float(mu)
