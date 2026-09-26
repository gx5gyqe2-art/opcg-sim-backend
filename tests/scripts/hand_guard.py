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


def guard_value_exact(items, xs, take_cost, s=1.0 - KO_P, turns=GUARD_TURNS, start=0):
    """**守る備えの厳密な最大値**（G-2 の修正・2026-09-26）——`guard_value` と**同じ目的・同じ割引・同じ地平**で、
    貪欲な割り当てではなく**全部の割り当ての最大**を取る:

        max over 互いに素な札の組の割り当て {(t, x) → S}:  Σ s^(start+t) · (受ける損 − Σ_{i∈S} v_i)
        ただし各組 S はカウンター合計 ≥ x + 1000（同値は命中）・割り当てない攻撃は受ける（節約 0）

    `items` = [(counter, v), …]（`v` の `None` は 0・`guard_cost_min_v` と同じ）・`xs` は来る攻撃（毎ターン同じと置く）。
    `start` は最初の相手ターンの割引の指数（`guard_value` は 0。**G-2 の判断では 1**＝今の窓から 1 ラウンド後）。

    * **札が増えて下がることが無い**（手札が大きいほど選べる割り当てが増えるだけ）＝`guard_value`（貪欲）の癖が無い。
    * 数えるのはカウンター値が正の札だけ（カウンター 0 の札は組を足りさせず、`v ≥ 0` なら組に入れて得をしない）。
    * `v ≥ 0` なら各組は**過不足の無い組**（どの札を抜いても足りない）だけ調べれば足りる（余計な札は `Σv` を増やすだけ）。
      負の `v` が在るときは足りる組を全部調べる（`use_value` は 0 で床を打つので記録の行では起きない）。
    * 計算は「攻撃の並びの何本目まで決めたか × 残りの札」の再帰（メモ化）＝札 k 枚で状態は高々 2^k・攻撃ごと。
    **新定数ゼロ**。`guard_value` は他の器が使うので変えない。"""
    take = float(take_cost)
    cards = [(float(c), (0.0 if v is None else float(v))) for c, v in items]
    cards = [cv for cv in cards if cv[0] > 0.0]
    k = len(cards)
    atks = [(t, float(x)) for t in range(int(turns)) for x in sorted(xs, reverse=True) if float(x) >= -PWR_EPS]
    if k == 0 or not atks:
        return 0.0
    full = (1 << k) - 1
    csum = [0.0] * (1 << k)
    vsum = [0.0] * (1 << k)
    for m in range(1, 1 << k):
        low = m & -m
        i = low.bit_length() - 1
        csum[m] = csum[m ^ low] + cards[i][0]
        vsum[m] = vsum[m ^ low] + cards[i][1]
    minimal_only = all(v >= 0.0 for _c, v in cards)
    sets_of = {}
    for _t, x in atks:
        if x in sets_of:
            continue
        need = x + 1000.0 - PWR_EPS
        ok = []
        for m in range(1, 1 << k):
            if csum[m] < need or take - vsum[m] <= 0.0:
                continue                                   # 足りない／受ける方が安い（割り当てない方が得）
            if minimal_only and any(csum[m ^ (1 << i)] >= need for i in range(k) if m >> i & 1):
                continue                                   # 余計な札を含む＝同じ攻撃を安く止める部分組が在る
            ok.append((m, take - vsum[m]))
        sets_of[x] = ok
    disc = [float(s) ** (int(start) + t) for t in range(int(turns))]
    memo = {}

    def best(j, avail):
        if j == len(atks) or avail == 0:
            return 0.0
        key = (j, avail)
        if key in memo:
            return memo[key]
        t, x = atks[j]
        out = best(j + 1, avail)                           # この攻撃は受ける
        w = disc[t]
        for m, sav in sets_of[x]:
            if m & avail == m:
                cand = w * sav + best(j + 1, avail & ~m)
                if cand > out:
                    out = cand
        memo[key] = out
        return out

    return float(best(0, full))


def delta_g(items, extra, xs, take_cost, s=1.0 - KO_P, turns=GUARD_TURNS):
    """`ΔG(札)` = 札を足した備え − 元の備え（≥ 0）。"""
    return guard_value(list(items) + [extra], xs, take_cost, s, turns) - guard_value(items, xs, take_cost, s, turns)


def incoming(tok_row):
    """来る攻撃の超過 `x` の並び（通らない攻撃は落とす）。"""
    return [x for x in incoming_x(tok_row) if x >= -PWR_EPS]


def take_cost_of(my_life, mu=MU):
    return float(TO.theta_take(my_life)) * float(mu)
