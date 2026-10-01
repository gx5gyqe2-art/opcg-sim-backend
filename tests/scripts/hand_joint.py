"""**手札の価値＝「1 枚 1 役」の最適な割り当ての値**（N-1・2026-09-26・ユーザ決定 判断6(a)・判断7(a)）。

それまで手札の価値は 3 通りに定義されていた:

1. 手札 1 枚の一律の値段 `μ`（平均）。
2. T67 の札ごとの価値 `max(ΔH_play, ΔG_guard)`（残りの手札に対して・`hand_plan.card_deltas`）。守る側の増分 `ΔG` は
   切る札の `v` を差し引いた**純額**。
3. G-2 の `V(手札) − V(手札 − S)`（`theory_bridge.hand_value_next`）。`V` は出す計画と守る備えを**別々に**全部の札で
   読んだ和なので、両方の役に立つ札は「出す＋守る」で数えられ、T67 の値より `min(ΔH, ΔG)` だけ大きい。

ユーザ決定: 手札の価値は**各札をちょうど 1 つの役に割り当てた最適値**とする:

```
V(手札) = max over 割り当て（札ごとに 出す／カウンター／持つ）
          出す計画(出す札)            … hand_plan.plan_value（ドンの枠・割引はそのまま）
        ＋ 守る備え(カウンターの札)    … 来る攻撃を、割り当てた札で止めた分 Σ s^(start+t)·受ける損
                                       （hand_guard.guard_value_exact と同じ攻撃・同じ割引・同じ地平・x + 1000 の規則）
札（組）の価値 = V(手札) − V(手札 − 札（組））
```

**守る側は「粗の」節約**（止めた攻撃ごとに受ける損そのもの）で数える——切った札の出す価値を差し引かない。
その札を出さなかった損は、割り当ての側（出す計画にその札が入らない）で既に数えているから（差し引くと二重に数える）。
`guard_value_exact` に `v = 0` の札を渡したのと同じ量。持つ（どちらにも使わない）札は 0。

**2 つの旧形との関係**（`tests/test_hand_joint.py` で値を固定）:

* 札が 1 枚だけの手札（2000 カウンター・`v = 0.03`・次の攻撃 1000・受ける損 0.0872・割引なし）:
  1 枚 1 役＝`max(v, 受ける損)`＝0.0872。T67 の `max(ΔH, ΔG)`＝`max(0.03, 0.0872 − 0.03)`＝0.0572（純額の `ΔG` は
  出す価値を 2 回引く）。G-2 の和＝`ΔH + ΔG`＝0.0872（割引なしでは一致）。1 ラウンド割り引くと
  1 枚 1 役＝`max(v, s·受ける損)`＝0.0620・G-2＝`v + s·(受ける損 − v)`＝0.0707（G-2 は割り引いた分 `(1 − s)v` だけ大きい）。
* 一般に `max(出す計画(全部), 守る備え(全部)) ≤ V ≤ 出す計画(全部) ＋ 守る備え(全部)`（証明は下）。

**単調性**（証明）: `V(手札) ≥ V(手札 − c)`。`手札 − c` の最適な割り当てに「c は持つ」を足せば `手札` の割り当てになり、
出す札・カウンターの札は同じなので値も同じ。`手札` の最大はそれ以上。よって札（組）の価値は 0 以上。
（相方待ち／条件の時計を**残りの手札で読み直す**とき〔`plan_items_of` を渡したとき〕は札の `v` が手札で変わるので
この証明は通らない＝その手札では調べる割り当てを絞らず全部調べ、差は 0 で床を打つ。）

**計算**（厳密）: カウンターの札の組 `G` ごとに `守る備え(G) + 出す計画(手札 − G)`。守る備えは全部の組について
1 回の再帰（攻撃の並び × 残りの札・メモ化）で出る。調べる `G` は、札が静的な手札（単調）では**無駄の無い組**
（どの 1 枚を抜いても守る備えが下がる組）だけで足りる（無駄な札を出す側へ戻しても出す計画は下がらない）。さらに
`守る備え(G) + 出す計画(手札)` が今の最良以下なら残りは調べない（出す計画は札が減って増えない）。
手札は高々 10 枚（規則）＝組は高々 1024。**新定数ゼロ**（`μ`・`KO_P`・`GUARD_TURNS`・既存の関数だけ）。
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from theory_order import KO_P, MU, PWR_EPS  # noqa: E402
from hand_guard import GUARD_TURNS  # noqa: E402

_EPS = 1e-12


def _plan_value(items, caps, s):
    import hand_plan as HP                                 # 遅延（`hand_plan` は橋を import する）
    return float(HP.plan_value(list(items), caps, s))


def guard_table(counters, xs, take_cost, s=1.0 - KO_P, turns=GUARD_TURNS, start=0):
    """**守る備え（粗）を札の組ごとに返す関数** `g(mask)`（bit i＝`counters[i]` の札をカウンターに割り当てる）。

    `guard_value_exact` と同じ攻撃（`xs` を毎ターン・大きい順）・同じ割引 `s^(start+t)`・同じ地平・同じ閾値（合計 ≥ x + 1000）。
    違いは札の `v` を 0 と置く（出す価値の損は割り当ての側で数える）ことだけ＝`guard_value_exact([(c, 0), …])` と一致。"""
    cs = [float(c) for c in counters]
    n = len(cs)
    take = float(take_cost)
    atks = [(t, float(x)) for t in range(int(turns)) for x in sorted(xs, reverse=True) if float(x) >= -PWR_EPS]
    csum = [0.0] * (1 << n)
    for m in range(1, 1 << n):
        low = m & -m
        csum[m] = csum[m ^ low] + cs[low.bit_length() - 1]
    pos = 0
    for i in range(n):
        if cs[i] > 0.0:
            pos |= 1 << i
    sets_of = {}
    for _t, x in atks:
        if x in sets_of:
            continue
        need = x + 1000.0 - PWR_EPS
        ok = []
        if take > 0.0:
            for m in range(1, 1 << n):
                if m & ~pos or csum[m] < need:
                    continue
                if any(csum[m ^ (1 << i)] >= need for i in range(n) if m >> i & 1):
                    continue                               # 過不足の無い組だけ（v = 0 なので余計な札は得をしない）
                ok.append(m)
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
        w = disc[t] * take
        for m in sets_of[x]:
            if m & avail == m:
                cand = w + best(j + 1, avail & ~m)
                if cand > out:
                    out = cand
        memo[key] = out
        return out

    return lambda mask: float(best(0, int(mask) & pos))


class JointValuer:
    """**1 枚 1 役の手札の価値 `V`** を、手札の部分集合ごとに（メモ化して）返す器。

    `n` 枚の札（添字 0..n−1）・`counters[i]`＝その札のカウンター値（印字＋イベントの上げ幅）。
    `plan_items_of(keep)`＝残った札（添字の frozenset）を出す計画に渡す `[(コスト, v), …]`。相方待ち／条件の時計を
    残った札で読み直すならここで読み直す。`monotone`＝札の `v` が手札で変わらない（読み直しが無い）ことを呼び手が保証する
    ——そのときだけ無駄の無い組・上界で絞る。"""

    def __init__(self, counters, plan_items_of, caps, xs, take_cost, s=1.0 - KO_P, turns=GUARD_TURNS, start=0,
                 monotone=True):
        self.n = len(counters)
        self.counters = [float(c) for c in counters]
        self.plan_items_of = plan_items_of
        self.caps = list(caps)
        self.s = float(s)
        self.monotone = bool(monotone)
        self.g = guard_table(self.counters, xs, take_cost, s, turns, start)
        self._plan = {}
        self._val = {}
        self._pos = [i for i in range(self.n) if self.counters[i] > 0.0]

    def plan_of(self, keep):
        key = frozenset(keep)
        if key not in self._plan:
            self._plan[key] = _plan_value(self.plan_items_of(key), self.caps, self.s) if key else 0.0
        return self._plan[key]

    def value(self, keep=None):
        """`V(keep)` と最良の割り当て（カウンターに回す札の添字の組）。`keep` 省略は手札全部。"""
        keep = frozenset(range(self.n)) if keep is None else frozenset(keep)
        if keep in self._val:
            return self._val[keep]
        pos = [i for i in self._pos if i in keep]
        masks = []
        for m in range(1 << len(pos)):
            mask = 0
            for b, i in enumerate(pos):
                if m >> b & 1:
                    mask |= 1 << i
            masks.append(mask)
        best = (self.plan_of(keep), ())                    # 誰もカウンターに回さない
        if self.monotone:
            gm = {m: self.g(m) for m in masks}
            tight = [m for m in masks if m and gm[m] > _EPS
                     and all(gm[m ^ (1 << i)] < gm[m] - _EPS for i in range(self.n) if m >> i & 1)]
            tight.sort(key=lambda m: -gm[m])
            top = best[0]                                  # 出す計画(keep) は札を減らして増えない＝上界
            for m in tight:
                if gm[m] + top <= best[0] + _EPS:
                    break
                cut = {i for i in range(self.n) if m >> i & 1}
                v = gm[m] + self.plan_of(keep - cut)
                if v > best[0] + _EPS:
                    best = (v, tuple(sorted(cut)))
        else:
            for m in masks:
                if not m:
                    continue
                cut = {i for i in range(self.n) if m >> i & 1}
                v = self.g(m) + self.plan_of(keep - cut)
                if v > best[0] + _EPS:
                    best = (v, tuple(sorted(cut)))
        out = (float(best[0]), best[1])
        self._val[keep] = out
        return out

    def loss(self, S):
        """組 `S` を手札から失ったときの価値の減り `max(0, V(手札) − V(手札 − S))`（単調な手札では床は効かない）。"""
        full = frozenset(range(self.n))
        return max(0.0, self.value(full)[0] - self.value(full - frozenset(S))[0])


def _norm(items, mu=MU):
    """`[(コスト, v, カウンター)]` または `{cost, v, counter}` の並びを揃える（`v` の `None` は `μ`＝G-2 と同じ規約）。"""
    out = []
    for it in items:
        if isinstance(it, dict):
            c, v, k = it["cost"], it["v"], it["counter"]
        else:
            c, v, k = it
        out.append((float(c), (float(mu) if v is None else v), float(k)))
    return out


def valuer_of(items, caps, xs, take_cost, s=1.0 - KO_P, turns=GUARD_TURNS, start=0, mu=MU):
    """静的な札の並び（`v` は数かターンごとの並び）から `JointValuer` を作る（読み直し無し＝単調）。"""
    its = _norm(items, mu)
    return JointValuer([k for _c, _v, k in its], lambda keep: [(its[i][0], its[i][1]) for i in sorted(keep)],
                       caps, xs, take_cost, s, turns, start, monotone=True)


def hand_value_joint(items, caps, xs, take_cost, s=1.0 - KO_P, turns=GUARD_TURNS, start=0, mu=MU, detail=False):
    """**`V`（1 枚 1 役）**。`detail=True` なら `(V, カウンターに回す札の添字)`。"""
    val, cut = valuer_of(items, caps, xs, take_cost, s, turns, start, mu).value()
    return (val, cut) if detail else val


def set_loss_joint(items, S, caps, xs, take_cost, s=1.0 - KO_P, turns=GUARD_TURNS, start=0, mu=MU):
    """組 `S`（添字）の価値＝`V(手札) − V(手札 − S)`（≥ 0）。"""
    return valuer_of(items, caps, xs, take_cost, s, turns, start, mu).loss(S)


def card_value_joint(items, k, caps, xs, take_cost, s=1.0 - KO_P, turns=GUARD_TURNS, start=0, mu=MU):
    """札 `k` 1 枚の価値＝`V(手札) − V(手札 − 札)`（≥ 0）。"""
    return set_loss_joint(items, (int(k),), caps, xs, take_cost, s, turns, start, mu)
