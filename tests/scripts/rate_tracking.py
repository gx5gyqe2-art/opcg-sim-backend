#!/usr/bin/env python3
"""**`A` は「このターンの損害」を追えているか**（T128・2026-09-20・ユーザ指示「進めてください」）。

## なぜ要るか

T127 で **`A` を時計に入れる 2 本の独立な道（`curve_scaled`／`theory`）が同じ約 0.2 の AUC を失った**。
結論は「**患部は `A` の推定そのもの**」だったが、**`A` の何が悪いのかは分かっていなかった**。

**水準と形はもう合っている**——T107 が**とどめのターンを外すと過大が消える**ことを示し
（実 1.21／1.48 → 0.90）、T108 が残り 15% を埋めた（0.97〜1.12／0.87〜1.20）。
**にもかかわらず順序づけでは平均の輪郭（`curve`）に大差で負ける。**

**残る可能性は 1 つ**——**`A` は `j` の平均としては合っているが、局ごと・ターンごとには追えていない**。
本器はそれを**直接**測る。

## 測る 3 つ（当てはめゼロ・回帰しない）

行は**席 × 自席ターン**（`crossing_bridge.collect` の `turn_harm`）。既定で
**とどめのターンを外す**（`t_left >= 2`・T107 の規則——勝った席の最後の自席ターンは
相手のライフが尽きた時点で終わるので `A` と実際の損害が構造的にずれる）。

```
harm   … そのターンに実際に与えた損害          （実測）
A      … そのターンの理論の速さ `slope_theory`  （理論・状態から）
prof[j]… 自席ターン番号 j の損害の平均の輪郭    （`curve` が使っている物・**別のセットの表**）
```

1. **水準**（検算）: `mean(A) / mean(harm)`——T107／T108 の後なので 1 の近くに出るはず。
2. **追随**: `corr(A, harm)` と `corr(prof[j], harm)`——**`A` は輪郭より良く追えているか**。
3. **輪郭を超える情報**（本命）: `corr(A − prof_th[j], harm − prof[j])`
   ——**ターン番号で説明できる分を両側から引いた残り**が相関するか。
   **ここが 0 なら、`A` は「今の盤面」を見ているのに「今のターンの損害」を 1 円も
   余分に説明していない**＝`curve` に負けるのは当然で、直すべきは形ではなく**読む対象**。

**輪郭は測る記録と別のセットから引く**（`profile_for`／`profile_th_for` の `cross`・§0.1 条件 1）
——同じ記録の平均を使うと 3 番目が自分自身との相関になる。

## 予告（測る前に書く）

**2 が拮抗し 3 がほぼ 0 なら**「`A` は平均としてしか効いていない」。
**3 が有意に正なら**「`A` は情報を持っているのに、時計の式がそれを潰している」＝患部は時計の側に戻る。

使い方:

    python tests/scripts/rate_tracking.py --in <records_dir> [--games N] [--json out.json]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import crossing_bridge as CB  # noqa: E402
from theory_order import MU, THETA  # noqa: E402


def corr(a, b):
    """相関（どちらかが定数なら `None`）。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def at(prof, j):
    """輪郭の `j` 番目（末尾を超えたら末尾＝`tau_from_profile` と同じ規約）。"""
    if not prof:
        return 0.0
    return float(prof[min(int(j), len(prof) - 1)])


def measure(turn_harm, prof, prof_th, include_last=False):
    """`turn_harm` の行から 3 つの量を出す（当てはめゼロ）。"""
    rows = [r for r in turn_harm
            if include_last or int(r.get("t_left") or 0) >= 2]      # T107 の規則
    if not rows:
        return {"n": 0}
    harm = np.array([float(r["harm"]) for r in rows])
    a = np.array([float(r["slope_theory"]) for r in rows])
    js = np.array([int(r["j"]) for r in rows])
    pj = np.array([at(prof, j) for j in js])
    ptj = np.array([at(prof_th, j) for j in js])
    mh = float(harm.mean())
    ss_tot = float(((harm - mh) ** 2).sum())

    def _r2(pred):
        """**当てはめない**予測としての説明力（そのまま引く・係数を合わせない）。"""
        return round(1.0 - float(((harm - pred) ** 2).sum()) / ss_tot, 4) if ss_tot > 0 else None

    out = {
        "n": len(rows), "games": len({(r["g"]) for r in rows}),
        "include_last": bool(include_last),
        # 1. 水準（検算）
        "harm_mean": round(mh, 4), "a_mean": round(float(a.mean()), 4),
        "level_ratio": round(float(a.mean()) / mh, 3) if mh > 1e-9 else None,
        "harm_sd": round(float(harm.std()), 4), "a_sd": round(float(a.std()), 4),
        # 2. 追随
        "corr_a": round(corr(a, harm), 4) if corr(a, harm) is not None else None,
        "corr_prof": round(corr(pj, harm), 4) if corr(pj, harm) is not None else None,
        "corr_j": round(corr(js, harm), 4) if corr(js, harm) is not None else None,
        # 3. 輪郭を超える情報（本命）
        "corr_resid": (round(corr(a - ptj, harm - pj), 4)
                       if corr(a - ptj, harm - pj) is not None else None),
        "resid_n": int(len(rows)),
        # 予測としての説明力（当てはめゼロ・そのまま引く）
        "r2_a": _r2(a), "r2_prof": _r2(pj),
    }
    # `j` ごとの内訳（どのターンで追随が落ちるか）
    by_j = {}
    for j in sorted(set(int(x) for x in js)):
        m = js == j
        if int(m.sum()) < 20:
            continue
        by_j[str(j)] = {"n": int(m.sum()),
                        "harm": round(float(harm[m].mean()), 4),
                        "a": round(float(a[m].mean()), 4),
                        "ratio": (round(float(a[m].mean()) / float(harm[m].mean()), 3)
                                  if harm[m].mean() > 1e-9 else None),
                        "corr": (round(corr(a[m], harm[m]), 4)
                                 if corr(a[m], harm[m]) is not None else None)}
    out["by_j"] = by_j
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", nargs="+", required=True, help="記録のディレクトリ（複数可）")
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--include-last", action="store_true",
                    help="とどめのターンも入れる（既定は T107 の規則で外す）")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    t0 = time.time()
    prof = CB.profile_for(a.inp)
    prof_th = CB.profile_th_for(a.inp)
    if not prof or not prof_th:
        raise SystemExit("輪郭が引けない（別のセットの表が要る・§0.1 条件 1）")
    _rows, _ledger, _stats, turn_harm, _tc = CB.collect(a.inp, a.games, THETA, MU)
    out = {"dirs": list(a.inp),
           "kind": CB.record_kind(a.inp),
           "excl": measure(turn_harm, prof, prof_th, include_last=False),
           "incl": measure(turn_harm, prof, prof_th, include_last=True),
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
