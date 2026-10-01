"""**必要な `κ` を測る**——「1 手の価値＝その手が動かした勝率」を**局所で**突き合わせる（T80・2026-09-17・読み取り専用）。

ユーザの問い 2 つに答えるための器:

1. **恒等式を区間で測る**（`2026-09-17_complete_information.md` §4 の案 1）。今までは 1 局ぶんの `ΔG` を最終結果
   （`z − 0.5`）と比べていた。理論が主張しているのは **1 手の価値＝その手が動かした勝率** なので、
   **ターンの前後で勝率がどれだけ動いたか**と**そのターンの価格の和**を並べるのが本来の形:

   ```
   W(t)      = Φ(D(t)/σ_D)                     その時点の勝率（席 0 から見る・`theory_order.prob_of_d`）
   ΔW(t)     = W(t+1) − W(t)                   そのターンで動いた勝率（実測の側）
   g(t)      = Σ(席 0 の手の価格) − Σ(席 1 の手の価格)   そのターンの帳簿（**κ を掛けない生の価格**）
   κ_必要(t) = ΔW(t) / g(t)                     この局面で価格 1 単位が勝率いくらに当たるべきか
   ```

2. **`κ` を掛ける量と、価格の中の残りターン数（`r_turns`）が二重に縮んでいないか**（ユーザ指摘）。
   `κ_必要` を **`r_turns` 別**と **`|D|` 別**に層別すれば、縮ませすぎているのが**量の側**（`r_turns`）か
   **単価の側**（`κ`）かが分かれる。**両方で不足していれば両方が過剰に縮んでいる**証拠。

**当てはめない**——`κ_必要` は測るだけで、式に戻すかはユーザ判定。`W` は既存の `σ_D` の積分で**新定数ゼロ**。

**限界**: 区間は 1 ターン（`D` はそのターン最初の行で読む）。`g` の小さいターンは `κ_必要` が発散するので
`|g| ≥ G_FLOOR` の行だけを使い、**中央値**で読む（平均は外れ値に弱い）。零和なので席 0 の視点に揃える
（席 1 の手は符号を返す・`D` も符号を返す）。

実行例（`theory_bridge` が集めた `kappa_needed` を読む）:
  OPCG_LOG_SILENT=1 python tests/scripts/theory_bridge.py --in ~/w41 --out ~/bridge.json
"""
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

def _PROB(d):
    """`W(D) = Φ(D/σ_D)`（`theory_order.prob_of_d`）。"""
    import theory_order as TO
    return TO.prob_of_d(d)


#: `κ_必要 = ΔW/g` が発散しないための下限（価格の単位・`μ` の 1/10 程度）
G_FLOOR = 0.005
#: `|D|` の層（接戦 → 決着）
D_BANDS = ((0.0, 1.0, "|D|<=1"), (1.0, 3.0, "1<|D|<=3"), (3.0, float("inf"), "|D|>3"))


def d_band(d):
    """`|D|` の層の名前（接戦 → 決着）。"""
    a = abs(float(d))
    for lo, hi, name in D_BANDS:
        if lo < a <= hi or (lo == 0.0 and a <= hi):
            return name
    return D_BANDS[-1][2]


def _fit(pairs):
    """**必要な `κ`＝`ΔW` を `g` に回した最小二乗の傾き**（比の中央値より頑健——`g` の符号が振れるため）。
    `corr` は相関で、**0 に近ければどんな `κ` でも恒等式は立たない**（＝価格の順序自体が勝率と結びついていない）。"""
    if len(pairs) < 10:
        return None
    g = np.asarray([p[0] for p in pairs], float)
    w = np.asarray([p[1] for p in pairs], float)
    gg = float((g * g).sum())
    if gg <= 0.0:
        return None
    sd = float(g.std()) * float(w.std())
    return {"n": len(pairs), "kappa_fit": round(float((g * w).sum() / gg), 3),
            "corr": (round(float(((g - g.mean()) * (w - w.mean())).mean() / sd), 3) if sd > 0 else None),
            "g_rms": round(float(np.sqrt(gg / len(g))), 4),
            "dW_rms": round(float(np.sqrt((w * w).mean())), 4)}


def windows(series, rounds=1):
    """**ラウンドの窓**（`rounds` ラウンド＝`2 × rounds` ターン）を並べる（T81）。

    `series` は 1 局のターンの並び（`{"d0","g0","g_fam","r_turns","chars"}`・`d0` は席 0 視点）。
    窓の `g` はその間の**生の価格の和**・`ΔW` は窓の前後の勝率の差・`dchars` は場のキャラ数の変化。
    """
    n = 2 * int(rounds)
    out = []
    for i in range(len(series) - n):
        a, b = series[i], series[i + n]
        fam = {}
        for k in range(i, i + n):
            for f, v in (series[k].get("g_fam") or {}).items():
                fam[f] = fam.get(f, 0.0) + float(v)
        out.append({"d0": a["d0"], "r_turns": a.get("r_turns"),
                    "g0": float(sum(series[k]["g0"] for k in range(i, i + n))),
                    "dW": _PROB(b["d0"]) - _PROB(a["d0"]),
                    "g_fam": fam,
                    "dchars": (None if a.get("chars") is None or b.get("chars") is None
                               else int(b["chars"]) - int(a["chars"]))})
    return out


def summarise_games(games, g_floor=G_FLOOR, max_rounds=3):
    """**相関 0.36 の中身を割る**（T81）——(a) 窓の広さ（時差か）・(b) 手の型（時計に見えない手か）・
    (c) 場が動いたか（時計が粗いか）。`games` は局ごとのターンの並び。"""
    out = {"by_window": {}, "by_family": {}, "by_combo": {}, "by_board_change": {}}
    for r in range(1, int(max_rounds) + 1):
        ws = [w for gsr in games for w in windows(gsr, r)]
        use = [w for w in ws if abs(w["g0"]) >= g_floor]
        out["by_window"][str(r)] = _fit([(w["g0"], w["dW"]) for w in use])
    ws = [w for gsr in games for w in windows(gsr, 1)]
    fams = sorted({f for w in ws for f in (w.get("g_fam") or {})})
    for f in fams:
        pairs = [(w["g_fam"].get(f, 0.0), w["dW"]) for w in ws if abs(w["g_fam"].get(f, 0.0)) >= g_floor]
        out["by_family"][f] = _fit(pairs)
        # **その型を抜いた残り**との相関（抜くと当てはまりが上がるか）
        rest = [(w["g0"] - w["g_fam"].get(f, 0.0), w["dW"]) for w in ws
                if abs(w["g0"] - w["g_fam"].get(f, 0.0)) >= g_floor]
        out["by_family"][f + "_除いた残り"] = _fit(rest)
    # **T81**: 時計が見る手（攻め・守り）だけの帳簿＝相関がどこまで上がるか
    def _sum(w, keys):
        return float(sum((w.get("g_fam") or {}).get(k, 0.0) for k in keys))
    out["by_combo"] = {}
    for name, keys in (("攻め＋守り", ("attack", "guard")),
                       ("攻め＋守り＋付与", ("attack", "guard", "attach")),
                       ("攻めだけ", ("attack",))):
        pairs = [(_sum(w, keys), w["dW"]) for w in ws if abs(_sum(w, keys)) >= g_floor]
        out["by_combo"][name] = _fit(pairs)
    drop = [(w["g0"] - _sum(w, ("play", "effect")), w["dW"]) for w in ws
            if abs(w["g0"] - _sum(w, ("play", "effect"))) >= g_floor]
    out["by_combo"]["出す＋効果を除いた残り"] = _fit(drop)
    for name, sel in (("場が動いた", lambda w: w.get("dchars") not in (None, 0)),
                      ("場は同じ", lambda w: w.get("dchars") == 0)):
        pairs = [(w["g0"], w["dW"]) for w in ws if sel(w) and abs(w["g0"]) >= g_floor]
        out["by_board_change"][name] = _fit(pairs)
    return out


def _stat(vals):
    if not vals:
        return None
    a = np.asarray(vals, float)
    return {"n": int(a.size), "median": round(float(np.median(a)), 3),
            "mean": round(float(a.mean()), 3),
            "p25": round(float(np.percentile(a, 25)), 3),
            "p75": round(float(np.percentile(a, 75)), 3)}


def summarise(turns, g_floor=G_FLOOR):
    """ターンの並び（`{"d0","g0","r_turns","dW"}`）→ `κ_必要` の全体と層別。

    `dW` は次のターンとの差（呼び側が入れる）。`g0` は**生の価格**（`κ` を掛けない）。
    """
    use = [t for t in turns if t.get("dW") is not None and abs(float(t["g0"])) >= g_floor]
    if not use:
        return {"n_turns": len(turns), "n_used": 0}
    out = {"n_turns": len(turns), "n_used": len(use), "g_floor": g_floor,
           "dW_mean": round(float(np.mean([t["dW"] for t in use])), 4),
           "g_abs_mean": round(float(np.mean([abs(t["g0"]) for t in use])), 4),
           # **正本は回帰の傾き**（比の中央値は `g` の符号が振れるので参考）
           "fit": _fit([(t["g0"], t["dW"]) for t in use]),
           "ratio": _stat([t["dW"] / t["g0"] for t in use])}
    by_d, by_r = {}, {}
    for t in use:
        by_d.setdefault(d_band(t["d0"]), []).append((t["g0"], t["dW"]))
        r = t.get("r_turns")
        if r is not None:
            by_r.setdefault(str(int(round(float(r)))), []).append((t["g0"], t["dW"]))
    out["by_d"] = {k: _fit(v) for k, v in sorted(by_d.items())}
    out["by_r_turns"] = {k: _fit(v) for k, v in sorted(by_r.items())}
    return out
