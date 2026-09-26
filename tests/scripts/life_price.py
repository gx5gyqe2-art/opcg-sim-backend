"""ライフ 1 枚の価格をライフ水準ごとに測り、**恒等式で締める**（記録だけ・ネット不要）。

**なぜ要るか**（2026-09-14・ユーザ指示「反事実を回す前に机上で確認できる理論値はないか」）:
`λ` は理論のほぼ全部に入っている（`Θ = λ/μ − 1 − τ_value`・`c̄ = λ/μ − 1 − τ_value`・
守りの無差別点）のに、**実測は逆因果に最も弱い量**である
（「ライフが低い＝既に負けている」・`2026-09-13_lambda_w32.md` §限界）。
そこで**ルールから出る恒等式**で上から押さえる:

> **互角（勝率 0.5）から相手のライフを全部削れば勝つ（1.0）**。したがって
> **ライフ 1 枚ずつの価値を経路に沿って足すと 0.5 になる**はずである。

```
Σ_{ℓ=1..L} λ(ℓ) = P(win | ライフ L) − P(win | ライフ 0) ≈ 0.5      L = リーダーのライフ
```

**これは推定ではなく制約**なので、実測の `λ` が満たさなければ**実測が誤っている**。
向きも決まる——**逆因果は `λ` を過大にする**ので、和が 0.5 を超えたらその超過が過大の量。

### 測るもの

| 量 | 定義 |
|---|---|
| `lam[ℓ]` | **隣り合う水準の対比**＝同じ帯の中で「ライフ ℓ の行」と「ライフ ℓ−1 の行」の勝率の差 |
| `lam_pooled` | 1 本の傾き（`hand_value_slope --axis life` と同じ量＝**器の突き合わせ用**） |
| `telescope[L]` | `Σ_{ℓ=1..L} lam[ℓ]`＝恒等式の左辺 |
| `excess[L]` | `telescope[L] − 0.5`＝**逆因果による過大の量**（正なら過大） |

**水準ごとに測る理由**は 2 つ。(1) 恒等式は**経路の平均**を要求するが、1 本の傾きは
**帯の人口で重み付けした平均**で別物（人口は低いライフ帯に偏らない）。
(2) 理論（§11）は「`λ` はリーサル圏で跳ねる」と言う＝**`Θ` は水準ごとに違う**はずで、
1 つの定数を固定しきい値に使うのは誤り。跳ね方が出れば `Θ(ℓ)` が書ける。

### 帯（交絡の除き方）

**説明変数（自ライフ）以外を固定する**: `(手札, 相手ライフ, ターン帯, 自場のキャラ数)`。
`hand_value_slope --axis life` と**同じ帯**にしてある（数字を直接比べるため）。

### 推定量

水準 ℓ の対比は**帯ごとの差の逆分散重み付き平均**:

```
lam[ℓ] = Σ_b w_b (ȳ_{b,ℓ} − ȳ_{b,ℓ−1}) / Σ_b w_b        w_b = n_{b,ℓ}·n_{b,ℓ−1} / (n_{b,ℓ}+n_{b,ℓ−1})
```

**CI は対局のクラスタブートストラップ**（局を復元抽出して再計算）。
1 局から複数の行を採るので、**行の数で割った SE は数倍過小に出る**
（`docs/measurement.md` §14-7・零和と option value の両方で踏んだ）。

### 限界（必ず添える）

- **恒等式は `P(win | ライフ 0) ≈ 0` を置いている**。実際はライフ 0 でも相手を先に倒せば勝つので
  0 ではない＝**`excess` は過大の量の上限ではなく目安**。両方出す（`p_win_by_life`）。
- **`λ` は自ライフに対する傾き**で、恒等式は相手のライフを削る価値を言う。
  零和の対称性で読み替えているが、**零和は実測で破れている**（−0.058〜−0.170）ので
  その分の滑りが在る。
- 逆因果は帯では消えない（本器はそれを**測る**のではなく**上から押さえる**）。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/life_price.py --in ~/tvs_rec --out ~/life_price.json
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

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len")
#: scalars の列（`rust/opcg_engine/src/encode/scalars.rs`・**生の枚数**）
SC_MY_LIFE, SC_OPP_LIFE = 0, 1
SC_MY_HAND = 6
SC_MY_FIELD = 8
SC_TURN = 10
#: 恒等式の右辺（互角から相手を倒し切るまでの勝率の伸び）
GAME_VALUE = 0.5
#: 見るライフ水準（リーダーのライフは 2〜6・平均 4.51）
LEVELS = (1, 2, 3, 4, 5, 6)
BOOT = 400


def turn_band(t):
    return "T<=4" if t <= 4 else ("T5-8" if t <= 8 else "T9+")


def band_key(sc_row, band="gross"):
    """**自ライフ以外**を固定した帯。`band` で**手札を固定するかを切り替える**。

    ここが本器のいちばん大事な設計である。**ダメージを受けるとライフはそのまま手札に入る**
    （ワンピースカードのルール）＝**ライフ → 手札の経路が在る**。したがって:

    | `band` | 帯に手札を入れるか | 測れるもの |
    |---|---|---|
    | `gross`（既定） | **入れる**（固定する） | **総額の `λ`**＝「ライフが 1 減る」だけの効果。手札で固定するので**もらえる 1 枚は入らない** |
    | `net` | **入れない**（自由にする） | **正味の `λ`**＝「ライフが 1 減り、代わりに 1 枚もらう」の合計 |

    `gross` は `hand_value_slope --axis life` と同じ帯（数字を直接比べられる）。
    **恒等式に載るのは `net` の方**——相手のライフを全部削って勝つ道中、相手は
    その枚数ぶん手札をもらうので、**正味で 0.5 にならなければならない**。
    差は理論の §9 の恒等式の `1 + τ_value` に当たるので、**2 つの差から `τ_value` が出る**:

    ```
    λ_gross − λ_net = μ·(1 + τ_value)        ⇒  τ_value = (λ_gross − λ_net)/μ − 1
    ```

    `τ_value`（トリガーの期待値）は**この一式でしか測っていない未検証項目**
    （`game_theory.md` 第V部 1 番・今まで恒等式からの逆算 0.37〜0.67 しか無かった）。
    """
    key = (int(round(float(sc_row[SC_OPP_LIFE]))),
           turn_band(int(round(float(sc_row[SC_TURN])))),
           int(round(float(sc_row[SC_MY_FIELD]))))
    if band == "gross":
        return (int(round(float(sc_row[SC_MY_HAND]))),) + key
    return key


def _extra(dd, n):
    return {"sc": np.asarray(dd["scalars"])[:n].astype(np.float32)}


def collect(dirs, holdout_mod=0, limit_games=0, band="gross"):
    """記録 → (自席ターンの行, 相手ターンの行, 局数)。**ネットは使わない**。"""
    own, opp = [], []
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=(),
                                                    extra_fn=_extra):
        seed = int(rows["seed"][idx[0]])
        if holdout_mod and seed % holdout_mod != 0:
            continue
        games += 1
        if limit_games and games > limit_games:
            break
        zs = {int(rows["who"][i]): float(rows["z"][i]) for i in idx}
        seen_own, seen_opp = set(), set()
        for i in idx:
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1:
                continue
            z = zs.get(w, 0.0)
            if z == 0.0:                       # 引き分け＝勝敗の差が定義できない
                continue
            is_own = PL.is_own_turn(w, t)
            if is_own:
                if int(rows["kind"][i]) != 0 or (w, t) in seen_own:
                    continue
                seen_own.add((w, t))
            else:
                if (w, t) in seen_opp:
                    continue
                seen_opp.add((w, t))
            sc = ex["sc"][i]
            rec = {"seed": seed, "z": 1.0 if z > 0 else 0.0,
                   "life": int(round(float(sc[SC_MY_LIFE]))),
                   "hand": float(sc[SC_MY_HAND]),
                   "band": band_key(sc, band)}
            (own if is_own else opp).append(rec)
    return own, opp, games


def _contrast(recs, level):
    """水準 `level` の対比（同じ帯の中の「ライフ level」と「ライフ level−1」の勝率の差）。

    重みは差の分散の逆数に比例する `n1·n0/(n1+n0)`（片方が 0 の帯は使えない）。
    """
    by = {}
    for r in recs:
        if r["life"] in (level, level - 1):
            by.setdefault(r["band"], ([], []))[0 if r["life"] == level else 1].append(r["z"])
    num = den = 0.0
    bands = pairs = 0
    for _b, (hi, lo) in by.items():
        if not hi or not lo:
            continue
        w = len(hi) * len(lo) / float(len(hi) + len(lo))
        num += w * (float(np.mean(hi)) - float(np.mean(lo)))
        den += w
        bands += 1
        pairs += len(hi) + len(lo)
    return (num / den if den > 0 else None), bands, pairs


def _pooled_slope(recs, y_key="z"):
    """帯の中だけの傾き（`hand_value_slope --axis life` の `within` と同じ量）。

    `y_key="hand"` にすると **`∂手札/∂ライフ`** が出る＝**ライフ → 手札の経路の太さ**。
    ダメージを受けると 1 枚もらうので −1 に近いはずだが、**その札はやがて使われる**ので
    観測時点で残っている量は 1 より小さい。`gross` と `net` の差を解釈するのに要る。
    """
    by = {}
    for r in recs:
        by.setdefault(r["band"], []).append(r)
    num = den = 0.0
    for _b, sub in by.items():
        if len(sub) < 2:
            continue
        h = np.array([r["life"] for r in sub], float)
        y = np.array([r[y_key] for r in sub], float)
        if float(h.var()) <= 0.0:
            continue
        hd = h - h.mean(); yd = y - y.mean()
        num += float((hd * yd).sum()); den += float((hd * hd).sum())
    return (num / den) if den > 0 else None


def _boot_ci(recs, fn, seeds, rng, reps=BOOT):
    """**対局のクラスタブートストラップ**（局を復元抽出）。行で割った SE は過小になる。"""
    if reps <= 0:                      # `--boot 0`＝CI を出さない（点推定だけ見たいとき）
        return None, None
    by_game = {}
    for r in recs:
        by_game.setdefault(r["seed"], []).append(r)
    keys = list(by_game)
    if len(keys) < 3:                  # 局が 2 つ以下＝クラスタが足りず CI は出せない
        return None, None
    out = []
    for _ in range(reps):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        sub = [r for k in pick for r in by_game[keys[k]]]
        v = fn(sub)
        if v is not None:
            out.append(v)
    if len(out) < reps // 4:
        return None, None
    return (round(float(np.percentile(out, 2.5)), 5),
            round(float(np.percentile(out, 97.5)), 5))


def summarise(recs, seeds=None, reps=BOOT, seed=0):
    """水準ごとの `λ`・テレスコープ和・恒等式との差。"""
    if not recs:
        return None
    rng = np.random.default_rng(seed)
    out = {"n": len(recs), "games": len({r["seed"] for r in recs}),
           "winrate": round(float(np.mean([r["z"] for r in recs])), 4)}
    # 器の突き合わせ: 1 本の傾き（既存の計器の値と比べる）
    pooled = _pooled_slope(recs)
    lo, hi = _boot_ci(recs, _pooled_slope, seeds, rng, reps)
    out["lam_pooled"] = {"value": None if pooled is None else round(pooled, 5),
                         "ci95": [lo, hi]}
    # **ライフ → 手札の経路の太さ**（ダメージで 1 枚もらうが、観測時点で残っている量）
    dh = _pooled_slope(recs, "hand")
    out["d_hand_d_life"] = None if dh is None else round(dh, 5)
    # ライフ水準ごとの素の勝率（恒等式の両端を見るため）
    out["p_win_by_life"] = {}
    for lv in range(0, 7):
        sub = [r["z"] for r in recs if r["life"] == lv]
        if sub:
            out["p_win_by_life"][str(lv)] = {
                "n": len(sub), "p_win": round(float(np.mean(sub)), 4)}
    # 水準ごとの対比
    out["lam"] = {}
    for lv in LEVELS:
        val, bands, pairs = _contrast(recs, lv)
        if val is None:
            continue
        blo, bhi = _boot_ci(recs, lambda rs, _l=lv: _contrast(rs, _l)[0], seeds, rng, reps)
        out["lam"][str(lv)] = {"value": round(val, 5), "ci95": [blo, bhi],
                               "bands": bands, "n": pairs}
    # テレスコープ和と恒等式
    out["telescope"] = {}
    acc = 0.0
    for lv in LEVELS:
        k = str(lv)
        if k not in out["lam"]:
            break
        acc += out["lam"][k]["value"]
        out["telescope"][k] = {
            "sum": round(acc, 5),
            # **恒等式**: 経路で足すと 0.5 になるはず（超過＝逆因果による過大の目安）
            "excess": round(acc - GAME_VALUE, 5),
            "ratio": round(acc / GAME_VALUE, 4),
            # 同じ水準数を 1 本の傾きで代用したときの値（比較用）
            "pooled_x_levels": (None if pooled is None else round(pooled * lv, 5)),
        }
    # 恒等式が許す `λ` の経路平均（リーダーのライフ 4／5／平均 4.51 で）
    out["lam_allowed"] = {str(L): round(GAME_VALUE / L, 5) for L in (4, 5)}
    out["lam_allowed"]["4.51"] = round(GAME_VALUE / 4.51, 5)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--holdout-mod", type=int, default=0, help="seed %% mod == 0 の局だけ（0=全部）")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--mu", type=float, default=0.0433,
                    help="手札 1 枚の価格（`τ_value` の逆算に使う・既定は実測値）")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    res = {"game_value": GAME_VALUE, "mu": a.mu}
    for band in ("gross", "net"):
        own, opp, games = collect(a.src, a.holdout_mod, a.limit_games, band)
        res["games"] = games
        res[band] = {"own_turn": summarise(own, reps=a.boot, seed=1),
                     "opp_turn": summarise(opp, reps=a.boot, seed=2)}
    # **2 つの帯の差は「ライフ → 手札」の経路ぶん**である。
    #
    # **`τ_value` はこの対比では同定できない**（2026-09-14 に誤った式を書いて気付いた）。
    # トリガーは**ダメージを受けた瞬間に 1 回**起きる効果で、**手札の枚数を経由しない**＝
    # 帯で手札を固定しても遮断されない。したがって `gross` と `net` の差に入るのは
    # **もらった札が観測時点まで残っている分だけ**:
    #
    #     λ_gross − λ_net = |∂手札/∂ライフ| · μ
    #
    # ＝**`μ`・`λ`・経路の太さが互いに整合するかの検算**になる（比が 1 なら整合）。
    res["card_channel"] = {}
    for side in ("own_turn", "opp_turn"):
        g = (res["gross"][side] or {}).get("lam_pooled", {}).get("value")
        n = (res["net"][side] or {}).get("lam_pooled", {}).get("value")
        dh = (res["net"][side] or {}).get("d_hand_d_life")
        if g is None or n is None or dh is None:
            continue
        expect = abs(dh) * a.mu
        res["card_channel"][side] = {
            "lam_gross": g, "lam_net": n, "diff": round(g - n, 5),
            "d_hand_d_life": dh, "expected_diff": round(expect, 5),
            # 1 に近ければ `μ`・`λ`・経路の太さが整合する（自由パラメータ無しの検算）
            "ratio": (round((g - n) / expect, 4) if expect > 0 else None),
            "note": "λ_gross − λ_net = |∂hand/∂life|·μ （τ_value はこの対比では同定できない）",
        }
    res["seconds"] = round(time.time() - t0, 1)
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
