"""**`μ` は `max(λ/c̄, μ_develop)` のどちらの枝にいるか**（T26・読み取り専用）。

`docs/cpu_theory_gap.md` の **T26**・`game_theory.md` **§17.1.2/§17.1.3**。
T24（`2026-09-14_price_scaling.md` §4）が出した問い。

## なぜ要るか——**`Θ` の導出が片方の枝を前提にしている**

§17.1.3 は `Θ = λ/μ − 1 − τ` を**守り用途が縛っているとき**（`μ = λ/c̄`）に導いた。
**攻め用途が縛る帯（`μ_develop > λ/c̄`）では、この導出は成り立たない。**

```
μ_guard   = λ / c̄            守る枝: 手札 c̄ 枚で 1 撃止める＝ライフ 1 枚ぶん
μ_develop = 体になって自分の時計を縮める価値   攻め枝
μ         = max(μ_guard, μ_develop)
```

T24 は `λ/μ` の点推定が `A_opp` とともに**下がる**のを見た（4.36→2.90／3.35→2.75）。
**守る枝なら完全に平ら**（`w` も `1/A_opp` も約分される）・**作る枝なら下がる**。
CI は重なるので証拠にならないが、**枝そのものは行ごとに直接数えられる**——それが本器。

## 測り方——**どちらも既存の式から出す。新しい定数を入れない**

| 量 | 出し方 |
|---|---|
| `μ_guard` | `LAM / c̄`。**`c̄` は 2 通りで出して両方の判定を並べる**（下記） |
| `μ_develop` | **その行の PLAY 候補**の `play_value` の最大 **＋ `μ`** |

> **`play_value` は `ν − μ − 費用·δ`＝「登場の純益」**であって `μ_develop` ではない
> （2026-09-14 に踏んだ）。**`μ` を足し戻して `ν(体) − 費用·δ`** にしたものが
> 「カード 1 枚を作る用途に使ったときの価値」＝枝の比較に要る量である。
> 足し戻さないと**`μ` 1 個ぶん守る枝に寄る**。

**`c̄` を 2 通りで出す理由**——どちらにも既知の偏りがあるため、**片方だけでは判定できない**:

| | 出し方 | 既知の偏り |
|---|---|---|
| `row` | その行に来ている攻撃の `c_of(x)` の平均 | **低く出る**——自席ターンの入口では**相手のドンが外れている**ので、
実際に殴られるときのパワーより小さい値を読む（＝`μ_guard` が高く出る＝守る枝に寄る） |
| `const` | `λ/μ − 1 − τ` から出る **2.298**（§17.1.3 の恒等式） | **行ごとに動かない**＝状態依存が見えない |

**2 つの判定が一致しなければ、その一致しないこと自体を報告する**（片方を選ばない）。

## 読み方（事前登録）

- **`develop_share` が過半なら「作る枝」が既定**＝**`Θ = λ/μ − 1 − τ` の導出はその帯で効かない**。
- **帯で share が動くなら、`Θ` が状態の関数であるべき根拠が 1 本増える**（T23 とは別経路）。
- **どちらかに 9 割寄っていれば `max` は実質不要**＝式を単純化できる。

**限界**: `μ_develop` は**その行に実際に在った候補**の最大なので、
**手札が悪い行では低く出る**（枝の判定ではなく手札の運を測ってしまう）。
帯ごとに見て、手札枚数で割った内訳も出す。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/mu_branch.py --in ~/w39 ~/w42 --out ~/mu_branch.json
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
from theory_order import (own_attackers_of, LAM, MU, POL_COLS, PWR_EPS, ROW_COLS, SC_MY_DON,  # noqa: E402
                          SC_MY_HAND, SC_MY_LIFE, SC_MY_LEADER_POWER,
                          SC_OPP_LEADER_POWER, SC_OPP_LIFE, THETA, c_of, incoming_x,
                          score_candidate, slot_power, theta_of)

#: 恒等式から出る `c̄`（`Θ = c̄ − 1 − τ` を既定の `Θ=1.15`・`τ=0.148` で解いたもの）。
#: **行から出す `c̄` は低く出る**（相手のドンが外れた入口を読むため）ので**両方で判定する**。
CBAR_CONST = 2.298
#: `c̄` の出し方 3 通り。**`row_don` が筋の通った版**——`incoming_x(don=)` に
#: **相手が次のターンに付与するドン**を渡して、殴られるときのパワーで費用を測る。
CBAR_MODES = ("row", "row_don", "const")
#: 相手のドン（`scalars.rs`）。自席ターンの入口では全部戻っている。
SC_OPP_DON_ACTIVE, SC_OPP_DON_RESTED = 4, 5
#: ドンの上限（次のターンに 1 個増える）
DON_MAX = 10
#: 「作る枝」に数える `PLAY` 候補（`--play-kind`）。**`all`＝2026-09-14 以降の既定**（P2-1 で
#: イベント・ステージにも値が付くようになったのでそちらも混ざる）／**`char`＝体だけ**。
#: T26 の問い「理論は体を出すなと言っているか」に答えるのは **`char`** の方。
PLAY_KINDS = ("all", "char")
PLAY_KIND = "all"


def _counts_as_develop(cards, cid, kind=None):
    """この `PLAY` 候補を「作る枝」に数えるか。

    `char` は**体だけ**——イベント・ステージ（効果だけを買う札）は体にならないので外す。
    **カードが引けない候補も外す**（`all` では `score_candidate` が `None` を返して同じ結果）。
    """
    kind = PLAY_KIND if kind is None else kind
    if kind != "char":
        return True
    c = cards.info(cid) if cid else None
    return bool(c) and not (c.get("event") or c.get("stage"))
#: 手札の枝の内訳を見る帯（`μ_develop` は候補の質に依るので手札の枚数で割って読む）
HAND_BANDS = ((0, 2, "hand0_2"), (3, 5, "hand3_5"), (6, 99, "hand6+"))
TURN_BANDS = ((1, 4, "early"), (5, 8, "mid"), (9, 99, "late"))


def _extra(dd, n):
    return {"sc": np.asarray(dd["scalars"])[:n].astype(np.float32),
            "tok": np.asarray(dd["tokens"])[:n].astype(np.float32)}


def opp_don_next(sc_row):
    """**相手が次のターンに付与できるドンの数**（いま持っている数 +1・上限 10）。

    自席ターンの入口では相手のドンは**全部外れて戻っている**ので、枠のパワーは素の値。
    そのまま `c_of` に入れると**殴られるときより弱い攻撃**を測ることになる。
    """
    tot = float(sc_row[SC_OPP_DON_ACTIVE]) + float(sc_row[SC_OPP_DON_RESTED])
    return int(min(DON_MAX, tot + 1.0))


def cbar_of(tok_row, don=0):
    """**その行に来ている攻撃**の守備費用の平均（来ていなければ `None`）。

    `don` を渡すと**高い攻撃から順に +1000 ずつ**乗せる（`incoming_x` の機能）
    ＝**実際に殴られるときのパワー**で費用を測る。
    """
    xs = [x for x in incoming_x(tok_row, don=don) if x >= -PWR_EPS]
    if not xs:
        return None
    return float(np.mean([c_of(x) for x in xs]))


def band_of(v, bands):
    for lo, hi, name in bands:
        if lo <= v <= hi:
            return name
    return bands[-1][2]


def collect(dirs, limit_games=0, theta=THETA, mu=MU, lam=LAM, theta_mode="const"):
    """自席の main 行 → 2 つの枝の値と、どちらが縛っているか。"""
    cards = PL.Cards()
    recs = []
    stats = {"games": 0, "rows": 0, "no_incoming": 0, "no_play_cand": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        seed = int(rows["seed"][idx[0]])
        for i in idx:
            if int(rows["kind"][i]) != 0:
                continue
            k = int(L[i])
            if k < 1:
                continue
            sc, tok = ex["sc"][i], ex["tok"][i]
            stats["rows"] += 1
            cbar = cbar_of(tok)
            cbar_don = cbar_of(tok, don=opp_don_next(sc))
            if cbar is None or cbar <= 0.0 or cbar_don is None or cbar_don <= 0.0:
                stats["no_incoming"] += 1
                continue                      # **定数で埋めない**（守る枝に寄る偏りが入る）
            th = theta_of(tok, float(sc[SC_MY_LIFE]), float(sc[SC_MY_DON]),
                          mode=theta_mode, theta=theta)
            ctx = {"theta": th, "mu": mu,
                   "opp_leader_power": float(sc[SC_OPP_LEADER_POWER]) * 1e4,
                   "my_leader_power": float(sc[SC_MY_LEADER_POWER]) * 1e4,
                   "r_turns": max(1.0, min(5.0, float(sc[SC_OPP_LIFE]))),
                   "don_k": 1,
                   "attackers": own_attackers_of(tok, float(sc[SC_OPP_LEADER_POWER]) * 1e4),
                   "don_active": float(sc[SC_MY_DON])}   # 登場の機会費用（T43）
            b = int(ptr[i])
            best = None
            for j in range(b, b + k):
                sig = json.loads(pol["pol_sig"][j])
                if not sig or sig[0] != "PLAY":
                    continue
                cid = str(pol["pol_cid"][j]) or None
                if not _counts_as_develop(cards, cid, PLAY_KIND):
                    continue
                v = score_candidate(sig, cid, None, ctx, cards,
                                    src_power=slot_power(tok, pol["pol_si"][j]),
                                    don_k=pol["pol_k"][j])
                if v is not None and (best is None or v > best):
                    best = float(v)
            if best is None:
                stats["no_play_cand"] += 1
                continue                      # 出せる体が無い行では枝を比べられない
            recs.append({"seed": seed, "turn": int(rows["turn"][i]),
                         "cbar": cbar, "cbar_don": cbar_don,
                         "mu_guard_row": float(lam / cbar),
                         "mu_guard_row_don": float(lam / cbar_don),
                         "mu_guard_const": float(lam / CBAR_CONST),
                         # **`play_value` は純益（`ν − μ − 費用δ`）なので `μ` を足し戻す**
                         "mu_develop": best + mu,
                         "hand": int(float(sc[SC_MY_HAND]))})
    return recs, stats


def _share(sub, cbar_mode="row"):
    """**`c̄` の出し方を替えて両方の判定を出す**（片方を選ばない）。"""
    if not sub:
        return None
    gk = "mu_guard_%s" % cbar_mode
    dev = sum(1 for r in sub if r["mu_develop"] > r[gk])
    return {"n": len(sub), "develop_share": round(dev / len(sub), 4),
            "mu_guard_mean": round(float(np.mean([r[gk] for r in sub])), 4),
            "mu_develop_mean": round(float(np.mean([r["mu_develop"] for r in sub])), 4),
            "mu_max_mean": round(float(np.mean([max(r[gk], r["mu_develop"])
                                                for r in sub])), 4),
            # **登場の純益が正の行の割合**——理論が「出すべき」と言う行がどれだけ在るか。
            # これが 0 に近ければ、枝の判定より先に **`ν − 費用δ` の水準**を疑う。
            "play_positive_share": round(
                sum(1 for r in sub if r["mu_develop"] > MU) / len(sub), 4),
            "cbar_mean": round(float(np.mean(
                [r["cbar_don" if cbar_mode == "row_don" else "cbar"] for r in sub])), 3)}


def _boot_share(recs, reps=200, seed=0, cbar_mode="row"):
    """**対局を復元抽出**して share の CI（`measurement.md` §14-15）。"""
    if reps <= 0 or not recs:
        return [None, None]
    by = {}
    for r in recs:
        by.setdefault(r["seed"], []).append(r)
    gids = list(by)
    if len(gids) < 3:
        return [None, None]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(reps)):
        pick = rng.integers(0, len(gids), len(gids))
        sub = [r for g in pick for r in by[gids[g]]]
        s = _share(sub, cbar_mode)
        if s:
            vals.append(s["develop_share"])
    return ([round(float(np.percentile(vals, 2.5)), 4),
             round(float(np.percentile(vals, 97.5)), 4)] if len(vals) >= 10 else [None, None])


def _verdict(share):
    """**事前登録した読み方**——9 割寄れば `max` は実質不要、割れていれば状態の関数。"""
    return ("develop_binds" if share >= 0.9 else
            "guard_binds" if share <= 0.1 else "both_bind_state_dependent")


def summarise(recs, reps=200, seed=0):
    out = {"by_cbar": {}}
    for mode in CBAR_MODES:
        o = {"all": _share(recs, mode),
             "develop_share_ci95": _boot_share(recs, reps, seed, mode),
             "by_turn": {}, "by_hand": {}}
        for bands, key, get in ((TURN_BANDS, "by_turn", lambda r: r["turn"]),
                                (HAND_BANDS, "by_hand", lambda r: r["hand"])):
            for _lo, _hi, name in bands:
                sub = [r for r in recs if band_of(get(r), bands) == name]
                if len(sub) >= 200:
                    o[key][name] = _share(sub, mode)
        if o["all"]:
            o["verdict"] = _verdict(o["all"]["develop_share"])
        out["by_cbar"][mode] = o
    vs = {m: out["by_cbar"][m].get("verdict") for m in CBAR_MODES}
    out["verdict_by_cbar"] = vs
    out["verdicts_agree"] = bool(len(set(vs.values())) == 1 and None not in vs.values())
    out["verdict"] = (vs["row_don"] if out["verdicts_agree"]
                      else "cbar_mode_changes_the_answer")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--theta-mode", default="const", choices=("const", "board", "max"))
    ap.add_argument("--boot-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nu-mode", default=None, choices=("base", "pair"),
                    help="`ν` の形（省略時は `theory_order.NU_MODE`＝2026-09-15 から `pair`）。"
                         "**T26 の 2026-09-14 の数字は `base`**——比べるときは明示する")
    ap.add_argument("--play-kind", default="all", choices=PLAY_KINDS,
                    help="「作る枝」に数える PLAY 候補。`all`＝イベント・ステージも混ざる（既定）／"
                         "`char`＝体だけ（T26 の問いに答える方）")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    import theory_order as _TO
    if a.nu_mode is not None:
        _TO.set_nu_mode(a.nu_mode)
    global PLAY_KIND
    PLAY_KIND = a.play_kind
    t0 = time.time()
    recs, stats = collect(a.src, a.limit_games, theta_mode=a.theta_mode)
    res = {"stats": stats, "rows_used": len(recs),
           "frozen": {"lambda": LAM, "mu": MU, "theta": THETA, "theta_mode": a.theta_mode,
                      "nu_mode": _TO.NU_MODE, "play_kind": PLAY_KIND},
           "summary": summarise(recs, a.boot_reps, a.seed),
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
