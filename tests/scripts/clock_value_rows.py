#!/usr/bin/env python3
"""**候補 `OPCG_CLOCK_VALUE` の行ごとの器**（2026-10-10・`docs/reports/2026-10-10_clock_value.md` §1.5）。

`theory_bridge.py` を `OPCG_PRICE_DUMP=<jsonl>` で回した写し（行ごとに `[t, w, 打った手, 最善, κ, 打った手の型, 最善の型, 値段の付いた候補の数]`）
から、腕ごとに次を出す（新しい式は無い・`theory_bridge` の `s` と `shadow_forbid` の線をそのまま数える）:

* **打った手と理論の最善の差**（通貨・手札 1 枚の価値 μ で割る）の分布——行は「打った手に値段が付き、値段の付いた候補が 2 本以上」
  （`theory_bridge` の無言でない行と同じ規約）。
* **禁止率**（`shadow_forbid` の 1 倍の線＝差 ≥ μ）と、打った手の型ごとの率・禁止された行で理論が薦めた型の内訳。
  T18 の器は今の既定で測れない純付与（`attach`）を外さなかったので、付与を除いた率も並べる。
* 2 つの写しを渡すと（同じ記録・同じ行の並び）、行ごとに「理論の最善の型」が一致する割合と、両方で禁止される行の数。
* 局ごとの攻め側の逸脱の和の AUC（全部・最後のターンを落とす・攻撃の候補を外す）と、攻撃の値段の上乗せの診断（報告 §2.2）。

使い方:

    python tests/scripts/clock_value_rows.py --dump base.jsonl [--dump2 clock.jsonl] [--json out.json]
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from theory_rs import MU  # noqa: E402

LINE = 1.0                      # T18 の線（手札 1 枚の価値の 1 倍・`2026-10-10_t18_forbid.md` §3）


def load(path, with_z=False):
    out, zs = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            zs[j["seed"]] = j.get("z")
            for r in j["rows"]:
                out.append((j["seed"],) + tuple(r))
    return (out, zs) if with_z else out


def game_auc(rows, zs, drop_last=False, no_attack=False):
    """攻め側の逸脱の和 `ΔS_atk`（帳簿の単位＝× κ・席 0 − 席 1）の AUC（`theory_bridge` の `dS_atk` と同じ行の規約）。
    `drop_last` は局の最後のターン（とどめの一撃）の行を落とす（§0.7 の節目の規約）。`no_attack` は攻撃の候補を外して比べる
    （攻撃を打った行も外す・攻撃の値段の上乗せ〔報告 §2.2〕を抜いた読み）。"""
    from theory_bridge import auc
    by = {}
    for r in rows:
        by.setdefault(r[0], []).append(r)
    xs, ys = [], []
    for seed, rs in by.items():
        z = (zs or {}).get(seed) or {}
        if z.get("0") is None:
            continue
        tmax = max(r[1] for r in rs)
        ds = 0.0
        for r in rs:
            if not judged(r) or (drop_last and r[1] == tmax):
                continue
            best = r[4]
            if no_attack:
                if r[6] == "attack" or len(r) < 10:
                    continue
                vs = [v for f, v in r[9] if v is not None and f != "attack"]
                if len(vs) < 2:
                    continue
                best = max(vs)
            ds += (r[3] - best) * r[5] * (1.0 if r[2] == 0 else -1.0)
        xs.append(ds)
        ys.append(float(z["0"]))
    a = auc(xs, ys) if len(xs) >= 2 else None
    return {"games": len(xs), "auc": None if a is None else round(float(a), 4)}


def attack_premium(rows, zs):
    """攻撃の値段の上乗せ（報告 §2.2・診断だけ）: 攻撃と攻撃以外の候補の値段（× κ）の平均の差・行の中の値段の幅の中央・
    上乗せを攻撃の候補から引くと最善の型／禁止が変わる行の数・勝った席と負けた席の攻撃の値段の平均・
    勝った席の行で勝ち負けの差だけを引くと最善の型が変わる行の数。"""
    rs = [r for r in rows if judged(r) and len(r) >= 10]
    if not rs:
        return None
    atk, oth, spread = [], [], []
    side = {True: [], False: []}
    for r in rs:
        vs = [v * r[5] for f, v in r[9] if v is not None]
        spread.append(max(vs) - min(vs))
        a = [v * r[5] for f, v in r[9] if v is not None and f == "attack"]
        atk += a
        oth += [v * r[5] for f, v in r[9] if v is not None and f != "attack"]
        z = ((zs or {}).get(r[0]) or {}).get(str(r[2]))
        if z is not None:
            side[z > 0.5] += a
    prem = float(np.mean(atk) - np.mean(oth)) if atk and oth else 0.0

    def flips(sub, d):
        ch = fch = ab0 = ab1 = 0
        for r in sub:
            c = [(f, v * r[5] - (d if f == "attack" else 0.0)) for f, v in r[9] if v is not None]
            b = max(c, key=lambda x: x[1])
            pv = r[3] * r[5] - (d if r[6] == "attack" else 0.0)
            ch += b[0] != r[7]
            ab0 += r[7] == "attack"
            ab1 += b[0] == "attack"
            fch += ((r[4] - r[3]) / MU >= LINE) != ((b[1] - pv) / r[5] / MU >= LINE if r[5] > 0 else False)
        return {"rows": len(sub), "best_family_changes": ch, "forbid_changes": fch,
                "attack_best_before": ab0, "attack_best_after": ab1}

    out = {"premium_kappa": round(prem, 5), "median_row_spread_kappa": round(float(np.median(spread)), 5),
           "remove_premium": flips(rs, prem)}
    if side[True] and side[False]:
        gap = float(np.mean(side[True]) - np.mean(side[False]))
        wins = [r for r in rs if (((zs or {}).get(r[0]) or {}).get(str(r[2])) or 0) > 0.5]
        out.update({"winner_attack_mean_kappa": round(float(np.mean(side[True])), 5),
                    "loser_attack_mean_kappa": round(float(np.mean(side[False])), 5),
                    "winner_minus_loser_kappa": round(gap, 5),
                    "remove_winner_gap_on_winner_rows": flips(wins, gap)})
    return out


def family_levels(rows):
    """候補の型ごとの値段（帳簿の単位＝× κ）の平均と、行の中の順位（最善との差の平均）——型ごとの水準の偏りの診断。"""
    acc = {}
    for r in rows:
        if len(r) < 10 or not judged(r):
            continue
        for fam, v in r[9]:
            if v is None:
                continue
            e = acc.setdefault(fam, [0, 0.0, 0.0])
            e[0] += 1
            e[1] += v * r[5]
            e[2] += (r[4] - v) * r[5]
    return {k: {"n": e[0], "mean_kappa": round(e[1] / e[0], 5), "gap_to_best_kappa": round(e[2] / e[0], 5)}
            for k, e in sorted(acc.items(), key=lambda kv: -kv[1][0])}


def judged(r):
    """`theory_bridge` の無言でない行（打った手に値段・値段の付いた候補 2 本以上）。"""
    return r[3] is not None and r[4] is not None and r[8] >= 2


def summarise(rows, mu=MU):
    rs = [r for r in rows if judged(r)]
    dev = np.array([(r[4] - r[3]) / mu for r in rs]) if rs else np.zeros(0)
    out = {"rows": len(rows), "judged": len(rs)}
    if not rs:
        return out
    out["dev_mu"] = {"mean": round(float(dev.mean()), 4),
                     "q50": round(float(np.quantile(dev, 0.5)), 4), "q75": round(float(np.quantile(dev, 0.75)), 4),
                     "q90": round(float(np.quantile(dev, 0.9)), 4), "q95": round(float(np.quantile(dev, 0.95)), 4),
                     "share_zero": round(float(np.mean(dev <= 1e-9)), 4),
                     "share_ge_half": round(float(np.mean(dev >= 0.5)), 4),
                     "share_ge_1": round(float(np.mean(dev >= 1.0)), 4),
                     "share_ge_2": round(float(np.mean(dev >= 2.0)), 4)}
    forb = dev >= LINE
    out["forbid_rate"] = round(float(forb.mean()), 4)
    na = [i for i, r in enumerate(rs) if r[6] != "attach"]
    out["forbid_rate_excl_attach"] = round(float(np.mean([forb[i] for i in na])), 4) if na else None
    fam = {}
    for i, r in enumerate(rs):
        e = fam.setdefault(r[6], [0, 0])
        e[0] += 1
        e[1] += int(forb[i])
    out["by_played_family"] = {k: {"rows": v[0], "forbid": v[1], "rate": round(v[1] / v[0], 4)}
                               for k, v in sorted(fam.items(), key=lambda kv: -kv[1][0])}
    best = {}
    for i, r in enumerate(rs):
        if forb[i]:
            best[r[7]] = best.get(r[7], 0) + 1
    out["forbidden_best_family"] = dict(sorted(best.items(), key=lambda kv: -kv[1]))
    out["best_family"] = {}
    for r in rs:
        out["best_family"][r[7]] = out["best_family"].get(r[7], 0) + 1
    # 帳簿の単位（× κ）での差＝ほぼ「打った手の後の勝率 − 一番良い手の後の勝率」（候補の腕では）
    devk = np.array([(r[4] - r[3]) * r[5] for r in rs])
    out["dev_kappa_mean"] = round(float(devk.mean()), 5)
    return out


def compare(a, b, mu=MU):
    if len(a) != len(b):
        return {"aligned": False, "n_a": len(a), "n_b": len(b)}
    both = [(x, y) for x, y in zip(a, b) if judged(x) and judged(y)]
    same_best = np.mean([x[7] == y[7] for x, y in both]) if both else None
    fa = [((x[4] - x[3]) / mu >= LINE) for x, _ in both]
    fb = [((y[4] - y[3]) / mu >= LINE) for _, y in both]
    return {"aligned": True, "both_judged": len(both),
            "same_best_family": None if same_best is None else round(float(same_best), 4),
            "forbid_both": int(sum(p and q for p, q in zip(fa, fb))),
            "forbid_only_a": int(sum(p and not q for p, q in zip(fa, fb))),
            "forbid_only_b": int(sum(q and not p for p, q in zip(fa, fb)))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", required=True)
    ap.add_argument("--dump2", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    ra, za = load(a.dump, with_z=True)
    out = {"a": summarise(ra)}
    out["a"]["game_auc"] = {"all": game_auc(ra, za), "drop_last": game_auc(ra, za, True),
                            "no_attack": game_auc(ra, za, no_attack=True)}
    out["a"]["family_levels"] = family_levels(ra)
    out["a"]["attack_premium"] = attack_premium(ra, za)
    if a.dump2:
        rb, zb = load(a.dump2, with_z=True)
        out["b"] = summarise(rb)
        out["b"]["game_auc"] = {"all": game_auc(rb, zb), "drop_last": game_auc(rb, zb, True),
                                "no_attack": game_auc(rb, zb, no_attack=True)}
        out["b"]["family_levels"] = family_levels(rb)
        out["b"]["attack_premium"] = attack_premium(rb, zb)
        out["compare"] = compare(ra, rb)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
