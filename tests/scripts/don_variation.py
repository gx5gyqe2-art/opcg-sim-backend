"""WP `value-split`（`docs/search_efficiency_plan.md` §7.74・台帳 2+506）の付録: 教材の行で**自席の場のドンの総数**が
規則から決まる値からどれだけ外れているかを数える（コーディネータ追加 2026-11-03・次の WP の下見）。

仮説の下見: 「同じ盤面でドンだけ違う対が教材にほとんど無い＝ドン 1 枚の偏微分が教材から同定されにくい」。
規則: 先攻の 1 ターン目 1 枚・後攻の 1 ターン目 2 枚・以後は自分の手番ごとに +2・上限 10
（自分の n 手番目: 先攻 min(10, 2n−1)・後攻 min(10, 2n)）。外れ＝記録の総数 − 規則の値。

読むのは符号化の列だけ（dump v2 の `scalars`／`tokens`／`who`／`turn`／`kind`／`card_idx`）:
- 総数 ＝ アクティブ（`scalars[2]`）＋レスト（`scalars[3]`）＋リーダーの付与（`scalars[14]`×5）＋場の付与（`tokens[自場 5 枠, 2]`×5）。
  相手席は `scalars[4]`／`[5]`／`[15]`／`tokens[相場 5 枠, 2]`。列の並びは `opcg_sim/learned/encoder.py::encode`（v1・v2）と
  `n_rel_feat.S_COLS`（`attached_don`＝列 2・枠は 自L,相L,自場5,相場5,手札10）。fp16 なので丸めてから整数にする。
- 自席の先攻／後攻は手番の偶奇と `is_my_turn`（`scalars[11]`）から決める（奇数の手番の打ち手が先攻）。
- 自己点検: 総数＋ドンデッキ残（`scalars[66]`×10・実測で和が 10 になる率を出す）が 10 になる率。
- 色はリーダー（`card_idx[0]`）のカード DB の `colors`。

読み取り専用・1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。

    python tests/scripts/don_variation.py --in "~/n70f/n_records/n70f_w*_p[0-3]" --out n1_results/value_split/don_variation_n70f.json
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _bootstrap  # noqa: E402,F401

from opcg_sim.learned.vocab import load_db, shared_vocab  # noqa: E402

COL_MY_ACT, COL_MY_REST, COL_OP_ACT, COL_OP_REST = 2, 3, 4, 5
COL_IS_MY_TURN = 11
COL_MY_LDON, COL_OP_LDON = 14, 15          # リーダーの付与ドン /5（v2）
COL_MY_DONDECK, COL_OP_DONDECK = 66, 67    # ドンデッキ残 /10（v9・実測で「総数＋残＝10」が 93〜100% の列）
TOK_MY_FIELD, TOK_OP_FIELD = slice(2, 7), slice(7, 12)
TOK_ATT = 2                                # `S_COLS.attached_don`（/5）
DEV_RANGE = range(-3, 4)
KIND = {0: "main", 1: "window", 2: "commit"}


def rule(first, n):
    """自分の n 手番目（n ≥ 1）の規則の総数。"""
    return min(10, 2 * n - 1) if first else min(10, 2 * n)


def own_turns(first, turn):
    """全体の手番数 `turn`（1 始まり・先攻が奇数）のとき、その席がこれまでに迎えた自分の手番の数。"""
    return (turn + 1) // 2 if first else turn // 2


def leader_colors(db, inv, idx):
    cid = inv.get(int(idx))
    m = db.get_card(cid) if cid else None
    cols = sorted(getattr(x, "name", str(x)) for x in (getattr(m, "colors", ()) or ())) if m is not None else []
    return "/".join(cols) or "?"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", default=["~/n70f/n_records/n70f_w*_p[0-3]"])
    ap.add_argument("--out", default=os.path.join("n1_results", "value_split", "don_variation_n70f.json"))
    ap.add_argument("--limit", type=int, default=0, help="読む npz の上限（0＝全部・下見用）")
    a = ap.parse_args(argv)
    dirs = sorted(d for pat in a.src for d in glob.glob(os.path.expanduser(pat)))
    files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "n_record_*.npz")))
    if a.limit:
        files = files[:a.limit]
    db, voc = load_db(), shared_vocab()
    inv = {v: k for k, v in voc.items()}
    color_cache = {}

    dev = {s: collections.Counter() for s in ("all", "main_my", "window_opp", "commit")}    # 自席の外れ
    dev_opp = collections.Counter()                                                         # 相手席の外れ（all）
    dev_color = collections.defaultdict(collections.Counter)                                # 色 → 外れ（all）
    diff = {"my_turn": collections.Counter(), "opp_turn": collections.Counter()}            # 自席 − 相手席
    diff_rule = collections.Counter()                                                        # （自席 − 相手席）− 規則の差
    by_turn = collections.defaultdict(collections.Counter)                                   # (先攻?, 自分の手番 n) → 総数
    sums = {"n": 0, "x": 0.0, "y": 0.0, "xx": 0.0, "yy": 0.0, "xy": 0.0}                     # 総数 × 全体の手番数
    sums_n = dict(sums)                                                                       # 総数 × 自分の手番数
    check = {"rows": 0, "sum10": 0, "turn0": 0}
    for k, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        sc = d["scalars"].astype(np.float32)
        tok = d["tokens"].astype(np.float32)
        turn = d["turn"].astype(int)
        kind = d["kind"].astype(int)
        myturn = sc[:, COL_IS_MY_TURN] > 0.5
        me = np.rint(sc[:, COL_MY_ACT] + sc[:, COL_MY_REST] + sc[:, COL_MY_LDON] * 5 + tok[:, TOK_MY_FIELD, TOK_ATT].sum(1) * 5).astype(int)
        op = np.rint(sc[:, COL_OP_ACT] + sc[:, COL_OP_REST] + sc[:, COL_OP_LDON] * 5 + tok[:, TOK_OP_FIELD, TOK_ATT].sum(1) * 5).astype(int)
        deck_me = np.rint(sc[:, COL_MY_DONDECK] * 10).astype(int)
        first = (turn % 2 == 1) == myturn          # 奇数の手番の打ち手が先攻
        lead = d["card_idx"][:, 0]
        for i in range(len(sc)):
            check["rows"] += 1
            if turn[i] <= 0:
                check["turn0"] += 1
                continue
            check["sum10"] += int(me[i] + deck_me[i] == 10)
            n_me = own_turns(bool(first[i]), int(turn[i]))
            n_op = own_turns(not first[i], int(turn[i]))
            r_me = rule(bool(first[i]), n_me) if n_me >= 1 else 0
            r_op = rule(not first[i], n_op) if n_op >= 1 else 0
            dv = int(me[i]) - r_me
            dvo = int(op[i]) - r_op
            dev["all"][dv] += 1
            dev_opp[dvo] += 1
            if kind[i] == 0 and myturn[i]:
                dev["main_my"][dv] += 1
            elif kind[i] == 1 and not myturn[i]:
                dev["window_opp"][dv] += 1
            elif kind[i] == 2:
                dev["commit"][dv] += 1
            li = int(lead[i])
            if li not in color_cache:
                color_cache[li] = leader_colors(db, inv, li)
            dev_color[color_cache[li]][dv] += 1
            dd = int(me[i]) - int(op[i])
            diff["my_turn" if myturn[i] else "opp_turn"][dd] += 1
            diff_rule[dd - (r_me - r_op)] += 1
            by_turn[(bool(first[i]), n_me)][int(me[i])] += 1
            for S, x in ((sums, int(turn[i])), (sums_n, n_me)):
                S["n"] += 1; S["x"] += x; S["y"] += int(me[i]); S["xx"] += x * x; S["yy"] += int(me[i]) ** 2; S["xy"] += x * int(me[i])
        if (k + 1) % 100 == 0:
            print(f"[{k + 1}/{len(files)}] rows {check['rows']}", flush=True)

    def hist(c):
        n = sum(c.values())
        out = {str(v): round(c.get(v, 0) / max(n, 1), 4) for v in DEV_RANGE}
        out["<-3"] = round(sum(x for v, x in c.items() if v < -3) / max(n, 1), 4)
        out[">3"] = round(sum(x for v, x in c.items() if v > 3) / max(n, 1), 4)
        out["n"] = n
        out["nonzero"] = round(1 - c.get(0, 0) / max(n, 1), 4)
        out["mean"] = round(sum(v * x for v, x in c.items()) / max(n, 1), 4)
        return out

    def corr(S):
        n = S["n"]
        if n < 2:
            return None
        cov = S["xy"] / n - (S["x"] / n) * (S["y"] / n)
        vx = S["xx"] / n - (S["x"] / n) ** 2
        vy = S["yy"] / n - (S["y"] / n) ** 2
        return round(cov / np.sqrt(vx * vy), 4) if vx > 0 and vy > 0 else None

    turns = {}
    for (fst, n), c in sorted(by_turn.items()):
        tot = sum(c.values())
        vals = np.array(sorted(c)); w = np.array([c[v] for v in vals], float)
        mu = float((vals * w).sum() / tot); sd = float(np.sqrt(((vals - mu) ** 2 * w).sum() / tot))
        mode = int(vals[w.argmax()])
        turns[f"{'先攻' if fst else '後攻'} n={n}"] = {"rows": tot, "rule": rule(fst, n), "mean": round(mu, 3), "std": round(sd, 3),
                                                      "mode": mode, "mode_share": round(float(w.max() / tot), 4),
                                                      "hist": {str(int(v)): int(c[v]) for v in vals}}
    res = {"files": len(files), "rows": check["rows"], "turn0_rows": check["turn0"],
           "sum10_share": round(check["sum10"] / max(check["rows"] - check["turn0"], 1), 4),
           "dev_me": {s: hist(c) for s, c in dev.items()}, "dev_opp_all": hist(dev_opp),
           "dev_me_by_color": {k: hist(c) for k, c in sorted(dev_color.items(), key=lambda kv: -sum(kv[1].values()))},
           "diff_me_minus_opp": {s: hist(c) for s, c in diff.items()}, "diff_minus_rule_diff": hist(diff_rule),
           "corr_total_vs_turn": corr(sums), "corr_total_vs_own_turn_n": corr(sums_n), "by_turn": turns}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"files {len(files)} rows {check['rows']}（手番 0 を除く {check['rows'] - check['turn0']}）総数＋ドンデッキ残＝10 の率 {res['sum10_share']}")
    for s, h in res["dev_me"].items():
        print(f"自席の外れ [{s}] n {h['n']} 外れ≠0 {h['nonzero']:.3f} 平均 {h['mean']:+.3f} 分布 " + " ".join(f"{v}:{h[str(v)]:.3f}" for v in DEV_RANGE))
    print(f"相手席の外れ [all] 外れ≠0 {res['dev_opp_all']['nonzero']:.3f}")
    for k, h in res["dev_me_by_color"].items():
        print(f"  色 {k}: n {h['n']} 外れ≠0 {h['nonzero']:.3f} 平均 {h['mean']:+.3f} −1 {h['-1']:.3f} +1 {h['1']:.3f}")
    for s, h in res["diff_me_minus_opp"].items():
        print(f"自席 − 相手席 [{s}] 分布 " + " ".join(f"{v}:{h[str(v)]:.3f}" for v in DEV_RANGE))
    h = res["diff_minus_rule_diff"]
    print("（自席 − 相手席）− 規則の差 分布 " + " ".join(f"{v}:{h[str(v)]:.3f}" for v in DEV_RANGE))
    print(f"相関 総数×全体の手番数 {res['corr_total_vs_turn']} 総数×自分の手番数 {res['corr_total_vs_own_turn_n']}")
    for k, t in turns.items():
        print(f"  {k}: 行 {t['rows']} 規則 {t['rule']} 平均 {t['mean']} 標準偏差 {t['std']} 最頻 {t['mode']}（{t['mode_share']:.3f}）")
    print("DON_VARIATION_DONE")


if __name__ == "__main__":
    main()
