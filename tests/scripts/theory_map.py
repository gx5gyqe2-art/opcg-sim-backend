#!/usr/bin/env python3
"""**理論の目で記録を読む**（2026-10-11・`docs/reports/2026-10-11_theory_map.md`・1 回きりの測定・テストなし）。

理論のプログラムは変えない。既定の手の値段（打った後の 2 本の時計・並びを知らない形）の写し（`theory_bridge.py` を
`OPCG_PRICE_DUMP` で回したもの）と、交点の橋の行（`crossing_bridge.collect`・決着前の勝率の読みは `win_calib` と同じ相対の腕）を
記録の行と突き合わせ、報告 §1 で先に固定した区切りで 5 つの問いの表を出す。

使い方（1 本で全部・途中で止まっても出力の在る段は飛ばして再開する）:

    OPCG_LOG_SILENT=1 python tests/scripts/theory_map.py --in <記録> --out <map.json> [--work <dir>]

`--work` の既定は `<map.json>.work/`。段は (1) 手の値段の表（`<記録>/clock_succ/` が局の数だけ無ければ `clock_succ.py --in <記録>`）
→ (2) `theory_bridge.py`（`OPCG_PRICE_DUMP` の写し）→ (3) 時計の行 → (4) 集計。段ごとに回すときは:

    OPCG_LOG_SILENT=1 OPCG_PRICE_DUMP=<dir>/dump.jsonl python tests/scripts/theory_bridge.py --in <記録> --out <dir>/bridge.json
    OPCG_LOG_SILENT=1 python tests/scripts/theory_map.py clocks --in <記録> --out <dir>/clocks.json
    python tests/scripts/theory_map.py analyze --in <記録> --dump <dir>/dump.jsonl --clocks <dir>/clocks.json --out <dir>/map.json
"""

import argparse
import collections
import json
import math
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import tests._bootstrap  # noqa: E402,F401

# ---- 報告 §1 で固定した区切り（変えない） ----
GAP_BINS = ("先行2+", "先行1", "互角", "遅れ1", "遅れ2+")
TURN_BANDS = ("1-3", "4-6", "7+")
COLORS = ("赤", "緑", "青", "紫", "黒", "黄")
TYPES = ("attack", "attach", "play", "effect", "end")
MIN_GAMES = 30
BOOT = 2000
SEED = 0
HORIZON_FALLBACK = 6          # §1.5 (a): 地平が行に無いときの代わり（整数の時計 ≥ 6）


def gap_bin(n_me, n_opp):
    g = n_opp - n_me
    return "先行2+" if g >= 2 else "先行1" if g == 1 else "互角" if g == 0 else "遅れ1" if g == -1 else "遅れ2+"


def turn_band(t):
    return "1-3" if t <= 3 else "4-6" if t <= 6 else "7+"


def int_clock(tau, cap):
    return max(1, int(math.ceil(min(float(tau), cap))))


# ------------------------------------------------------------------ clocks
def cmd_clocks(a):
    import crossing_bridge as CB
    import lethal_rule as LR
    import theory_rs as TR
    import win_calib as WC
    from theory_rs import MU, THETA
    dirs = [a.src]
    CB.set_pre_settle_mode("off")                       # 分析は全行（決着の宣言は印で持つ）
    rows_out, _led, stats, turn_harm, _tc = CB.collect(dirs, a.games, THETA, MU, "const")
    sr = CB.sigma_rel_for(dirs)
    rs = WC.rows_of(rows_out, "theory")
    old = TR.CLOCK["W_ERR_MODE"]
    try:
        TR.set_w_err_mode("rel")
        p = WC.probs_of(rs, sr, "hyp")
    finally:
        TR.set_w_err_mode(old)
    for r, w in zip(rows_out, p):
        r["W"] = float(w)
    settled = LR.settled_map(dirs, a.games)
    out = {"sigma_rel": sr, "cap": CB.RACE_CAP, "rows_out": rows_out, "turn_harm": turn_harm,
           "settled": [[s, w, t, bool(d)] for (s, w, t), d in settled.items()], "stats": stats}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, default=float)
    print("clocks: rows_out %d turn_harm %d settled %d" % (len(rows_out), len(turn_harm), len(settled)))
    return 0


# ------------------------------------------------------------------ helpers
def family_of(sj):
    at = sj[0]
    if at == "ATTACK" or (at == "DON_BOX" and sj[2]):
        return "attack"
    if at in ("DON_BOX", "ATTACH_DON"):
        return "attach"
    return {"PLAY": "play", "ACTIVATE_MAIN": "effect", "TURN_END": "end"}.get(at, "other")


def load_dump(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            out[int(j["seed"])] = j
    return out


def leader_colors():
    d = json.load(open(os.path.join(_ROOT, "opcg_sim", "data", "opcg_cards.json"), encoding="utf-8"))
    return {c["number"]: c["色"].split("/") for c in d if c.get("種類") == "リーダー"}


class Roles:
    def __init__(self):
        from opcg_sim.loop import decks as D
        from opcg_sim.learned.n_rel_feat import ROLES, roles_of
        self.db, self.names, self.f, self.c = D.load_db(), ROLES, roles_of, {}

    def of(self, cid):
        if cid not in self.c:
            m = self.db.get_card(cid) if cid else None
            if m is None:
                self.c[cid] = ("?",)
            else:
                v = self.f(m)
                self.c[cid] = tuple(n for n, x in zip(self.names, v) if x > 0.5) or ("none",)
                if getattr(getattr(m, "type", None), "name", "") == "LEADER":
                    self.c[cid] = ("leader",)
        return self.c[cid]


def seat_leaders(meta, gm):
    """席ごとの実際のリーダー。**記録の `decks` が `"user"`（実デッキ）のとき `meta_games.json` の `leaders` は
    使われなかった乱数のリーダー対**（`decks.build_pair` は `la`／`lb` を無視して `user_deck_pair(seed)` を使う）
    ので、そちらから取る（2026-10-12 の訂正・報告 §訂正）。合成（`synth_roles` 等）は meta のまま。"""
    if meta.get("decks") == "user":
        from opcg_sim.loop import decks as D
        up = D.user_deck_pair(int(gm["seed"]))
        return [up[0][0], up[1][0]]
    return list(gm["leaders"])


def build_rows(src, dump, clocks):
    """記録の判断の行と写しの行を突き合わせ、行ごとの dict を返す（§1.2）。"""
    from opcg_sim.learned.train import plan_labels as PL
    meta = json.load(open(os.path.join(src, "meta_games.json"), encoding="utf-8"))
    games_meta = {int(g["seed"]): dict(g, leaders=seat_leaders(meta, g)) for g in meta["games"]}
    lcol = leader_colors()
    roles = Roles()
    cap = float(clocks["cap"])
    # 時計の行: (seed, who, t)
    ck = {}
    g2seed = {}
    for r in clocks["rows_out"]:
        ck[(int(r["seed"]), int(r["who"]), int(r["t"]))] = r
        g2seed[int(r["g"])] = int(r["seed"])
    th = {}
    for r in clocks["turn_harm"]:
        s = g2seed.get(int(r["g"]))
        if s is not None:
            th[(s, int(r["who"]), int(r["j"]))] = r
    settled = collections.defaultdict(lambda: 10 ** 9)      # (seed) → 最初に宣言されたターン（どちらの席でも）
    for s, w, t, d in clocks["settled"]:
        if d:
            settled[int(s)] = min(settled[int(s)], int(t))
    rows_all, align = [], collections.Counter()
    cols = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen")
    for rows, pol, _ex, L, ptr, idx in PL.iter_games([src], row_cols=cols, pol_cols=("pol_sig", "pol_cid", "pol_tcid")):
        seed = int(rows["seed"][idx[0]])
        gm = games_meta.get(seed)
        dj = dump.get(seed)
        if gm is None or dj is None:
            align["game_missing"] += 1
            continue
        u2c = PL.uuid_map(pol, L, ptr, idx)
        cand = []
        for i in idx:
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t) or int(rows["kind"][i]) != 0:
                continue
            k = int(L[i]); ch = int(rows["pol_chosen"][i])
            if k < 1:
                continue
            s0 = json.loads(pol["pol_sig"][ptr[i]])
            if s0[0] == "RESOLVE_EFFECT_SELECTION":
                continue
            if k < 2 or ch < 0 or ch >= k:
                continue
            sigs = [json.loads(pol["pol_sig"][j]) for j in range(ptr[i], ptr[i] + k)]
            cand.append((i, w, t, sigs, ch))
        drows = dj["rows"]
        if len(drows) != len(cand):
            align["game_len_mismatch"] += 1
            continue
        z = dj.get("z") or {}
        winner = gm.get("winner")
        first_turn_of = {}
        for (i, w, t, sigs, ch), dr in zip(cand, drows):
            fams = [family_of(s) for s in sigs]
            dfams = [c[0] for c in dr[8]]
            if int(dr[0]) != t or int(dr[1]) != w or fams != dfams:
                align["row_mismatch"] += 1
                continue
            align["rows"] += 1
            pv, bv, kap, ptype, btype, nsc = dr[2], dr[3], dr[4], dr[5], dr[6], dr[7]
            zw = 1.0 if (winner == ("p1" if w == 0 else "p2")) else 0.0
            lead = gm["leaders"][w]
            opp_lead = gm["leaders"][1 - w]
            c = ck.get((seed, w, t))
            sj = sigs[ch]
            role = ()
            target = None
            if ptype in ("play", "effect"):
                role = roles.of(u2c.get(sj[1]))
            elif ptype == "attack":
                tg = sj[2][0] if sj[2] else None
                tc = u2c.get(tg)
                target = "leader" if (tc and roles.of(tc) == ("leader",)) else ("char" if tc else "?")
            ft = first_turn_of.setdefault((w, t), i)
            rec = {"seed": seed, "w": w, "t": t, "step": int(rows["step"][i]), "z": zw, "leader": lead, "opp_leader": opp_lead,
                   "colors": lcol.get(lead, ["?"]), "opp_colors": lcol.get(opp_lead, ["?"]),
                   "ptype": ptype, "btype": btype, "pv": pv, "bv": bv, "kap": kap, "nsc": nsc,
                   "cands": dr[8], "role": list(role), "target": target,
                   "first_in_turn": ft == i, "settled_after": t >= settled[seed]}
            if c is not None:
                n_me, n_op = int_clock(c["tau_me_theory"], cap), int_clock(c["tau_opp_theory"], cap)
                rec.update({"gap": gap_bin(n_me, n_op), "n_me": n_me, "n_opp": n_op, "W": c["W"],
                            "tau_me": float(c["tau_me_theory"]), "tau_opp": float(c["tau_opp_theory"])})
            rows_all.append(rec)
    return rows_all, align, ck, th, games_meta, lcol


# ------------------------------------------------------------------ stats
def boot_games(units, fn, reps=BOOT, seed=SEED):
    """units: {seed: [unit, …]}。局を復元抽出して fn(list_of_units) の 95% 区間。"""
    keys = list(units)
    if not keys:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(keys), len(keys))
        us = [u for k in pick for u in units[keys[k]]]
        v = fn(us)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return None
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def follow_split(rows):
    """§1.4: 単位（局×席）ごとに理論どおりの割合 f → 2 群の勝率と重み。"""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["seed"], r["w"])].append(r)
    units = []
    for (s, w), rs in by.items():
        f = sum(1 for r in rs if r["ptype"] == r["btype"]) / len(rs)
        units.append((s, f, rs[0]["z"]))
    return units


def wr_of(units):
    sa = sum(f for _, f, _ in units); sb = sum(1 - f for _, f, _ in units)
    wa = sum(f * z for _, f, z in units) / sa if sa > 0 else float("nan")
    wb = sum((1 - f) * z for _, f, z in units) / sb if sb > 0 else float("nan")
    return wa, wb, sa, sb


def follow_table(rows, reps=BOOT):
    units = follow_split(rows)
    if not units:
        return None
    wa, wb, sa, sb = wr_of(units)
    bys = collections.defaultdict(list)
    for u in units:
        bys[u[0]].append(u)
    best = collections.Counter(r["btype"] for r in rows)
    played = collections.Counter(r["ptype"] for r in rows)
    out = {"rows": len(rows), "units": len(units), "games": len(bys),
           "best_mode": best.most_common(1)[0][0],
           "best_share": {k: round(v / len(rows), 3) for k, v in best.most_common()},
           "played_share": {k: round(v / len(rows), 3) for k, v in played.most_common()},
           "follow_rate": round(float(np.mean([f for _, f, _ in units])), 3),
           "wr_follow": round(wa, 4) if not math.isnan(wa) else None, "w_follow": round(sa, 1),
           "wr_other": round(wb, 4) if not math.isnan(wb) else None, "w_other": round(sb, 1)}
    if sa < MIN_GAMES or sb < MIN_GAMES:
        out["verdict"] = "局数不足"
        return out
    out["diff"] = round(wa - wb, 4)
    out["diff_ci"] = boot_games(bys, lambda us: (lambda r: r[0] - r[1])(wr_of(us)), reps)
    out["wr_follow_ci"] = boot_games(bys, lambda us: wr_of(us)[0], reps)
    out["wr_other_ci"] = boot_games(bys, lambda us: wr_of(us)[1], reps)
    ci = out["diff_ci"]
    out["verdict"] = ("確かな外れ" if ci and ci[1] < 0 else "外れの気配" if out["diff"] < 0
                      else "理論どおりが勝つ（確か）" if ci and ci[0] > 0 else "理論どおりが勝つ（気配）")
    return out


def pair_table(rows):
    """理論の最善の型 X × 打った型 Y の勝率（単位の重み: その単位の行のうち (X,Y) の割合）。"""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["seed"], r["w"])].append(r)
    acc = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for _k, rs in by.items():
        n = len(rs)
        c = collections.Counter((r["btype"], r["ptype"]) for r in rs)
        for xy, m in c.items():
            a = acc[xy]
            a[0] += m / n * rs[0]["z"]; a[1] += m / n; a[2] += m
    return {"%s→%s" % xy: {"w": round(v[1], 1), "rows": v[2], "wr": round(v[0] / v[1], 4) if v[1] > 0 else None}
            for xy, v in sorted(acc.items(), key=lambda kv: -kv[1][1])}


def share_by(rows, key):
    """単位ごとの割合の平均（§1.5 の 1）。"""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["seed"], r["w"])].append(r[key])
    if not by:
        return {}
    acc = collections.Counter()
    for v in by.values():
        c = collections.Counter(v)
        for k, m in c.items():
            acc[k] += m / len(v)
    return {k: round(acc[k] / len(by), 3) for k in TYPES if acc.get(k)}


def auc(x, y):
    from win_calib import auc_of
    return auc_of(x, y)


# ------------------------------------------------------------------ analyze
def cmd_analyze(a):
    dump = load_dump(a.dump)
    clocks = json.load(open(a.clocks, encoding="utf-8"))
    rows_all, align, ck, th, games_meta, lcol = build_rows(a.src, dump, clocks)
    from theory_rs import MU
    rows = [r for r in rows_all if r["pv"] is not None and r["nsc"] >= 2 and r["ptype"] in TYPES and r["btype"] in TYPES]
    out = {"record": a.src, "align": dict(align), "rows_all": len(rows_all), "rows": len(rows),
           "games": len({r["seed"] for r in rows}), "mu": MU}
    reps = a.boot

    # ---- 問い 1: 立ち回り（色 × ターン帯・リーダーは表だけ）
    q1 = {"by_color_band": {}, "by_band": {}, "by_leader": {}}
    for band in TURN_BANDS:
        rb = [r for r in rows if turn_band(r["t"]) == band]
        q1["by_band"][band] = {"units": len({(r["seed"], r["w"]) for r in rb}),
                               "best": share_by(rb, "btype"), "played": share_by(rb, "ptype")}
    for col in COLORS:
        for band in TURN_BANDS:
            rb = [r for r in rows if col in r["colors"] and turn_band(r["t"]) == band]
            nu = len({(r["seed"], r["w"]) for r in rb})
            e = {"units": nu}
            if nu >= MIN_GAMES:
                e.update({"best": share_by(rb, "btype"), "played": share_by(rb, "ptype")})
            else:
                e["verdict"] = "局数不足"
            q1["by_color_band"]["%s|%s" % (col, band)] = e
    q1["by_deck_band"] = {}
    for ld in sorted({r["leader"] for r in rows}):
        for band in TURN_BANDS:
            rb = [r for r in rows if r["leader"] == ld and turn_band(r["t"]) == band]
            nu = len({(r["seed"], r["w"]) for r in rb})
            if nu >= MIN_GAMES:
                q1["by_deck_band"]["%s|%s" % (ld, band)] = {"units": nu, "best": share_by(rb, "btype"), "played": share_by(rb, "ptype")}
    lead_units = collections.Counter()
    for r in {(r["seed"], r["w"]): r for r in rows}.values():
        lead_units[r["leader"]] += 1
    q1["by_leader"] = {"leaders": len(lead_units), "max_units": max(lead_units.values()) if lead_units else 0,
                       "leaders_with_30": sum(1 for v in lead_units.values() if v >= MIN_GAMES),
                       "verdict": "全リーダー局数不足" if all(v < MIN_GAMES for v in lead_units.values()) else "一部あり"}
    out["q1"] = q1

    # ---- 問い 2: 強い盤面（ターンの最初の行・時計の部品）
    q2 = collect_q2(clocks, th, games_meta, lcol, a.src)
    out["q2"] = q2

    # ---- 問い 3: 盤面 × 行動（時計の差 × ターン帯）
    rc = [r for r in rows if "gap" in r]
    out["rows_with_clock"] = len(rc)
    q3 = {"all": follow_table(rc, reps), "cells": {}, "cells_color": {}, "by_gap": {}, "by_band": {}}
    for g in GAP_BINS:
        q3["by_gap"][g] = follow_table([r for r in rc if r["gap"] == g], reps)
    for band in TURN_BANDS:
        q3["by_band"][band] = follow_table([r for r in rc if turn_band(r["t"]) == band], reps)
    for g in GAP_BINS:
        for band in TURN_BANDS:
            sub = [r for r in rc if r["gap"] == g and turn_band(r["t"]) == band]
            e = follow_table(sub, reps)
            if e is not None:
                e["pairs"] = pair_table(sub)
                e["marks"] = marks_of(sub, clocks["cap"])
            q3["cells"]["%s|%s" % (g, band)] = e
    for col in COLORS:
        for g in GAP_BINS:
            for band in TURN_BANDS:
                sub = [r for r in rc if col in r["colors"] and r["gap"] == g and turn_band(r["t"]) == band]
                if len({(r["seed"], r["w"]) for r in sub}) >= MIN_GAMES:
                    q3["cells_color"]["%s|%s|%s" % (col, g, band)] = follow_table(sub, reps)
    for col in COLORS:
        sub = [r for r in rc if col in r["colors"]]
        q3.setdefault("by_color", {})[col] = follow_table(sub, reps)
    q3["by_deck"] = {}
    q3["cells_deck"] = {}
    for ld in sorted({r["leader"] for r in rc}):
        sub = [r for r in rc if r["leader"] == ld]
        if len({(r["seed"], r["w"]) for r in sub}) >= MIN_GAMES:
            q3["by_deck"][ld] = follow_table(sub, reps)
            for g in GAP_BINS:
                for band in TURN_BANDS:
                    s2 = [r for r in sub if r["gap"] == g and turn_band(r["t"]) == band]
                    if len({(r["seed"], r["w"]) for r in s2}) >= MIN_GAMES:
                        q3["cells_deck"]["%s|%s|%s" % (ld, g, band)] = follow_table(s2, reps)
    q3["marks_all"] = marks_of(rc, clocks["cap"])
    out["q3"] = q3

    # ---- 問い 4: 相手の手（打った側の勝率の動き＝こちらの符号違い）
    out["q4"] = collect_q4(rows, ck, clocks, games_meta, lcol, MU)

    # ---- 問い 5（相手側）: 理論は小さな損と言ったのに負けている手
    out["q5_small_loss"] = collect_q5(rc, MU, reps)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("analyze: rows %d (clock %d) align %s" % (len(rows), len(rc), dict(align)))
    return 0


def marks_of(rows, cap):
    """§1.5 の既知の外れの印の割合: (a) 自分の時計が整数で 6 以上・(b) ターンの 2 手目以降・(c) 効果・出すが絡む行。"""
    if not rows:
        return None
    n = len(rows)
    a_ = sum(1 for r in rows if r.get("n_me", 0) >= HORIZON_FALLBACK or r.get("n_opp", 0) >= HORIZON_FALLBACK)
    b_ = sum(1 for r in rows if not r["first_in_turn"])
    c_ = sum(1 for r in rows if r["btype"] in ("effect", "play") or r["ptype"] in ("effect", "play"))
    return {"n": n, "long_clock": round(a_ / n, 3), "mid_turn": round(b_ / n, 3), "effect_or_play": round(c_ / n, 3)}


def collect_q2(clocks, th, games_meta, lcol, src):
    """ターンの最初の行（`rows_out`）で、時計の部品の「自分 − 相手」が勝敗をどれだけ並べるか（単位の AUC）と寄与の平均。"""
    cap = float(clocks["cap"])
    recs = []
    for r in clocks["rows_out"]:
        s, w = int(r["seed"]), int(r["who"])
        gm = games_meta.get(s)
        if gm is None:
            continue
        t_me = th.get((s, w, int(r["j_me"])))
        tm = min(float(r["tau_me_theory"]), cap); to = min(float(r["tau_opp_theory"]), cap)
        sl_me, sl_op = float(r["slope_theory_me"]), float(r["slope_theory_opp"])
        th_me, th_op = float(r["theta_me"]), float(r["theta_opp"])
        e = {"seed": s, "w": w, "t": int(r["t"]), "z": 1.0 if r["won"] else 0.0,
             "colors": lcol.get(gm["leaders"][w], ["?"]), "leader": gm["leaders"][w], "W": r.get("W"),
             "D": to - tm, "logratio": math.log(max(to, 1e-6) / max(tm, 1e-6)),
             # 耐久: 相手を倒すのに残っている量（自分の時計の分子）− 自分の耐久（相手の時計の分子）
             "d_theta": th_op - th_me,             # 正＝自分の耐久が相手の耐久より厚い（＝有利）
             "d_speed": sl_me - sl_op,             # 正＝自分が速い
             "d_board": float(r["board_me"]) - float(r["board_opp"]),
             "d_stock": float(r["stock_me"]) - float(r["stock_opp"]),
             "d_flow": float(r["flow_me"]) - float(r["flow_opp"]),
             "d_shield": -(float(r.get("shield_me") or 0) - float(r.get("shield_opp") or 0)),
             "lr_theta": math.log(max(th_op, 1e-6) / max(th_me, 1e-6)),
             "lr_speed": math.log(max(sl_me, 1e-6) / max(sl_op, 1e-6))}
        if t_me is not None:
            # 相手の耐久の中身（自分が削る側）: ライフ・手札・体（大きいほど自分に不利）
            e["opp_th_life"] = float(t_me["theta"]) - float(t_me["th_hand"]) - float(t_me["th_body"])
            e["opp_th_hand"] = float(t_me["th_hand"]); e["opp_th_body"] = float(t_me["th_body"])
            e["opp_life"] = float(t_me["d_life"]); e["opp_hand"] = float(t_me["d_hand"])
        recs.append(e)
    # 自分の耐久の中身は、同じ局の相手の行の turn_harm（相手のターンの最初・自分を削る側）を時間の近いもので引く
    by_seat = collections.defaultdict(list)
    for e in recs:
        by_seat[(e["seed"], e["w"])].append(e)
    for e in recs:
        prev = [x for x in by_seat.get((e["seed"], 1 - e["w"]), []) if x["t"] < e["t"] and "opp_th_life" in x]
        if prev and "opp_th_life" in e:
            p = prev[-1]
            e["d_life_part"] = p["opp_th_life"] - e["opp_th_life"]     # 正＝自分のライフの耐久が厚い
            e["d_hand_part"] = p["opp_th_hand"] - e["opp_th_hand"]
            e["d_body_part"] = p["opp_th_body"] - e["opp_th_body"]
            e["d_life_raw"] = p["opp_life"] - e["opp_life"]
            e["d_hand_raw"] = p["opp_hand"] - e["opp_hand"]
    feats = ["D", "logratio", "lr_theta", "lr_speed", "d_theta", "d_speed", "d_board", "d_stock", "d_flow",
             "d_shield", "d_life_part", "d_hand_part", "d_body_part", "d_life_raw", "d_hand_raw"]

    def block(rs):
        o = {"rows": len(rs), "units": len({(e["seed"], e["w"]) for e in rs})}
        if o["units"] < MIN_GAMES:
            o["verdict"] = "局数不足"
            return o
        for f in feats:
            xs = [(e[f], e["z"]) for e in rs if f in e]
            if len(xs) < 20:
                continue
            x = np.array([v for v, _ in xs]); z = np.array([v for _, v in xs])
            a = auc(x, z)
            o[f] = {"auc": round(a, 4) if a is not None else None, "mean": round(float(x.mean()), 4),
                    "mean_won": round(float(x[z > .5].mean()), 4) if (z > .5).any() else None,
                    "mean_lost": round(float(x[z <= .5].mean()), 4) if (z <= .5).any() else None}
        lr = np.array([e["logratio"] for e in rs]); a_ = np.array([e["lr_theta"] for e in rs]); b_ = np.array([e["lr_speed"] for e in rs])
        o["abs_share_theta"] = round(float(np.abs(a_).mean() / max(1e-9, np.abs(a_).mean() + np.abs(b_).mean())), 3)
        o["corr_logratio_static"] = round(float(np.corrcoef(lr, a_ + b_)[0, 1]), 3) if len(lr) > 2 else None
        return o

    q2 = {"all": block(recs), "by_band": {}, "by_color": {}, "by_color_band": {}}
    for band in TURN_BANDS:
        q2["by_band"][band] = block([e for e in recs if turn_band(e["t"]) == band])
    for col in COLORS:
        q2["by_color"][col] = block([e for e in recs if col in e["colors"]])
        for band in TURN_BANDS:
            q2["by_color_band"]["%s|%s" % (col, band)] = block([e for e in recs if col in e["colors"] and turn_band(e["t"]) == band])
    q2["by_deck"] = {}
    q2["by_deck_band"] = {}
    for ld in sorted({e["leader"] for e in recs}):
        b = block([e for e in recs if e["leader"] == ld])
        if b.get("verdict") is None:
            q2["by_deck"][ld] = b
            for band in TURN_BANDS:
                q2["by_deck_band"]["%s|%s" % (ld, band)] = block([e for e in recs if e["leader"] == ld and turn_band(e["t"]) == band])
    # 時計の差の区切りごとの実勝率（強い盤面の土台）
    wr = {}
    gb = collections.defaultdict(list)
    for r in clocks["rows_out"]:
        g = gap_bin(int_clock(r["tau_me_theory"], cap), int_clock(r["tau_opp_theory"], cap))
        gb[(g, turn_band(int(r["t"])))].append((int(r["seed"]), 1.0 if r["won"] else 0.0, r.get("W")))
    for (g, band), xs in gb.items():
        units = collections.defaultdict(list)
        for s, z, w in xs:
            units[s].append((z, w))
        zs = [z for s, z, w in xs]; ws = [w for s, z, w in xs if w is not None]
        wr["%s|%s" % (g, band)] = {"rows": len(xs), "games": len(units), "win_rate": round(float(np.mean(zs)), 4),
                                    "theory_W": round(float(np.mean(ws)), 4) if ws else None}
    q2["gap_band_winrate"] = wr
    return q2


def collect_q4(rows, ck, clocks, games_meta, lcol, mu):
    """相手の手を打った側の勝率の動き（値段 × κ）で並べる・型／役割／色ごと＋ターン単位の時計の動き。"""
    out = {"by_type": {}, "by_role": {}, "by_target": {}, "by_type_color": {}, "turn_clock": {}}

    def agg(rs):
        if not rs:
            return None
        dW = np.array([r["pv"] * r["kap"] for r in rs])
        dev = np.array([(r["bv"] - r["pv"]) / mu for r in rs])
        units = len({(r["seed"], r["w"]) for r in rs})
        return {"rows": len(rs), "units": units, "dW_mean": round(float(dW.mean()), 4),
                "dW_p10": round(float(np.percentile(dW, 10)), 4), "dev_mean": round(float(dev.mean()), 3),
                "forbid_rate": round(float((dev >= 1.0).mean()), 3),
                "mover_win": round(float(np.mean([r["z"] for r in rs])), 4),
                "verdict": None if units >= MIN_GAMES else "局数不足"}
    for ty in TYPES:
        out["by_type"][ty] = agg([r for r in rows if r["ptype"] == ty])
    roles = collections.defaultdict(list)
    for r in rows:
        for ro in r["role"]:
            roles["%s:%s" % (r["ptype"], ro)].append(r)
    out["by_role"] = {k: agg(v) for k, v in sorted(roles.items())}
    for tg in ("leader", "char"):
        out["by_target"][tg] = agg([r for r in rows if r["ptype"] == "attack" and r["target"] == tg])
    for col in COLORS:
        for ty in TYPES:
            out["by_type_color"]["%s|%s" % (col, ty)] = agg([r for r in rows if col in r["colors"] and r["ptype"] == ty])
    out["by_type_deck"] = {}
    for ld in sorted({r["leader"] for r in rows}):
        if len({(r["seed"], r["w"]) for r in rows if r["leader"] == ld}) >= MIN_GAMES:
            for ty in TYPES:
                out["by_type_deck"]["%s|%s" % (ld, ty)] = agg([r for r in rows if r["leader"] == ld and r["ptype"] == ty])
    # 打った手のうち、打った側の勝率を最も下げた手（相手から見て「まずい」の逆＝相手にとっての好手は dW 大）
    # ターン単位: 相手のターン t（席 o）の最初 → 次の自分のターン t+1（席 me）の最初で、自分の時計・相手の時計の動き
    cap = float(clocks["cap"])
    tc = collections.defaultdict(list)
    turn_moves = collections.defaultdict(list)
    for r in rows:
        turn_moves[(r["seed"], r["w"], r["t"])].append(r)
    for (s, o, t), ms in turn_moves.items():
        a = ck.get((s, o, t)); b = ck.get((s, 1 - o, t + 1))
        if a is None or b is None:
            continue
        # a は相手（席 o）の側: tau_me＝相手の時計・tau_opp＝自分の時計。b は自分の側。
        my0, op0 = min(a["tau_opp_theory"], cap), min(a["tau_me_theory"], cap)
        my1, op1 = min(b["tau_me_theory"], cap), min(b["tau_opp_theory"], cap)
        kinds = set()
        for m in ms:
            kinds.add(m["ptype"])
            for ro in m["role"]:
                if ro in ("removal", "reduction", "lock", "blocker", "big", "draw"):
                    kinds.add("%s:%s" % (m["ptype"], ro))
            if m["ptype"] == "attack" and m["target"]:
                kinds.add("attack:%s" % m["target"])
        e = (s, my1 - my0, op1 - op0, 1.0 - ms[0]["z"])      # 自分の時計の延び・相手の時計の縮み（負＝縮む）・自分の勝ち
        tc["all"].append(e)
        for k in kinds:
            tc[k].append(e)
    for k, es in tc.items():
        units = len({x[0] for x in es})
        out["turn_clock"][k] = {"turns": len(es), "games": units,
                                "my_clock_ext": round(float(np.mean([x[1] for x in es])), 3),
                                "opp_clock_change": round(float(np.mean([x[2] for x in es])), 3),
                                "my_win": round(float(np.mean([x[3] for x in es])), 4),
                                "verdict": None if units >= MIN_GAMES else "局数不足"}
    return out


def collect_q5(rows, mu, reps):
    """理論は小さな損（逸脱 < 手札 1 枚）と言った手を、型・役割・時計の差ごとに「実際の勝率 − ターンの最初の理論の勝率」で並べる。"""
    small = [r for r in rows if (r["bv"] - r["pv"]) / mu < 1.0 and r.get("W") is not None]
    big = [r for r in rows if (r["bv"] - r["pv"]) / mu >= 1.0 and r.get("W") is not None]
    groups = collections.defaultdict(list)
    for r in small:
        groups["type:" + r["ptype"]].append(r)
        groups["type×gap:%s|%s" % (r["ptype"], r["gap"])].append(r)
        for ro in r["role"]:
            groups["role:%s:%s" % (r["ptype"], ro)].append(r)
        if r["target"]:
            groups["attack:" + r["target"]].append(r)

    def res(rs):
        by = collections.defaultdict(list)
        for r in rs:
            by[(r["seed"], r["w"])].append(r["z"] - r["W"])
        return float(np.mean([np.mean(v) for v in by.values()])) if by else None

    out = {"small_rows": len(small), "big_rows": len(big), "groups": {}}
    for nm, rs in [("ALL_small", small), ("ALL_big", big)] + sorted(groups.items()):
        units = {(r["seed"], r["w"]) for r in rs}
        e = {"rows": len(rs), "units": len(units)}
        if len(units) < MIN_GAMES:
            e["verdict"] = "局数不足"
        else:
            bys = collections.defaultdict(list)
            for r in rs:
                bys[r["seed"]].append(r)
            e["resid"] = round(res(rs), 4)
            e["resid_ci"] = boot_games(bys, res, reps)
            e["win"] = round(float(np.mean([r["z"] for r in rs])), 4)
            e["W"] = round(float(np.mean([r["W"] for r in rs])), 4)
        out["groups"][nm] = e
    return out


def cmd_run(a):
    """1 本で全部（表 → 写し → 時計 → 集計）。在る出力は作り直さない。"""
    import subprocess
    work = a.work or (a.out + ".work")
    os.makedirs(work, exist_ok=True)
    env = dict(os.environ, OPCG_LOG_SILENT="1")
    meta = json.load(open(os.path.join(a.src, "meta_games.json"), encoding="utf-8"))
    n_games = len(meta["games"]) if not a.games else a.games
    tdir = os.path.join(a.src, "clock_succ")
    have = len([f for f in os.listdir(tdir) if f.endswith(".npz") and not f.endswith(".tmp.npz")]) if os.path.isdir(tdir) else 0
    if have < n_games:
        print("[1/4] 手の値段の表（%d/%d 局在る）" % (have, n_games), flush=True)
        cmd = [sys.executable, os.path.join(_HERE, "clock_succ.py"), "--in", a.src]
        if a.games:
            cmd += ["--games", str(a.games)]
        subprocess.run(cmd, env=env, check=True)
    dump = os.path.join(work, "dump.jsonl")
    if not os.path.exists(dump + ".ok"):
        print("[2/4] theory_bridge（写し）", flush=True)
        cmd = [sys.executable, os.path.join(_HERE, "theory_bridge.py"), "--in", a.src, "--out", os.path.join(work, "bridge.json")]
        if a.games:
            cmd += ["--limit-games", str(a.games)]
        with open(os.path.join(work, "bridge.log"), "w") as lf:
            subprocess.run(cmd, env=dict(env, OPCG_PRICE_DUMP=dump), check=True, stdout=lf, stderr=subprocess.STDOUT)
        open(dump + ".ok", "w").close()
    clocks = os.path.join(work, "clocks.json")
    if not os.path.exists(clocks):
        print("[3/4] 時計の行", flush=True)
        cmd_clocks(argparse.Namespace(src=a.src, games=a.games, out=clocks))
    print("[4/4] 集計", flush=True)
    return cmd_analyze(argparse.Namespace(src=a.src, dump=dump, clocks=clocks, boot=a.boot, out=a.out))


# ------------------------------------------------------------------ turns（2026-10-12・`2026-10-12_turn_plans.md` §1）
PLAN_CATS = ("a", "b", "c", "d", "e", "f")


def turn_category(fams):
    """ターンの手の型の並び → (a)〜(f)・効果の有無（報告 2026-10-12 §1.2）。"""
    atk = [i for i, x in enumerate(fams) if x == "attack"]
    play = [i for i, x in enumerate(fams) if x == "play"]
    eff = any(x == "effect" for x in fams)
    if atk:
        if not play:
            return "a", eff
        return ("b" if play[0] < atk[0] else "c"), eff
    if play:
        return "d", eff
    if any(x in ("attach", "effect") for x in fams):
        return "e", eff
    return "f", eff


def turn_followed(rec, cat, fams):
    """(iii) の「理論どおり」: 勧めの型をそのターンのどこかで打ったか（終えるは何もせず終えた f のときだけ）。"""
    if rec == "end":
        return cat == "f"
    return rec in fams


def build_turns(src, dump, clocks):
    from opcg_sim.learned.train import plan_labels as PL
    rows_all, align, _ck, _th, games_meta, _lcol = build_rows(src, dump, clocks)
    first = {}
    for r in rows_all:
        if r["pv"] is None or r["nsc"] < 2 or r["ptype"] not in TYPES or r["btype"] not in TYPES:
            continue
        first.setdefault((r["seed"], r["w"], r["t"]), r)
    cols = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen")
    turns, cnt = [], collections.Counter()
    for rows, pol, _ex, L, ptr, idx in PL.iter_games([src], row_cols=cols, pol_cols=("pol_sig", "pol_cid", "pol_tcid")):
        seed = int(rows["seed"][idx[0]])
        seq = collections.OrderedDict()
        for i in idx:
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t) or int(rows["kind"][i]) != 0 or int(L[i]) < 1:
                continue
            if json.loads(pol["pol_sig"][ptr[i]])[0] == "RESOLVE_EFFECT_SELECTION":
                continue
            seq.setdefault((w, t), []).append(family_of(json.loads(rows["sig"][i])))
        for (w, t), fams in seq.items():
            cnt["turns"] += 1
            r0 = first.get((seed, w, t))
            if r0 is None:
                cnt["no_judged_row"] += 1
                continue
            if "gap" not in r0:
                cnt["no_clock"] += 1
                continue
            cat, eff = turn_category(fams)
            turns.append({"seed": seed, "w": w, "t": t, "z": r0["z"], "leader": r0["leader"], "colors": r0["colors"],
                          "gap": r0["gap"], "band": turn_band(t), "rec": r0["btype"], "cat": cat, "eff": eff,
                          "attacked": cat in ("a", "b", "c"), "followed": turn_followed(r0["btype"], cat, fams)})
    cnt["used"] = len(turns)
    return turns, dict(cnt), dict(align)


def split_table(turns, flag, reps=BOOT):
    """2 群（flag 真・偽）の勝率・重みは §1.4 と同じ（局×席の重み 1 を群の割合で分ける）。"""
    if not turns:
        return None
    rows = [dict(r, ptype="x" if flag(r) else "y", btype="x") for r in turns]
    e = follow_table(rows, reps)
    for k in ("best_mode", "best_share", "played_share"):
        e.pop(k, None)
    e["share_true"] = e.pop("follow_rate")
    e["turns"] = e.pop("rows")
    return e


def order_table(turns, reps=BOOT):
    """(ii): 攻撃したターンを a・b・c の 3 群に分け、勝率と b−a・c−a。"""
    turns = [r for r in turns if r["cat"] in ("a", "b", "c")]
    by = collections.defaultdict(list)
    for r in turns:
        by[(r["seed"], r["w"])].append(r)
    units = []
    for (s, w), rs in by.items():
        n = len(rs)
        units.append((s, {c: sum(1 for r in rs if r["cat"] == c) / n for c in "abc"}, rs[0]["z"]))
    if not units:
        return None

    def wr(us):
        o = {}
        for c in "abc":
            sw = sum(u[1][c] for u in us)
            o[c] = (sum(u[1][c] * u[2] for u in us) / sw) if sw > 0 else float("nan")
            o["w_" + c] = sw
        return o
    w0 = wr(units)
    out = {"turns": len(turns), "games": len({u[0] for u in units}),
           "count": {c: sum(1 for r in turns if r["cat"] == c) for c in "abc"}}
    for c in "abc":
        out["wr_" + c] = None if math.isnan(w0[c]) else round(w0[c], 4)
        out["w_" + c] = round(w0["w_" + c], 1)
    bys = collections.defaultdict(list)
    for u in units:
        bys[u[0]].append(u)
    for c in "bc":
        if w0["w_a"] < MIN_GAMES or w0["w_" + c] < MIN_GAMES:
            out["%s_minus_a" % c] = "局数不足"
            continue
        out["%s_minus_a" % c] = round(w0[c] - w0["a"], 4)
        out["%s_minus_a_ci" % c] = boot_games(bys, lambda us, c=c: (lambda o: o[c] - o["a"])(wr(us)), reps)
    return out


def cmd_turns(a):
    dump_p = a.dump or os.path.join(a.work, "dump.jsonl")
    clocks_p = a.clocks or os.path.join(a.work, "clocks.json")
    turns, cnt, align = build_turns(a.src, load_dump(dump_p), json.load(open(clocks_p, encoding="utf-8")))
    reps = a.boot
    decks = sorted({r["leader"] for r in turns})
    out = {"record": a.src, "align": align, "turn_counts": cnt}
    # 中身の件数（6 つ × 効果の有無）と、勧めごとの中身の内訳
    out["categories"] = {c: {"all": sum(1 for r in turns if r["cat"] == c),
                             "with_effect": sum(1 for r in turns if r["cat"] == c and r["eff"])} for c in PLAN_CATS}
    out["rec_by_cat"] = {rec: {c: sum(1 for r in turns if r["rec"] == rec and r["cat"] == c) for c in PLAN_CATS}
                         for rec in TYPES}
    # 勧めごと・中身ごとの勝率（単位の重み・参考）
    out["rec_cat_wr"] = {}
    for rec in TYPES:
        sub = [r for r in turns if r["rec"] == rec]
        for c in PLAN_CATS:
            e = split_table(sub, lambda r, c=c: r["cat"] == c, reps=0)
            if e:
                out["rec_cat_wr"]["%s|%s" % (rec, c)] = {"w": e["w_follow"], "wr": e["wr_follow"], "w_rest": e["w_other"], "wr_rest": e["wr_other"]}

    def blocks(sel, fn):
        o = {"all": fn(sel)}
        o["by_gap"] = {g: fn([r for r in sel if r["gap"] == g]) for g in GAP_BINS}
        o["by_band"] = {b: fn([r for r in sel if r["band"] == b]) for b in TURN_BANDS}
        o["cells"] = {"%s|%s" % (g, b): fn([r for r in sel if r["gap"] == g and r["band"] == b]) for g in GAP_BINS for b in TURN_BANDS}
        o["by_deck"] = {}
        for d in decks:
            sub = [r for r in sel if r["leader"] == d]
            if len({(r["seed"], r["w"]) for r in sub}) >= MIN_GAMES:
                o["by_deck"][d] = fn(sub)
        return o
    atk_rec = [r for r in turns if r["rec"] == "attack"]
    out["i_attack_rec"] = blocks(atk_rec, lambda ts: split_table(ts, lambda r: r["attacked"], reps))
    out["i_attack_rec_e_vs_rest"] = blocks(atk_rec, lambda ts: split_table(ts, lambda r: r["cat"] == "e", reps))
    o2 = {"all": order_table(atk_rec, reps)}
    o2["by_gap"] = {g: order_table([r for r in atk_rec if r["gap"] == g], reps) for g in GAP_BINS}
    o2["by_band"] = {b: order_table([r for r in atk_rec if r["band"] == b], reps) for b in TURN_BANDS}
    o2["by_deck"] = {d: order_table([r for r in atk_rec if r["leader"] == d], reps) for d in decks
                     if len({(r["seed"], r["w"]) for r in atk_rec if r["leader"] == d}) >= MIN_GAMES}
    o2["any_rec_all"] = order_table(turns, reps)
    out["ii_order"] = o2
    out["iii_follow"] = blocks(turns, lambda ts: split_table(ts, lambda r: r["followed"], reps))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("turns: %s" % cnt)
    return 0


# ------------------------------------------------------------------ deckcal（2026-10-12・`2026-10-12_turn_plans.md` §11）
DON_RULE_CAP = 10          # 規則のドン!!デッキの枚数（エネル〔OP15-058〕は「ドン‼デッキは 6 枚」）
DON_DECK_SIZE = {"OP15-058": 6}


def cmd_deckcal(a):
    """デッキ別の較正（ターン先頭・決着前）・役割の外れのデッキ切り分け・ドンの検算（理論の計画の段のドン vs 記録）。"""
    from opcg_sim.learned.train import plan_labels as PL
    dump_p = a.dump or os.path.join(a.work, "dump.jsonl")
    clocks_p = a.clocks or os.path.join(a.work, "clocks.json")
    clocks = json.load(open(clocks_p, encoding="utf-8"))
    from theory_rs import MU
    rows_all, _align, _ck, _th, games_meta, _lcol = build_rows(a.src, load_dump(dump_p), clocks)
    settled = collections.defaultdict(lambda: 10 ** 9)
    for s_, w_, t_, d_ in clocks["settled"]:
        if d_:
            settled[int(s_)] = min(settled[int(s_)], int(t_))
    focus = a.focus
    out = {"record": a.src, "focus": focus, "don_cap": a.don_cap}
    # 1. 較正（ターン先頭の行・決着前）
    cal = []
    for r in clocks["rows_out"]:
        s_, w_, t_ = int(r["seed"]), int(r["who"]), int(r["t"])
        gm = games_meta.get(s_)
        if gm is None or t_ >= settled[s_] or r.get("W") is None:
            continue
        cal.append({"seed": s_, "w": w_, "t": t_, "leader": gm["leaders"][w_], "W": float(r["W"]), "z": 1.0 if r["won"] else 0.0})

    def calrow(rs):
        if not rs:
            return None
        bys = collections.defaultdict(list)
        for x in rs:
            bys[x["seed"]].append(x)
        e = {"rows": len(rs), "games": len(bys), "W": round(float(np.mean([x["W"] for x in rs])), 4),
             "win": round(float(np.mean([x["z"] for x in rs])), 4)}
        e["win_minus_W"] = round(e["win"] - e["W"], 4)
        if len(bys) >= MIN_GAMES:
            e["ci"] = boot_games(bys, lambda xs: float(np.mean([x["z"] - x["W"] for x in xs])), a.boot)
        else:
            e["verdict"] = "局数不足"
        return e
    decks = sorted({x["leader"] for x in cal})
    out["calibration"] = {d: {"all": calrow([x for x in cal if x["leader"] == d]),
                              **{b: calrow([x for x in cal if x["leader"] == d and turn_band(x["t"]) == b]) for b in TURN_BANDS}}
                          for d in decks}
    out["calibration"]["ALL"] = {"all": calrow(cal), **{b: calrow([x for x in cal if turn_band(x["t"]) == b]) for b in TURN_BANDS}}
    # 2. 役割の外れ（問い 5 の相手側の表）を focus のデッキと他で分ける
    rc = [r for r in rows_all if r["pv"] is not None and r["nsc"] >= 2 and r["ptype"] in TYPES
          and r["btype"] in TYPES and "gap" in r]
    out["q5_focus"] = collect_q5([r for r in rc if r["leader"] == focus], MU, a.boot)
    out["q5_others"] = collect_q5([r for r in rc if r["leader"] != focus], MU, a.boot)
    out["q5_by_deck"] = {d: collect_q5([r for r in rc if r["leader"] == d], MU, a.boot)["groups"].get("ALL_small") for d in decks}
    # 3. ドンの検算: 計画の段 k（1・2）のドンの上限 min(10, 今の場のドン + 2k)（`curve.rs::caps_of`）と、記録の k ターン後の場のドン
    cols = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen")
    seats = collections.defaultdict(list)
    for rows, _pol, ex, _L, _ptr, idx in PL.iter_games([a.src], row_cols=cols, pol_cols=("pol_sig",),
                                                        extra_fn=lambda dd, n: {"sc": np.asarray(dd["scalars"])[:n].astype(np.float64)}):
        seed = int(rows["seed"][idx[0]])
        gm = games_meta.get(seed)
        if gm is None:
            continue
        seen = set()
        for i in idx:
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t) or int(rows["kind"][i]) != 0 or (w, t) in seen:
                continue
            seen.add((w, t))
            ld = gm["leaders"][w]
            size = DON_DECK_SIZE.get(ld, DON_RULE_CAP)
            field = size - int(round(float(ex["sc"][i, 66]) * 10))
            seats[(seed, w, ld)].append((t, field))
    don = collections.defaultdict(lambda: {1: [], 2: []})
    for (seed, w, ld), ts in seats.items():
        ts.sort()
        for j, (t, f0) in enumerate(ts):
            for k in (1, 2):
                if j + k < len(ts):
                    # `--don-cap room`（2026-10-12・`2026-10-12_don_deck.md`）: 上限を場のドン＋ドン‼デッキの残り（＝そのデッキのドン‼デッキの枚数）で切る候補の式
                    cap = min(DON_DECK_SIZE.get(ld, DON_RULE_CAP) if a.don_cap == "room" else DON_RULE_CAP, f0 + 2 * k)
                    don[ld][k].append((cap - ts[j + k][1], f0, ts[j + k][1], turn_band(t)))
    out["don_check"] = {}
    for ld, byk in don.items():
        e = {"deck_size": DON_DECK_SIZE.get(ld, DON_RULE_CAP)}
        for k, xs in byk.items():
            ov = np.array([x[0] for x in xs], float)
            e["step%d" % k] = {"n": len(xs), "over_mean": round(float(ov.mean()), 3),
                               "over_p10_p50_p90": [round(float(q), 1) for q in np.percentile(ov, [10, 50, 90])],
                               "share_over": round(float((ov > 0).mean()), 3), "share_under": round(float((ov < 0).mean()), 3),
                               "dist": {str(int(v)): int((ov == v).sum()) for v in sorted(set(ov.tolist()))},
                               "by_band": {b: round(float(np.mean([x[0] for x in xs if x[3] == b])), 3)
                                           for b in TURN_BANDS if any(x[3] == b for x in xs)}}
        out["don_check"][ld] = e
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("deckcal: cal rows %d" % len(cal))
    return 0


def cmd_paircal(a):
    """対面ごとの較正（2026-10-12・`2026-10-12_don_deck.md` §7）: 先手リーダー × 後手リーダーの組ごとに、席ごとの
    「実 − 理論」（ターン先頭の行・決着の宣言の前・局ごとの再抽出）。帯（1-3／4-6／7+）でも。"""
    clocks = json.load(open(a.clocks, encoding="utf-8"))
    meta = json.load(open(os.path.join(a.src, "meta_games.json"), encoding="utf-8"))
    games_meta = {int(g["seed"]): dict(g, leaders=seat_leaders(meta, g)) for g in meta["games"]}
    settled = collections.defaultdict(lambda: 10 ** 9)
    for s_, w_, t_, d_ in clocks["settled"]:
        if d_:
            settled[int(s_)] = min(settled[int(s_)], int(t_))
    rows = []
    for r in clocks["rows_out"]:
        s_, w_, t_ = int(r["seed"]), int(r["who"]), int(r["t"])
        gm = games_meta.get(s_)
        if gm is None or t_ >= settled[s_] or r.get("W") is None:
            continue
        rows.append({"seed": s_, "w": w_, "t": t_, "pair": "%s|%s" % tuple(gm["leaders"]), "W": float(r["W"]),
                     "z": 1.0 if r["won"] else 0.0})

    def cal(rs):
        if not rs:
            return None
        bys = collections.defaultdict(list)
        for x in rs:
            bys[x["seed"]].append(x)
        e = {"rows": len(rs), "games": len(bys), "W": round(float(np.mean([x["W"] for x in rs])), 4),
             "win": round(float(np.mean([x["z"] for x in rs])), 4)}
        e["win_minus_W"] = round(e["win"] - e["W"], 4)
        if len(bys) >= a.min_games:
            e["ci"] = boot_games(bys, lambda xs: float(np.mean([x["z"] - x["W"] for x in xs])), a.boot)
        else:
            e["verdict"] = "局数不足"
        return e
    out = {"record": a.src, "min_games": a.min_games, "pairs": {}}
    for pr in sorted({x["pair"] for x in rows}):
        sub = [x for x in rows if x["pair"] == pr]
        e = {"games": len({x["seed"] for x in sub})}
        for w in (0, 1):
            sw = [x for x in sub if x["w"] == w]
            e["seat%d" % w] = {"all": cal(sw), **{b: cal([x for x in sw if turn_band(x["t"]) == b]) for b in TURN_BANDS}}
        out["pairs"][pr] = e
    # 席のデッキ × 相手のデッキ（先手・後手をまとめる）
    out["deck_vs_deck"] = {}
    for x in rows:
        gm = games_meta[x["seed"]]
        x["me"], x["opp"] = gm["leaders"][x["w"]], gm["leaders"][1 - x["w"]]
    for me in sorted({x["me"] for x in rows}):
        for op in sorted({x["opp"] for x in rows}):
            sub = [x for x in rows if x["me"] == me and x["opp"] == op]
            if sub:
                out["deck_vs_deck"]["%s|%s" % (me, op)] = {"all": cal(sub), **{b: cal([x for x in sub if turn_band(x["t"]) == b]) for b in TURN_BANDS}}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("paircal: rows %d pairs %d" % (len(rows), len(out["pairs"])))
    return 0


def cmd_clockside(a):
    """どちらの時計が外れているか（2026-10-12・`2026-10-12_don_deck.md` §8）。ターン先頭・決着の宣言の前の行で、席のデッキ × 帯ごとに:
    - 自分の時計: 整数 `max(1, ⌈τ_me⌉)` − 実際に相手を倒すまでの自席ターン（勝った席＝打ち切りなし）。負けた席は打ち切り（実際は「少なくとも」）
      ＝読みが打ち切りより短い割合（「この局のうちに倒せる」と読んだのに倒せなかった）を数える。
    - 相手の時計: 整数 `max(1, ⌈τ_opp⌉)` − 実際に自分が倒されるまでの相手のターン（負けた席＝打ち切りなし）。勝った席は打ち切り
      ＝読みが打ち切りより短い割合（「この局のうちに倒される」と読んだのに倒されなかった）を数える。
    正＝読みが長い（攻めを低く／守りを高く読む）・負＝読みが短い（攻めを高く／守りを低く読む）。"""
    clocks = json.load(open(a.clocks, encoding="utf-8"))
    cap = float(clocks["cap"])
    meta = json.load(open(os.path.join(a.src, "meta_games.json"), encoding="utf-8"))
    games_meta = {int(g["seed"]): dict(g, leaders=seat_leaders(meta, g)) for g in meta["games"]}
    settled = collections.defaultdict(lambda: 10 ** 9)
    for s_, w_, t_, d_ in clocks["settled"]:
        if d_:
            settled[int(s_)] = min(settled[int(s_)], int(t_))
    rows = []
    for r in clocks["rows_out"]:
        s_, w_, t_ = int(r["seed"]), int(r["who"]), int(r["t"])
        gm = games_meta.get(s_)
        if gm is None or t_ >= settled[s_]:
            continue
        rows.append({"seed": s_, "t": t_, "leader": gm["leaders"][w_], "won": bool(r["won"]),
                     "me": int_clock(r["tau_me_theory"], cap) - int(r["t_me_act"]),
                     "opp": int_clock(r["tau_opp_theory"], cap) - int(r["t_opp_act"]),
                     "me_f": min(float(r["tau_me_theory"]), cap) - float(r["t_me_act"]),
                     "opp_f": min(float(r["tau_opp_theory"]), cap) - float(r["t_opp_act"])})

    def blk(rs):
        o = {}
        for side, full, cens in (("me", True, False), ("opp", False, True)):
            # 打ち切りなし: 自分の時計は勝った席・相手の時計は負けた席
            un = [x for x in rs if x["won"] == full]
            ce = [x for x in rs if x["won"] == cens]
            e = {"n": len(un), "games": len({x["seed"] for x in un})}
            if un:
                e["mean_int"] = round(float(np.mean([x[side] for x in un])), 3)
                e["mean_frac"] = round(float(np.mean([x[side + "_f"] for x in un])), 3)
                bys = collections.defaultdict(list)
                for x in un:
                    bys[x["seed"]].append(x)
                if len(bys) >= MIN_GAMES:
                    e["ci_int"] = boot_games(bys, lambda xs, side=side: float(np.mean([x[side] for x in xs])), a.boot)
                else:
                    e["verdict"] = "局数不足"
            e["censored_n"] = len(ce)
            if ce:
                # 読み（整数）が打ち切りの時刻より短い＝その局のうちに届くと読んだのに届かなかった
                e["censored_short_share"] = round(float(np.mean([x[side] < 0 for x in ce])), 3)
            o[side] = e
        return o
    decks = sorted({x["leader"] for x in rows})
    out = {"record": a.src, "by_deck": {d: {"all": blk([x for x in rows if x["leader"] == d]),
                                            **{b: blk([x for x in rows if x["leader"] == d and turn_band(x["t"]) == b]) for b in TURN_BANDS}}
                                        for d in decks},
           "ALL": blk(rows)}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("clockside: rows %d" % len(rows))
    return 0


def cmd_guardside(a):
    """守りの時計の外れをもう 1 段割る（2026-10-12・`2026-10-12_don_deck.md` §10）。
    (1) 理論の守りの窓の読み（`theory_bridge` の写しの `guard`＝`guard_dump`）: デッキ × 帯ごとに「止められると読んだ」割合・
        そのうち実際に受けた（止めなかった／止められなかった）割合・払えるカウンター札の枚数／手札の枚数・実際に消えた札の価値。
    (2) 記録だけの量: 守りの窓での手札の中身（カウンター 0／カウンターイベント／カウンター値あり・カードの印字）、
        ライフに加えた枚数・自分のターン中にライフが離れた回数・相手のリーダーへの攻撃 1 回あたりに失ったライフ（1 局×席あたり）。"""
    import deck_refill as DR
    import theory_rs as TR
    from opcg_sim.learned.train import plan_labels as PL
    from opcg_sim.loop import decks as D
    meta = json.load(open(os.path.join(a.src, "meta_games.json"), encoding="utf-8"))
    games_meta = {int(g["seed"]): dict(g, leaders=seat_leaders(meta, g)) for g in meta["games"]}
    out = {"record": a.src}
    # (1) 理論の守りの窓
    win = []
    with open(a.dump, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            gm = games_meta.get(int(j["seed"]))
            if gm is None:
                continue
            for g in j.get("guard") or []:
                t, w, played, can, says, src_, x, ctake, chand, cgs, nh, nc, spent, life, margin = g
                win.append({"seed": int(j["seed"]), "t": int(t), "leader": gm["leaders"][int(w)], "played": played,
                            "can": bool(can), "says": says, "src": src_, "nh": nh or 0, "nc": nc or 0, "spent": spent})

    def gblk(ws):
        if not ws:
            return None
        can = [x for x in ws if x["can"]]
        says_g = [x for x in ws if x["says"] == "guard"]
        sp = [x["spent"] for x in ws if x["spent"] is not None]
        return {"windows": len(ws), "games": len({x["seed"] for x in ws}),
                "can_share": round(len(can) / len(ws), 3),
                "can_but_took": round(sum(1 for x in can if x["played"] == "take") / len(can), 3) if can else None,
                "says_guard_share": round(len(says_g) / len(ws), 3),
                "says_guard_but_took": round(sum(1 for x in says_g if x["played"] == "take") / len(says_g), 3) if says_g else None,
                "took_share": round(sum(1 for x in ws if x["played"] == "take") / len(ws), 3),
                "src": dict(collections.Counter(x["src"] for x in ws)),
                "payable_counter_per_hand": round(float(np.mean([x["nc"] / x["nh"] for x in ws if x["nh"]])), 3) if any(x["nh"] for x in ws) else None,
                "payable_counter_cards": round(float(np.mean([x["nc"] for x in ws])), 3),
                "spent_mean": round(float(np.mean(sp)), 4) if sp else None}
    decks = sorted({x["leader"] for x in win})
    out["guard"] = {d: {"all": gblk([x for x in win if x["leader"] == d]),
                        **{b: gblk([x for x in win if x["leader"] == d and turn_band(x["t"]) == b]) for b in TURN_BANDS}} for d in decks}
    out["guard"]["ALL"] = {"all": gblk(win)}
    # (2) 記録だけ
    idx2cid = TR.idx2cid()
    db = D.load_db()
    info = {}

    def cinfo(cid):
        if cid not in info:
            m = db.get_card(cid)
            if m is None:
                info[cid] = None
            else:
                ty = getattr(getattr(m, "type", None), "name", "")
                txt = getattr(m, "effect_text", "") or ""
                info[cid] = {"counter": int(getattr(m, "counter", 0) or 0), "cev": ty == "EVENT" and "【カウンター】" in txt}
        return info[cid]
    cols = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen")
    hand = collections.defaultdict(collections.Counter)
    life_add, life_left_own, seats, attacks, life_lost = (collections.Counter() for _ in range(5))
    for rows, pol, ex, L, ptr, idx in PL.iter_games([a.src], row_cols=cols, pol_cols=("pol_sig", "pol_cid", "pol_tcid"),
                                                    extra_fn=lambda dd, n: {"sc": np.asarray(dd["scalars"])[:n].astype(float),
                                                                            "ci": np.asarray(dd["card_idx"])[:n]}):
        seed = int(rows["seed"][idx[0]])
        gm = games_meta.get(seed)
        if gm is None:
            continue
        lead = gm["leaders"]
        u2c = PL.uuid_map(pol, L, ptr, idx)
        for w in (0, 1):
            seats[lead[w]] += 1
        seen, prev_life, prev_turn, start_life = set(), {0: None, 1: None}, {0: None, 1: None}, {}
        for i in idx:
            w = int(rows["who"][i]); t = int(rows["turn"][i]); sc = ex["sc"][i]
            if t < 1:
                continue
            own = PL.is_own_turn(w, t)
            if not own and (w, t) not in seen:
                seen.add((w, t))
                for cid in DR.hand_ids(ex["ci"][i], idx2cid):
                    inf = cinfo(cid)
                    if inf is None:
                        continue
                    k = "cev" if inf["cev"] else ("c0" if inf["counter"] == 0 else "cpos")
                    hand[(lead[w], turn_band(t))][k] += 1
                    hand[(lead[w], "all")][k] += 1
            start_life.setdefault((w, t), sc[0])
            if prev_life[w] is not None:
                dl = sc[0] - prev_life[w]
                if dl > 0.5:
                    life_add[lead[w]] += dl
                if dl < -0.5 and own and prev_turn[w] == t:
                    life_left_own[lead[w]] += -dl
            prev_life[w], prev_turn[w] = sc[0], t
            if own and int(rows["kind"][i]) == 0:
                sj = json.loads(rows["sig"][i])
                if sj[0] in ("ATTACK", "DON_BOX") and sj[2] and u2c.get(sj[2][0]) == lead[1 - w]:
                    attacks[lead[1 - w]] += 1
        for (w, t), l0 in start_life.items():
            if not PL.is_own_turn(w, t):
                nxt = start_life.get((w, t + 1))
                if nxt is not None and nxt < l0:
                    life_lost[lead[w]] += l0 - nxt
    out["hand"] = {"%s|%s" % k: {"cards": sum(c.values()), **{x: round(c[x] / sum(c.values()), 3) for x in ("c0", "cev", "cpos")}}
                   for k, c in sorted(hand.items())}
    out["life"] = {d: {"seats": seats[d], "life_added_per_seat": round(life_add[d] / seats[d], 3),
                       "life_left_own_turn_per_seat": round(life_left_own[d] / seats[d], 3),
                       "leader_attacks_received": attacks[d], "life_lost_per_attack": round(life_lost[d] / max(1, attacks[d]), 3)}
                   for d in seats}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("guardside: windows %d" % len(win))
    return 0


def cmd_defturns(a):
    """守る側の相手のターン 1 回ごとの量（2026-10-13・`2026-10-13_defence_gap.md` §8・代案の材料・式には入れない）。
    記録だけで、デッキ × 帯ごとに: 相手のターンに来た攻撃の本数（全部の対象／リーダーへ）の分布（0／1／2／3 本以上）と、
    そのターンの最初の守る側の行で手札に「カウンター 0 の札」（カウンターイベントを除く）が 1 枚以上在る割合。"""
    import deck_refill as DR
    import theory_rs as TR
    from opcg_sim.learned.train import plan_labels as PL
    from opcg_sim.loop import decks as D
    meta = json.load(open(os.path.join(a.src, "meta_games.json"), encoding="utf-8"))
    games_meta = {int(g["seed"]): dict(g, leaders=seat_leaders(meta, g)) for g in meta["games"]}
    idx2cid = TR.idx2cid()
    db = D.load_db()
    info = {}

    def cinfo(cid):
        if cid not in info:
            m = db.get_card(cid)
            if m is None:
                info[cid] = None
            else:
                ty = getattr(getattr(m, "type", None), "name", "")
                txt = getattr(m, "effect_text", "") or ""
                info[cid] = {"counter": int(getattr(m, "counter", 0) or 0), "cev": ty == "EVENT" and "【カウンター】" in txt}
        return info[cid]
    cols = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen")
    turns = []                                    # (守る側のリーダー, 帯, 攻撃の本数, リーダーへの本数, 手札にカウンター 0 が在るか)
    for rows, pol, ex, L, ptr, idx in PL.iter_games([a.src], row_cols=cols, pol_cols=("pol_sig", "pol_cid", "pol_tcid"),
                                                    extra_fn=lambda dd, n: {"ci": np.asarray(dd["card_idx"])[:n]}):
        seed = int(rows["seed"][idx[0]])
        gm = games_meta.get(seed)
        if gm is None:
            continue
        lead = gm["leaders"]
        u2c = PL.uuid_map(pol, L, ptr, idx)
        att, att_l, c0, seen = collections.Counter(), collections.Counter(), {}, set()
        for i in idx:
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1:
                continue
            if PL.is_own_turn(w, t):
                seen.add((1 - w, t))
                if int(rows["kind"][i]) == 0:
                    sj = json.loads(rows["sig"][i])
                    if sj[0] in ("ATTACK", "DON_BOX") and sj[2]:
                        att[(1 - w, t)] += 1
                        att_l[(1 - w, t)] += int(u2c.get(sj[2][0]) == lead[1 - w])
            elif (w, t) not in c0:
                has = False
                for cid in DR.hand_ids(ex["ci"][i], idx2cid):
                    inf = cinfo(cid)
                    if inf is not None and not inf["cev"] and inf["counter"] == 0:
                        has = True
                c0[(w, t)] = has
        for (d, t) in sorted(seen):
            turns.append((lead[d], turn_band(t), att[(d, t)], att_l[(d, t)], c0.get((d, t))))

    def blk(ts):
        if not ts:
            return None
        def dist(k):
            c = collections.Counter(min(3, x[k]) for x in ts)
            return {("3+" if n == 3 else str(n)): round(c[n] / len(ts), 3) for n in range(4)}
        known = [x for x in ts if x[4] is not None]
        return {"turns": len(ts), "attacks": dist(2), "leader_attacks": dist(3),
                "attacks_mean": round(float(np.mean([x[2] for x in ts])), 3),
                "leader_attacks_mean": round(float(np.mean([x[3] for x in ts])), 3),
                "hand_has_c0": round(sum(1 for x in known if x[4]) / len(known), 3) if known else None}
    decks = sorted({x[0] for x in turns})
    out = {"record": a.src, "by_deck": {d: {"all": blk([x for x in turns if x[0] == d]),
                                            **{b: blk([x for x in turns if x[0] == d and x[1] == b]) for b in TURN_BANDS}} for d in decks}}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("defturns: turns %d" % len(turns))
    return 0


def cmd_parts(a):
    """耐久を部品ごとに割る（2026-10-13・`2026-10-13_enel_parts.md` §1・1 回きり）。攻める席のターン先頭の行（損害の行・決着の宣言の前）ごとに、
    守る席の耐久の 3 項（ライフ＝Θ − 手札 − 体・手札・体）と、その行から終局までの実際（ライフの減り × λ・守りの窓で消えた札の価値・
    ブロックの回数 × ブロッカー 1 体の読み）を並べ、守る席のデッキ × 帯ごとに「実際の和 ÷ 理論の和」を、倒れた局／倒れなかった局（打ち切り）で出す。"""
    from opcg_sim.learned.train import plan_labels as PL
    lam = 0.1362                                  # `leaves_to::LAM`
    meta = json.load(open(os.path.join(a.src, "meta_games.json"), encoding="utf-8"))
    games_meta = {int(g["seed"]): dict(g, leaders=seat_leaders(meta, g)) for g in meta["games"]}
    ck = json.load(open(a.clocks, encoding="utf-8"))
    g2s = {int(r["g"]): int(r["seed"]) for r in ck["rows_out"]}
    tseq, settled = collections.defaultdict(list), set()
    for sd, w, t, done in ck["settled"]:
        tseq[(int(sd), int(w))].append(int(t))
        if done:
            settled.add((int(sd), int(w), int(t)))
    for k in tseq:
        tseq[k].sort()
    # 守りの窓で消えた札の価値（同じ相手のターンでは最大＝最初の窓を 1 回）
    spent = collections.defaultdict(float)
    with open(a.dump, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            for g in j.get("guard") or []:
                t, w, sp = int(g[0]), int(g[1]), g[12]
                if sp is not None:
                    k = (int(j["seed"]), w, t)
                    spent[k] = max(spent[k], float(sp))
    # 記録: 席ごとのライフの減り（ターンごと）・ブロックの回数（ターンごと）・勝敗
    cols = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen")
    lost, blocks, zof = collections.defaultdict(float), collections.Counter(), {}
    for rows, pol, ex, L, ptr, idx in PL.iter_games([a.src], row_cols=cols, pol_cols=("pol_sig",),
                                                    extra_fn=lambda dd, n: {"sc": np.asarray(dd["scalars"])[:n].astype(float)}):
        seed = int(rows["seed"][idx[0]])
        prev = {0: None, 1: None}
        for i in idx:
            w = int(rows["who"][i]); t = int(rows["turn"][i]); sc = ex["sc"][i]
            zof[(seed, w)] = float(rows["z"][i])
            for d, life in ((w, sc[0]), (1 - w, sc[1])):
                if prev[d] is not None and life < prev[d] - 0.5:
                    lost[(seed, d, t)] += prev[d] - life
                prev[d] = life
            if json.loads(rows["sig"][i])[0] == "SELECT_BLOCKER":
                blocks[(seed, w, t)] += 1
    def by_seat(dct):
        out = collections.defaultdict(dict)
        for (sd, dd, tt), v in dct.items():
            out[(sd, dd)][tt] = out[(sd, dd)].get(tt, 0.0) + v
        return out
    lost_s, spent_s, blocks_s = by_seat(lost), by_seat(spent), by_seat(blocks)

    def after(m, key, t):
        return sum(v for tt, v in m.get(key, {}).items() if tt >= t)
    recs = []
    for r in ck["turn_harm"]:
        seed = g2s.get(int(r["g"]))
        if seed is None or seed not in games_meta:
            continue
        w = int(r["who"]); jj = int(r["j"]); ts = tseq.get((seed, w), [])
        if jj >= len(ts):
            continue
        t = ts[jj]
        if (seed, w, t) in settled:
            continue
        d = 1 - w
        th, hh, bb = float(r["theta"]), float(r["th_hand"]), float(r["th_body"])
        nb = int(r.get("d_blk_n") or 0)
        per_b = bb / nb if nb > 0 else bb
        a_life = lam * after(lost_s, (seed, d), t)
        a_hand = after(spent_s, (seed, d), t)
        a_body = per_b * after(blocks_s, (seed, d), t)
        fell = (zof.get((seed, d)) or 0.0) < 0.5
        recs.append({"leader": games_meta[seed]["leaders"][d], "band": turn_band(t), "fell": fell,
                     "th": {"life": th - hh - bb, "hand": hh, "body": bb}, "ac": {"life": a_life, "hand": a_hand, "body": a_body}})

    def blk(rs):
        if not rs:
            return None
        out = {"rows": len(rs)}
        for part in ("life", "hand", "body"):
            st = sum(x["th"][part] for x in rs); sa = sum(x["ac"][part] for x in rs)
            out[part] = {"theory": round(st / len(rs), 4), "actual": round(sa / len(rs), 4), "ratio": round(sa / st, 3) if st > 1e-12 else None}
        return out
    decks = sorted({x["leader"] for x in recs})
    res = {}
    for dk in decks + ["ALL"]:
        rd = [x for x in recs if dk == "ALL" or x["leader"] == dk]
        res[dk] = {}
        for band in ("all",) + TURN_BANDS:
            rb = [x for x in rd if band == "all" or x["band"] == band]
            res[dk][band] = {"both": blk(rb), "fell": blk([x for x in rb if x["fell"]]), "survived": blk([x for x in rb if not x["fell"]])}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"record": a.src, "clocks": a.clocks, "dump": a.dump, "lam": lam, "parts": res}, f, ensure_ascii=False, indent=1)
    print("parts: rows %d" % len(recs))
    return 0


def cmd_reread(a):
    """対局の読み直し（2026-10-14・`2026-10-14_reread.md` §1・1 回きり・記録だけ）。
    (E) 出した体が次の相手のターンのうちに場から居なくなった割合を、持ち主のデッキ × 印字のパワー帯 × 出したターンの帯で（場の札の多重集合の差で数える）。
        理論の読み（`ko_p_of` の山形・定数 0.289）と並べる。
    (N) 守る席のデッキ別に、相手のターンに自分のライフが減った直後に自分が効果の選択をした回数（【トリガー】の発動の代わりの数）と、
        ライフが減った回数（1 局あたり・帯別）。デッキの【トリガー】の札の効果の種類（効果 JSON の TRIGGER 能力の動作）を並べる。"""
    import theory_rs as TR
    from opcg_sim.learned.train import plan_labels as PL
    from opcg_sim.loop import decks as D
    meta = json.load(open(os.path.join(a.src, "meta_games.json"), encoding="utf-8"))
    games_meta = {int(g["seed"]): dict(g, leaders=seat_leaders(meta, g)) for g in meta["games"]}
    idx2cid = TR.idx2cid()
    db = D.load_db()
    eff = json.load(open(os.path.join("opcg_sim", "data", "opcg_effects.json"), encoding="utf-8"))["cards"]
    pinfo = {}

    def char_power(cid):
        if cid not in pinfo:
            m = db.get_card(cid)
            ty = getattr(getattr(m, "type", None), "name", "") if m is not None else ""
            pinfo[cid] = float(getattr(m, "power", 0) or 0) if ty == "CHARACTER" else None
        return pinfo[cid]

    def pband(p):
        return "<=2000" if p <= 2000 else "3000-4000" if p <= 4000 else "5000-6000" if p <= 6000 else "7000+"

    def ko_curve(p):                               # `leaves_to::ko_p_of`
        for hi, v in ((2000.0, 0.2417), (4000.0, 0.2782), (6000.0, 0.3297), (8000.0, 0.2436)):
            if p < hi:
                return v
        return 0.1987
    cols = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen")
    body = collections.defaultdict(lambda: [0, 0, 0.0])      # (deck, pband, band) → [出した, 居なくなった, 理論の和]
    trig = collections.defaultdict(lambda: collections.Counter())
    seats = collections.Counter()
    for rows, pol, ex, L, ptr, idx in PL.iter_games([a.src], row_cols=cols, pol_cols=("pol_sig",),
                                                    extra_fn=lambda dd, n: {"sc": np.asarray(dd["scalars"])[:n].astype(float),
                                                                            "ci": np.asarray(dd["card_idx"])[:n]}):
        seed = int(rows["seed"][idx[0]])
        gm = games_meta.get(seed)
        if gm is None:
            continue
        lead = gm["leaders"]
        for w in (0, 1):
            seats[lead[w]] += 1
        first, last = {}, {}
        prev = {0: None, 1: None}
        for i in idx:
            w = int(rows["who"][i]); t = int(rows["turn"][i]); sc = ex["sc"][i]; ci = ex["ci"][i]
            if t < 1:
                continue
            for d in (0, 1):
                sl = range(2, 7) if d == w else range(7, 12)
                f = collections.Counter(c for c in (idx2cid.get(int(ci[s])) for s in sl) if c)
                first.setdefault((d, t), f)
                last[(d, t)] = f
            life = {w: sc[0], 1 - w: sc[1]}
            sj = json.loads(rows["sig"][i])
            own = PL.is_own_turn(w, t)
            if not own and prev[w] is not None and sj[0] == "RESOLVE_EFFECT_SELECTION" and life[w] < prev[w] - 0.5:
                trig[(lead[w], turn_band(t))]["sel_after_loss"] += 1
            for d in (0, 1):
                if prev[d] is not None and life[d] < prev[d] - 0.5 and not PL.is_own_turn(d, t):
                    trig[(lead[d], turn_band(t))]["life_lost"] += prev[d] - life[d]
                prev[d] = life[d]
        for (d, t), fe in last.items():
            if not PL.is_own_turn(d, t) or (d, t + 2) not in first:
                continue
            fs = first.get((d, t), collections.Counter())
            fn = first[(d, t + 2)]
            for cid, ke in fe.items():
                p = char_power(cid)
                if p is None:
                    continue
                played = max(0, ke - fs.get(cid, 0))
                if played <= 0:
                    continue
                old = ke - played
                surv = max(0, min(played, fn.get(cid, 0) - old))
                b = body[(lead[d], pband(p), turn_band(t))]
                b[0] += played; b[1] += played - surv; b[2] += played * ko_curve(p)
    out = {"record": a.src, "body": {}, "trigger": {}, "trigger_cards": {}}
    for dk in sorted({k[0] for k in body}):
        out["body"][dk] = {}
        for pb in ("<=2000", "3000-4000", "5000-6000", "7000+"):
            for band in ("all",) + TURN_BANDS:
                ks = [k for k in body if k[0] == dk and k[1] == pb and (band == "all" or k[2] == band)]
                n = sum(body[k][0] for k in ks)
                if n:
                    out["body"][dk]["%s|%s" % (pb, band)] = {"played": n, "gone_next_opp_turn": round(sum(body[k][1] for k in ks) / n, 3),
                                                             "theory_ko_p_of": round(sum(body[k][2] for k in ks) / n, 3), "theory_const": 0.289}
    for dk in sorted(seats):
        out["trigger"][dk] = {}
        for band in ("all",) + TURN_BANDS:
            ks = [k for k in trig if k[0] == dk and (band == "all" or k[1] == band)]
            sel = sum(trig[k]["sel_after_loss"] for k in ks); lost = sum(trig[k]["life_lost"] for k in ks)
            out["trigger"][dk][band] = {"sel_after_loss_per_game": round(sel / seats[dk], 3), "life_lost_per_game": round(lost / seats[dk], 3),
                                        "sel_per_life_lost": round(sel / lost, 3) if lost else None}
        deck = None
        for g in meta["games"]:
            up = D.user_deck_pair(int(g["seed"])) if meta.get("decks") == "user" else None
            if up:
                for l, dd in up[:2]:
                    if l == dk:
                        deck = list(dd)
            if deck:
                break
        kinds, n_trig = collections.Counter(), 0
        for cid in deck or []:
            for ab in (eff.get(cid) or {}).get("abilities", []):
                if ab.get("trigger") != "TRIGGER":
                    continue
                n_trig += 1
                acts = []

                def walk(x):
                    if isinstance(x, dict):
                        if x.get("node") == "GameAction" and x.get("type"):
                            acts.append(x["type"])
                        for v in x.values():
                            walk(v)
                    elif isinstance(x, list):
                        for v in x:
                            walk(v)
                walk(ab.get("effect"))
                for t_ in set(acts):
                    kinds[t_] += 1
        out["trigger_cards"][dk] = {"deck_cards": len(deck or []), "trigger_cards": n_trig, "action_types": dict(kinds.most_common())}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("reread: bodies %d" % sum(v[0] for v in body.values()))
    return 0


def cmd_speed(a):
    """相手を削る速さの理論と実際（2026-10-14・`2026-10-14_attack_speed.md` §1・1 回きり）。攻める席のターン先頭の行（損害の行・決着の宣言の前）ごとに、
    理論の見込み（`slope_theory`）と実際の耐久の減り（`harm`）を、攻める側のデッキ × 帯で比べる。記録から、そのターンの相手のライフの減り・
    攻撃の数・相手のリーダーへの攻撃の数・相手のブロックの数（1 ターンあたり）も出す。"""
    from opcg_sim.learned.train import plan_labels as PL
    meta = json.load(open(os.path.join(a.src, "meta_games.json"), encoding="utf-8"))
    games_meta = {int(g["seed"]): dict(g, leaders=seat_leaders(meta, g)) for g in meta["games"]}
    ck = json.load(open(a.clocks, encoding="utf-8"))
    g2s = {int(r["g"]): int(r["seed"]) for r in ck["rows_out"]}
    tseq, settled = collections.defaultdict(list), set()
    for sd, w, t, done in ck["settled"]:
        tseq[(int(sd), int(w))].append(int(t))
        if done:
            settled.add((int(sd), int(w), int(t)))
    for k in tseq:
        tseq[k].sort()
    cols = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen")
    rec = collections.defaultdict(collections.Counter)          # (seed, w, t) → 量
    for rows, pol, ex, L, ptr, idx in PL.iter_games([a.src], row_cols=cols, pol_cols=("pol_sig", "pol_cid", "pol_tcid"),
                                                    extra_fn=lambda dd, n: {"sc": np.asarray(dd["scalars"])[:n].astype(float)}):
        seed = int(rows["seed"][idx[0]])
        gm = games_meta.get(seed)
        if gm is None:
            continue
        lead = gm["leaders"]
        u2c = PL.uuid_map(pol, L, ptr, idx)
        prev = {0: None, 1: None}
        for i in idx:
            w = int(rows["who"][i]); t = int(rows["turn"][i]); sc = ex["sc"][i]
            if t < 1:
                continue
            life = {w: sc[0], 1 - w: sc[1]}
            att = w if PL.is_own_turn(w, t) else 1 - w           # このターンの攻める席
            sj = json.loads(rows["sig"][i])
            lost_now = prev[1 - att] is not None and life[1 - att] < prev[1 - att] - 0.5
            if lost_now:
                rec[(seed, att, t)]["opp_life_lost"] += prev[1 - att] - life[1 - att]
            if prev[1 - att] is not None and life[1 - att] > prev[1 - att] + 0.5:
                rec[(seed, att, t)]["def_life_gained"] += life[1 - att] - prev[1 - att]
            if w != att and sj[0] == "SELECT_COUNTER":
                rec[(seed, att, t)]["counters"] += 1
            if w != att and lost_now and sj[0] == "RESOLVE_EFFECT_SELECTION":
                rec[(seed, att, t)]["trig_proxy"] += 1
            for d in (0, 1):
                prev[d] = life[d]
            if w == att and int(rows["kind"][i]) == 0 and sj[0] in ("ATTACK", "DON_BOX") and sj[2]:
                rec[(seed, att, t)]["attacks"] += 1
                rec[(seed, att, t)]["leader_attacks"] += int(u2c.get(sj[2][0]) == lead[1 - att])
            if w != att and sj[0] == "SELECT_BLOCKER":
                rec[(seed, att, t)]["blocks"] += 1
    recs = []
    for r in ck["turn_harm"]:
        seed = g2s.get(int(r["g"]))
        if seed is None or seed not in games_meta:
            continue
        w = int(r["who"]); jj = int(r["j"]); ts = tseq.get((seed, w), [])
        if jj >= len(ts):
            continue
        t = ts[jj]
        if (seed, w, t) in settled:
            continue
        q = rec.get((seed, w, t), collections.Counter())
        recs.append({"deck": games_meta[seed]["leaders"][w], "def": games_meta[seed]["leaders"][1 - w], "band": turn_band(t), "theory": float(r.get("slope_theory") or 0.0),
                     "harm": float(r.get("harm") or 0.0), **{k: float(q.get(k, 0)) for k in ("opp_life_lost", "attacks", "leader_attacks", "blocks", "counters", "def_life_gained", "trig_proxy")}})

    def blk(rs):
        if not rs:
            return None
        n = len(rs)
        st = sum(x["theory"] for x in rs); sh = sum(x["harm"] for x in rs)
        la = sum(x["leader_attacks"] for x in rs); ll = sum(x["opp_life_lost"] for x in rs)
        return {"turns": n, "theory_per_turn": round(st / n, 4), "harm_per_turn": round(sh / n, 4), "ratio": round(sh / st, 3) if st > 1e-12 else None,
                "opp_life_lost_per_turn": round(ll / n, 3), "attacks_per_turn": round(sum(x["attacks"] for x in rs) / n, 3),
                "leader_attacks_per_turn": round(la / n, 3), "life_per_leader_attack": round(ll / la, 3) if la else None,
                "blocks_per_turn": round(sum(x["blocks"] for x in rs) / n, 3),
                "counters_per_turn": round(sum(x["counters"] for x in rs) / n, 3),
                "def_life_gained_per_turn": round(sum(x["def_life_gained"] for x in rs) / n, 3),
                "trig_proxy_per_turn": round(sum(x["trig_proxy"] for x in rs) / n, 3)}
    decks = sorted({x["deck"] for x in recs})
    out = {"record": a.src, "clocks": a.clocks, "by_deck": {d: {"all": blk([x for x in recs if x["deck"] == d]),
                                                               **{b: blk([x for x in recs if x["deck"] == d and x["band"] == b]) for b in TURN_BANDS}}
                                                           for d in decks}}
    # 守る側（行の相手）で集める・守る側 × 攻める側の比（2026-10-14・`2026-10-14_defence_side.md`）
    out["by_defender"] = {d: {"all": blk([x for x in recs if x["def"] == d]),
                              **{b: blk([x for x in recs if x["def"] == d and x["band"] == b]) for b in TURN_BANDS}} for d in decks}
    out["def_x_att"] = {d: {o: {b: (lambda r: r and r["ratio"])(blk([x for x in recs if x["def"] == d and x["deck"] == o and (b == "all" or x["band"] == b)]))
                                for b in ("all",) + TURN_BANDS} for o in decks} for d in decks}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("speed: rows %d" % len(recs))
    return 0


def cmd_namipower(a):
    """ナミのリーダーの +2000 の取りこぼし（2026-10-14・`2026-10-14_nami_power.md` §1・1 回きり・記録だけ）。ナミが守る相手のターンごとに:
    相手のターンの先頭の行（攻める側の最初の行＝理論の判定と同じ盤面）でのナミのリーダーの付与中のドン・手札の枚数・リーダーのパワー、
    相手の各攻撃の時点の付与中のドン、ナミの行（守りの窓）でリーダーのパワーが印字 + 2000 以上だったか、ナミが効果を選んだ数。
    理論の読み＝先頭の行で（ドン ≥ 1・手札 ≥ 1・パワー < 印字 + 2000）なら 7000、それ以外は 5000（`drv_cb::opp_attack_power_adjust`）。"""
    from opcg_sim.learned.train import plan_labels as PL
    NAMI, PRINTED, X = "OP11-041", 5000.0, 2000.0
    meta = json.load(open(os.path.join(a.src, "meta_games.json"), encoding="utf-8"))
    games_meta = {int(g["seed"]): dict(g, leaders=seat_leaders(meta, g)) for g in meta["games"]}
    cols = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen")
    turns = []
    import theory_rs as TR
    idx2cid = TR.idx2cid()
    apow = collections.defaultdict(list)                      # 帯 → 攻撃のパワー（攻める側の札の欄・その行）
    for rows, pol, ex, L, ptr, idx in PL.iter_games([a.src], row_cols=cols, pol_cols=("pol_sig", "pol_cid", "pol_tcid"),
                                                    extra_fn=lambda dd, n: {"sc": np.asarray(dd["scalars"])[:n].astype(float),
                                                                            "ci": np.asarray(dd["card_idx"])[:n],
                                                                            "tok": np.asarray(dd["tokens"])[:n].astype(float)}):
        seed = int(rows["seed"][idx[0]])
        gm = games_meta.get(seed)
        if gm is None or NAMI not in gm["leaders"]:
            continue
        u2c = PL.uuid_map(pol, L, ptr, idx)
        for d in (0, 1):
            if gm["leaders"][d] != NAMI:
                continue
            cur = {}
            for i in idx:
                w = int(rows["who"][i]); t = int(rows["turn"][i]); sc = ex["sc"][i]
                if t < 1 or PL.is_own_turn(d, t):
                    continue
                me = (w == d)
                don = (sc[14] if me else sc[15]) * 5.0
                hand = sc[6] if me else sc[7]
                pw = (sc[12] if me else sc[13]) * 1e4
                c = cur.setdefault(t, {"t": t, "start": None, "atk": 0, "atk_don": 0, "win": 0, "win_buff": 0, "sel": 0, "buff_after_atk": 0, "_pend": False})
                if not me and c["start"] is None:
                    c["start"] = {"don": don, "hand": hand, "pw": pw}
                sj = json.loads(rows["sig"][i])
                if not me and int(rows["kind"][i]) == 0 and sj[0] in ("ATTACK", "DON_BOX") and sj[2]:
                    c["atk"] += 1
                    c["atk_don"] += int(don >= 0.5)
                    c["_pend"] = True
                    acid = u2c.get(sj[1])
                    for sl in [0] + list(range(2, 7)):
                        if acid and idx2cid.get(int(ex["ci"][i][sl])) == acid:
                            apow[turn_band(t)].append(float(ex["tok"][i][sl][0]) * 1e4)
                            break
                if me:
                    c["win"] += 1
                    c["win_buff"] += int(pw >= PRINTED + X - 1)
                    if c["_pend"]:
                        c["buff_after_atk"] += int(pw >= PRINTED + X - 1)
                        c["_pend"] = False
                    if sj[0] == "RESOLVE_EFFECT_SELECTION":
                        c["sel"] += 1
            for t, c in cur.items():
                if c["start"] is None:
                    continue
                st = c["start"]
                theory7000 = st["don"] >= 0.5 and st["hand"] >= 1 and st["pw"] < PRINTED + X - 1
                turns.append({"band": turn_band(t), "don_start": st["don"] >= 0.5, "theory7000": theory7000,
                              "already7000_start": st["pw"] >= PRINTED + X - 1, "atk": c["atk"], "atk_don": c["atk_don"],
                              "win": c["win"], "win_buff": c["win_buff"], "buff_seen": c["win_buff"] > 0,
                              "buff_after_atk": c["buff_after_atk"], "sel": c["sel"]})

    def blk(ts):
        if not ts:
            return None
        n = len(ts)
        atk = sum(x["atk"] for x in ts)
        win = sum(x["win"] for x in ts)
        miss = [x for x in ts if (x["don_start"] or x["atk_don"] > 0) and x["buff_seen"] and not x["theory7000"]]
        return {"opp_turns": n, "don_on_leader_at_turn_start": round(sum(x["don_start"] for x in ts) / n, 3),
                "don_on_leader_at_attack": round(sum(x["atk_don"] for x in ts) / atk, 3) if atk else None,
                "theory_reads_7000": round(sum(x["theory7000"] for x in ts) / n, 3),
                "already_7000_at_turn_start": round(sum(x["already7000_start"] for x in ts) / n, 3),
                "turns_with_buff_seen": round(sum(x["buff_seen"] for x in ts) / n, 3),
                "windows_with_buff": round(sum(x["win_buff"] for x in ts) / win, 3) if win else None,
                "attacks_followed_by_buffed_window": round(sum(x["buff_after_atk"] for x in ts) / atk, 3) if atk else None,
                "effect_selections_per_turn": round(sum(x["sel"] for x in ts) / n, 3),
                "don_and_buff_but_theory_5000": round(len(miss) / n, 3),
                "buff_seen_but_theory_5000": round(sum(1 for x in ts if x["buff_seen"] and not x["theory7000"]) / n, 3),
                "attacks_per_turn": round(atk / n, 3)}
    out = {"record": a.src, "nami": {"all": blk(turns), **{b: blk([x for x in turns if x["band"] == b]) for b in TURN_BANDS}}}

    def pdist(ps):
        if not ps:
            return None
        n = len(ps)
        return {"attacks_with_power": n, "lt5000": round(sum(p < 5000 - 1 for p in ps) / n, 3),
                "5000_6999": round(sum(5000 - 1 <= p < 7000 - 1 for p in ps) / n, 3), "ge7000": round(sum(p >= 7000 - 1 for p in ps) / n, 3),
                "mean": round(float(np.mean(ps)), 1)}
    out["attack_power_into_nami"] = {"all": pdist(sum(apow.values(), [])), **{b: pdist(apow.get(b, [])) for b in TURN_BANDS}}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("namipower: opp turns %d" % len(turns))
    return 0


def cmd_reach(a):
    """リーダーに届く攻撃の数（2026-10-14・`2026-10-14_reach.md` §1・1 回きり）。攻める席のターンの先頭の行（損害の行・決着の宣言の前）ごとに、
    理論の計画が 1 段目で届くとみなした攻撃の数（`reach_theory`・`OPCG_REACH_DUMP=1` の時計）と、記録の実際（相手のリーダーへの攻撃ごとに、
    宣言の次の行で攻め手のパワー ≥ 守る側のリーダーのパワー なら届いた）・ドンを付けずに攻撃した割合を、守る側のデッキ × 帯で並べる。"""
    import theory_rs as TR
    from opcg_sim.learned.train import plan_labels as PL
    idx2cid = TR.idx2cid()
    meta = json.load(open(os.path.join(a.src, "meta_games.json"), encoding="utf-8"))
    games_meta = {int(g["seed"]): dict(g, leaders=seat_leaders(meta, g)) for g in meta["games"]}
    ck = json.load(open(a.clocks, encoding="utf-8"))
    g2s = {int(r["g"]): int(r["seed"]) for r in ck["rows_out"]}
    tseq, settled = collections.defaultdict(list), set()
    for sd, w, t, done in ck["settled"]:
        tseq[(int(sd), int(w))].append(int(t))
        if done:
            settled.add((int(sd), int(w), int(t)))
    for k in tseq:
        tseq[k].sort()
    cols = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen")
    rec = collections.defaultdict(collections.Counter)
    for rows, pol, ex, L, ptr, idx in PL.iter_games([a.src], row_cols=cols, pol_cols=("pol_sig", "pol_cid", "pol_tcid"),
                                                    extra_fn=lambda dd, n: {"sc": np.asarray(dd["scalars"])[:n].astype(float),
                                                                            "ci": np.asarray(dd["card_idx"])[:n],
                                                                            "tok": np.asarray(dd["tokens"])[:n].astype(float)}):
        seed = int(rows["seed"][idx[0]])
        gm = games_meta.get(seed)
        if gm is None:
            continue
        lead = gm["leaders"]
        u2c = PL.uuid_map(pol, L, ptr, idx)
        idl = list(idx)
        for k, i in enumerate(idl[:-1]):
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t) or int(rows["kind"][i]) != 0:
                continue
            sj = json.loads(rows["sig"][i])
            if sj[0] not in ("ATTACK", "DON_BOX") or not sj[2] or u2c.get(sj[2][0]) != lead[1 - w]:
                continue
            acid = u2c.get(sj[1])
            j = idl[k + 1]
            wj = int(rows["who"][j]); ci = ex["ci"][j]; tok = ex["tok"][j]; sc = ex["sc"][j]
            slots = ([0] + list(range(2, 7))) if wj == w else ([1] + list(range(7, 12)))
            dpow = (sc[13] if wj == w else sc[12]) * 1e4
            sl = next((q for q in slots if acid and idx2cid.get(int(ci[q])) == acid), None)
            r = rec[(seed, w, t)]
            r["leader_attacks"] += 1
            if sl is None:
                r["unresolved"] += 1
                continue
            p = float(tok[sl][0]) * 1e4
            r["reached"] += int(p >= dpow - 1)
            r["no_don"] += int(float(tok[sl][2]) * 5 < 0.5)
            r["resolved"] += 1
    recs = []
    for r in ck["turn_harm"]:
        seed = g2s.get(int(r["g"]))
        if seed is None or seed not in games_meta or "reach_theory" not in r:
            continue
        w = int(r["who"]); jj = int(r["j"]); ts = tseq.get((seed, w), [])
        if jj >= len(ts):
            continue
        t = ts[jj]
        if (seed, w, t) in settled:
            continue
        q = rec.get((seed, w, t), collections.Counter())
        recs.append({"def": games_meta[seed]["leaders"][1 - w], "band": turn_band(t), "theory": float(r["reach_theory"] or 0),
                     **{k2: float(q.get(k2, 0)) for k2 in ("leader_attacks", "reached", "no_don", "resolved", "unresolved")}})

    def blk(rs):
        if not rs:
            return None
        n = len(rs)
        th = sum(x["theory"] for x in rs); re_ = sum(x["reached"] for x in rs); res = sum(x["resolved"] for x in rs)
        return {"turns": n, "theory_reach_per_turn": round(th / n, 3), "actual_reach_per_turn": round(re_ / n, 3),
                "theory_over_actual": round(th / re_, 3) if re_ else None,
                "leader_attacks_per_turn": round(sum(x["leader_attacks"] for x in rs) / n, 3),
                "no_don_share": round(sum(x["no_don"] for x in rs) / res, 3) if res else None,
                "unresolved_attacks": int(sum(x["unresolved"] for x in rs))}
    decks = sorted({x["def"] for x in recs})
    out = {"record": a.src, "clocks": a.clocks, "by_defender": {d: {"all": blk([x for x in recs if x["def"] == d]),
                                                                   **{b: blk([x for x in recs if x["def"] == d and x["band"] == b]) for b in TURN_BANDS}}
                                                               for d in decks}}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("reach: rows %d" % len(recs))
    return 0


def cmd_counter(a):
    """カウンターの効き（2026-10-14・`2026-10-14_counter.md` §1・1 回きり）。相手のリーダーへの攻撃ごとに「宣言の行」から「攻める側の次の選択の行」までを
    区間とし、届いたか・超過・使ったカウンター（守る側の `SELECT_COUNTER`）・上がったパワー・止めたか（守る側のライフが減らない）を数える。
    時計の行（`OPCG_REACH_DUMP=1`）の理論の `stopped_theory`（決着までに手札で止めると見込む攻撃の数）と、その行から終局までに実際に
    カウンターで止めた攻撃の数を、守る側のデッキ × 帯で比べる。"""
    import theory_rs as TR
    from opcg_sim.learned.train import plan_labels as PL
    idx2cid = TR.idx2cid()
    meta = json.load(open(os.path.join(a.src, "meta_games.json"), encoding="utf-8"))
    games_meta = {int(g["seed"]): dict(g, leaders=seat_leaders(meta, g)) for g in meta["games"]}
    ck = json.load(open(a.clocks, encoding="utf-8"))
    g2s = {int(r["g"]): int(r["seed"]) for r in ck["rows_out"]}
    tseq, settled = collections.defaultdict(list), set()
    for sd, w, t, done in ck["settled"]:
        tseq[(int(sd), int(w))].append(int(t))
        if done:
            settled.add((int(sd), int(w), int(t)))
    for k in tseq:
        tseq[k].sort()
    cols = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen")
    atks, zof = [], {}
    stops = collections.defaultdict(lambda: collections.Counter())       # (seed, 守る席) → {t: カウンターで止めた数}
    for rows, pol, ex, L, ptr, idx in PL.iter_games([a.src], row_cols=cols, pol_cols=("pol_sig", "pol_cid", "pol_tcid"),
                                                    extra_fn=lambda dd, n: {"sc": np.asarray(dd["scalars"])[:n].astype(float),
                                                                            "ci": np.asarray(dd["card_idx"])[:n],
                                                                            "tok": np.asarray(dd["tokens"])[:n].astype(float)}):
        seed = int(rows["seed"][idx[0]])
        gm = games_meta.get(seed)
        if gm is None:
            continue
        lead = gm["leaders"]
        u2c = PL.uuid_map(pol, L, ptr, idx)
        idl = list(idx)
        for i in idl:
            zof[(seed, int(rows["who"][i]))] = float(rows["z"][i])
        for k, i in enumerate(idl[:-1]):
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t) or int(rows["kind"][i]) != 0:
                continue
            sj = json.loads(rows["sig"][i])
            if sj[0] not in ("ATTACK", "DON_BOX") or not sj[2] or u2c.get(sj[2][0]) != lead[1 - w]:
                continue
            d = 1 - w
            acid = u2c.get(sj[1])
            j = idl[k + 1]
            wj = int(rows["who"][j]); ci = ex["ci"][j]; tok = ex["tok"][j]; sc = ex["sc"][j]
            slots = ([0] + list(range(2, 7))) if wj == w else ([1] + list(range(7, 12)))
            sl = next((q for q in slots if acid and idx2cid.get(int(ci[q])) == acid), None)
            if sl is None:
                continue
            dpow0 = (sc[13] if wj == w else sc[12]) * 1e4
            apow = float(tok[sl][0]) * 1e4
            life0 = ex["sc"][i][1]
            counters, pmax, lmin = 0, dpow0, life0
            for j2 in idl[k + 1:]:
                w2 = int(rows["who"][j2])
                if int(rows["turn"][j2]) != t or (w2 == w and int(rows["kind"][j2]) == 0):
                    break
                sc2 = ex["sc"][j2]
                dp = (sc2[12] if w2 == d else sc2[13]) * 1e4
                lf = sc2[0] if w2 == d else sc2[1]
                pmax = max(pmax, dp); lmin = min(lmin, lf)
                if w2 == d and json.loads(rows["sig"][j2])[0] == "SELECT_COUNTER":
                    counters += 1
            reached = apow >= dpow0 - 1
            stopped = lmin >= life0 - 0.5
            cstop = stopped and counters > 0
            atks.append({"def": lead[d], "band": turn_band(t), "reached": reached, "excess": apow - dpow0, "counters": counters,
                         "raise": pmax - dpow0, "stopped": stopped, "cstop": cstop})
            if cstop:
                stops[(seed, d)][t] += 1

    def ablk(xs):
        r = [x for x in xs if x["reached"]]
        if not r:
            return None
        cs = [x for x in r if x["cstop"]]
        used = sum(x["counters"] for x in r)
        return {"reached_attacks": len(r), "counters_per_reached": round(used / len(r), 3),
                "raise_per_reached": round(sum(max(0.0, x["raise"]) for x in r) / len(r), 1),
                "stops_per_counter": round(len(cs) / used, 3) if used else None,
                "stopped_share": round(sum(x["stopped"] for x in r) / len(r), 3), "counter_stop_share": round(len(cs) / len(r), 3),
                "excess_mean_counter_stops": round(float(np.mean([x["excess"] for x in cs])), 1) if cs else None,
                "counters_mean_counter_stops": round(float(np.mean([x["counters"] for x in cs])), 3) if cs else None}
    recs = []
    for r in ck["turn_harm"]:
        seed = g2s.get(int(r["g"]))
        if seed is None or seed not in games_meta or "stopped_theory" not in r:
            continue
        w = int(r["who"]); jj = int(r["j"]); ts = tseq.get((seed, w), [])
        if jj >= len(ts):
            continue
        t = ts[jj]
        if (seed, w, t) in settled:
            continue
        d = 1 - w
        act = sum(v for tt, v in stops.get((seed, d), {}).items() if tt >= t)
        recs.append({"def": games_meta[seed]["leaders"][d], "band": turn_band(t), "theory": float(r["stopped_theory"] or 0.0),
                     "cut": float(r.get("cut_theory") or 0.0), "actual": float(act), "fell": (zof.get((seed, d)) or 0.0) < 0.5})

    def rblk(xs):
        if not xs:
            return None
        th = sum(x["theory"] for x in xs); ac = sum(x["actual"] for x in xs)
        return {"rows": len(xs), "theory_stopped": round(th / len(xs), 3), "actual_counter_stops": round(ac / len(xs), 3),
                "actual_over_theory": round(ac / th, 3) if th > 1e-12 else None, "theory_cut": round(sum(x["cut"] for x in xs) / len(xs), 3)}
    decks = sorted({x["def"] for x in atks})
    out = {"record": a.src, "clocks": a.clocks, "attacks": {}, "horizon": {}}
    for dk in decks:
        out["attacks"][dk] = {"all": ablk([x for x in atks if x["def"] == dk]), **{b: ablk([x for x in atks if x["def"] == dk and x["band"] == b]) for b in TURN_BANDS}}
        out["horizon"][dk] = {}
        for b in ("all",) + TURN_BANDS:
            xs = [x for x in recs if x["def"] == dk and (b == "all" or x["band"] == b)]
            out["horizon"][dk][b] = {"both": rblk(xs), "fell": rblk([x for x in xs if x["fell"]]), "survived": rblk([x for x in xs if not x["fell"]])}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("counter: attacks %d rows %d" % (len(atks), len(recs)))
    return 0


# ------------------------------------------------------------------ playgap（2026-10-14・`2026-10-14_play_gap.md` §1）
def _mix64(x):
    x &= 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def playgap_points(src, rowlist=None):
    """§1.3: 記録の自席ターンのメインの判断の行（効果の途中の選択を除く・候補 2 本以上）を (席のデッキ, 帯) ごとに。
    戻り: ({(deck, band): [(hash, seed, step, w, t)]}, {(seed, w): {band: 行の数}}, {(seed, w): deck})。"""
    from opcg_sim.learned.train import plan_labels as PL
    meta = json.load(open(os.path.join(src, "meta_games.json"), encoding="utf-8"))
    gms = {int(g["seed"]): dict(g, leaders=seat_leaders(meta, g)) for g in meta["games"]}
    pts, cnt, dof = collections.defaultdict(list), collections.defaultdict(collections.Counter), {}
    cols = ("who", "turn", "seed", "kind", "step", "pol_len")
    for rows, pol, _ex, L, ptr, idx in PL.iter_games([src], row_cols=cols, pol_cols=("pol_sig",)):
        seed = int(rows["seed"][idx[0]])
        gm = gms.get(seed)
        if gm is None:
            continue
        for w in (0, 1):
            dof[(seed, w)] = gm["leaders"][w]
            cnt[(seed, w)]
        for i in idx:
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t) or int(rows["kind"][i]) != 0 or int(L[i]) < 2:
                continue
            if json.loads(pol["pol_sig"][ptr[i]])[0] == "RESOLVE_EFFECT_SELECTION":
                continue
            st = int(rows["step"][i]); b = turn_band(t)
            pts[(gm["leaders"][w], b)].append((_mix64(seed * 1000003 + st), seed, st, w, t))
            cnt[(seed, w)][b] += 1
            if rowlist is not None:
                rowlist.append((seed, w, t))
    return pts, cnt, dof


def cmd_playdeep(a):
    """§1.3 の物差し 2: 選んだ点の盤面を記録と同じ設定で打ち直して再現し、シミュレーションを `--mult` 倍にした探索（雑音・温度なし）を回す。
    局ごとに 1 行の jsonl（在る局は飛ばす＝再開できる）。"""
    from opcg_sim.loop import decks as D
    from opcg_sim.loop import driver as DR
    from opcg_sim.loop import engine as E
    from opcg_sim.loop import record_gen as RG
    pts, _cnt, dof = playgap_points(a.src)
    pick = collections.defaultdict(dict)
    for key, xs in sorted(pts.items()):
        for h, seed, st, w, t in sorted(xs)[:a.per]:
            pick[seed][st] = (w, t, key[0], key[1])
    seeds = sorted(pick)
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        seeds = [s for k, s in enumerate(seeds) if k % n == i]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            done.add(int(json.loads(line)["seed"]))
    meta = json.load(open(os.path.join(a.src, "meta_n_record.json"), encoding="utf-8"))
    # 照合は「記録と同じ手を選んだか」で読む（符号化の行は新しいエンジンで列が変わったので、ビットの照合は使えない・報告 §2）
    from opcg_sim.learned.train import plan_labels as PL
    rec, want_s = {}, set(seeds)
    for rows, _pol, _ex, L, _ptr, idx in PL.iter_games([a.src], row_cols=("seed", "kind", "step", "sig", "pol_len"), pol_cols=()):
        sd = int(rows["seed"][idx[0]])
        if sd in want_s:
            rec[sd] = {int(rows["step"][i]): (json.loads(rows["sig"][i]), int(L[i])) for i in idx if int(rows["kind"][i]) == 0}
    E.engine()
    search = {"worlds": int(meta["search"]["worlds"])} if (meta.get("search") or {}).get("worlds") else {}
    spec = E.SeatSpec(meta.get("net"), sims=int(meta["sims"]), dirichlet_eps=float(meta["dirichlet_eps"]),
                      temp_turns=int(meta["temp_turns"]), prune_futile=E.GEN_PRUNE_FUTILE, **search)
    deep = E.SeatSpec(meta.get("net"), sims=int(meta["sims"]) * a.mult, prune_futile=E.GEN_PRUNE_FUTILE, **search)
    db = D.load_db()

    class _Stop(Exception):
        pass
    import time
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as fo:
        for seed in seeds:
            if seed in done:
                continue
            want = pick[seed]
            rr = rec.get(seed, {})
            st_ = {"bad": False, "left": len(want), "pts": [], "lost": 0}

            def obs(game, name, turn, step, out, move):
                if out.get("kind") != "main":
                    return
                ref = rr.get(int(step))
                same = ref is not None and list(ref[0]) == list(out.get("sig") or []) and ref[1] == len(out.get("groups") or [])
                if int(step) not in want:
                    if not same:
                        st_["bad"] = True
                    return
                if not same:
                    st_["bad"] = True
                st_["left"] -= 1
                w, t, deck, band = want[int(step)]
                if st_["bad"]:
                    st_["lost"] += 1
                else:
                    o2 = json.loads(game.fork().decide(name, deep.decide_opts(seed, int(turn), name, None)))
                    legal, legal2 = out["stats"]["legal"], o2["stats"]["legal"]

                    def chosen(o, lg):
                        # 選んだ束は記録と同じ鍵（箱レベルの sig と付けるドンの枚数・`record_gen._Recorder`）で引く
                        k = o.get("k")
                        for g in o["groups"]:
                            rep = lg[g["rep"]]
                            dk = (rep.get("payload") or {}).get("don_k")
                            if RG.move_sig(rep) == o.get("sig") and (None if dk is None else float(dk)) == (None if k is None else float(k)):
                                return g
                        return None
                    gp0, gd = chosen(out, legal), chosen(o2, legal2)
                    rec_ = {"step": int(step), "w": w, "t": t, "deck": deck, "band": band, "groups": len(out["groups"]),
                            "deep_groups": len(o2["groups"]), "same_legal": legal == legal2}
                    g2of = {g["rep"]: g for g in o2["groups"]}
                    gp = g2of.get(gp0["rep"]) if gp0 is not None else None
                    if gp is None or gd is None or legal != legal2:
                        rec_["unmapped"] = True
                    else:
                        rec_.update({"agree": gp["rep"] == gd["rep"], "q_played": gp["q"], "n_played": gp["n"],
                                     "q_deep": gd["q"], "n_deep": gd["n"],
                                     "q_max": max(g["q"] for g in o2["groups"] if g["n"] > 0),
                                     "shallow_q_played": gp0["q"], "shallow_n_played": gp0["n"]})
                        if gp["n"] > 0:
                            rec_["loss"] = (gd["q"] - gp["q"]) / 2.0
                        # 記録の候補の並び（`out["groups"]`＝記録の `pol` の順）での束の番号と、深い探索の束ごとの訪問・値（`theory_vs_deep` §1.2）
                        ri = {g["rep"]: k for k, g in enumerate(out["groups"])}
                        rec_["gi_played"] = ri.get(gp0["rep"])
                        rec_["gi_deep"] = ri.get(gd["rep"])
                        rec_["deep_nq"] = [[g2of[g["rep"]]["n"], g2of[g["rep"]]["q"]] if g["rep"] in g2of else None for g in out["groups"]]
                    st_["pts"].append(rec_)
                if st_["left"] <= 0 or st_["bad"]:
                    raise _Stop()

            def obs_(game, name, turn, step, out, move):
                obs(game, name, turn, step, out, move)
                if st_["bad"]:
                    raise _Stop()

            la, lb = D.leader_pair(db, seed, "random")
            p1, p2 = D.build_pair(db, la, lb, seed, meta["decks"])
            t1 = time.time()
            try:
                DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, max_steps=RG.MAX_STEPS, observer=obs_)
            except _Stop:
                pass
            lost = st_["lost"] + max(0, st_["left"])
            fo.write(json.dumps({"seed": seed, "diverged": st_["bad"], "lost": lost, "points": st_["pts"]}) + "\n")
            fo.flush()
            print("PLAYDEEP seed=%d pts=%d lost=%d bad=%d %.1fs (total %.0fs)" % (seed, len(st_["pts"]), lost, st_["bad"],
                                                                                  time.time() - t1, time.time() - t0), flush=True)
    return 0


def cmd_playgap(a):
    """§1.2・§1.3 の集計: 物差し 1（理論の値段の写し）と物差し 2（`playdeep` の jsonl）。"""
    _pts, cnt, dof = playgap_points(a.src)
    decks = sorted(set(dof.values()))
    bands = TURN_BANDS
    out = {"record": a.src, "dump": a.dump, "deep": a.deep}
    rng = np.random.default_rng(SEED)

    def summarize(per_unit, reps=BOOT, seeds=None):
        """per_unit: {(seed, w): {band: 量}} → デッキ × 帯（と全体）の 局×席の平均・エネル − 他・ナミ中盤 − 他・局の再抽出の区間。
        `seeds`＝数える局（写しに在る局だけ・その局の行の無い帯は 0）。"""
        seeds = sorted(seeds if seeds is not None else {s for s, _ in dof})

        def means(ss):
            m = {}
            for dk in decks:
                us = [(s, w) for s in ss for w in (0, 1) if dof.get((s, w)) == dk]
                if not us:
                    continue
                for b in bands + ("all",):
                    vals = [sum(per_unit.get(u, {}).values()) if b == "all" else per_unit.get(u, {}).get(b, 0.0) for u in us]
                    m[(dk, b)] = float(np.mean(vals))
            return m

        def contrasts(m):
            o = {}
            for focus, b in (("OP15-058", "all"), ("OP11-041", "4-6")):
                oth = [m[(dk, b)] for dk in decks if dk != focus and (dk, b) in m]
                if (focus, b) in m and oth:
                    o["%s|%s" % (focus, b)] = m[(focus, b)] - float(np.mean(oth))
            return o
        m0 = means(seeds)
        c0 = contrasts(m0)
        boots = collections.defaultdict(list)
        for _ in range(reps):
            ss = [seeds[k] for k in rng.integers(0, len(seeds), len(seeds))]
            for k, v in contrasts(means(ss)).items():
                boots[k].append(v)
        return {"by_deck": {dk: {b: round(m0[(dk, b)], 4) for b in bands + ("all",) if (dk, b) in m0} for dk in decks},
                "contrast": {k: {"diff": round(v, 4), "ci": [round(float(np.percentile(boots[k], 2.5)), 4),
                                                             round(float(np.percentile(boots[k], 97.5)), 4)]} for k, v in c0.items()}}

    # ---- 物差し 1
    if a.dump:
        stub = {"cap": 99.0, "rows_out": [], "turn_harm": [], "settled": []}
        rows_all, align, *_r = build_rows(a.src, load_dump(a.dump), stub)
        rows = [r for r in rows_all if r["pv"] is not None and r["bv"] is not None and r["nsc"] >= 2]
        dseeds = set(load_dump(a.dump))
        per = collections.defaultdict(lambda: collections.defaultdict(float))
        nrow = collections.defaultdict(lambda: collections.defaultdict(int))
        for r in rows:
            v = (float(r["bv"]) - float(r["pv"])) * float(r["kap"])
            per[(r["seed"], r["w"])][turn_band(r["t"])] += v
            nrow[(r["seed"], r["w"])][turn_band(r["t"])] += 1
        out["theory"] = {"align": dict(align), "rows_all": len(rows_all), "rows": len(rows),
                         "per_row_by_deck": {dk: {b: round(float(np.mean([(float(r["bv"]) - float(r["pv"])) * float(r["kap"]) for r in rows
                                                                          if dof.get((r["seed"], r["w"])) == dk and turn_band(r["t"]) == b] or [float("nan")])), 5)
                                                  for b in bands} for dk in decks},
                         "per_row_followed": {dk: round(float(np.mean([r["ptype"] == r["btype"] and abs(float(r["bv"]) - float(r["pv"])) < 1e-12
                                                                       for r in rows if dof.get((r["seed"], r["w"])) == dk])), 4) for dk in decks},
                         "games": len(dseeds),
                         "rows_per_game": summarize({u: dict(v) for u, v in nrow.items()}, reps=1, seeds=dseeds)["by_deck"],
                         **summarize({u: dict(v) for u, v in per.items()}, seeds=dseeds)}
    # ---- 物差し 2
    if a.deep:
        games = [json.loads(line) for line in open(a.deep, encoding="utf-8")]
        P = [dict(p, seed=g["seed"]) for g in games for p in g["points"]]
        rows_pg = {}
        for dk in decks:
            us = [u for u in dof if dof[u] == dk]
            for b in bands:
                rows_pg[(dk, b)] = float(np.mean([cnt[u].get(b, 0) for u in us]))
        d2 = {"games": len(games), "diverged_games": sum(g["diverged"] for g in games), "points": len(P),
              "lost_points": sum(g["lost"] for g in games), "unmapped": sum(1 for p in P if p.get("unmapped")),
              "no_visit_played": sum(1 for p in P if not p.get("unmapped") and "loss" not in p), "by_deck": {}, "contrast": {}}

        def est(sel):
            """点の集合 → デッキ × 帯の (一致率, 1 点の劣る量, 1 局の劣る量)。"""
            o = {}
            for dk in decks:
                tot, agw, n_ag = 0.0, 0.0, 0.0
                for b in bands:
                    ps = [p for p in sel if p["deck"] == dk and p["band"] == b and not p.get("unmapped")]
                    ls = [p["loss"] for p in ps if "loss" in p]
                    ag = float(np.mean([p["agree"] for p in ps])) if ps else float("nan")
                    lm = float(np.mean(ls)) if ls else float("nan")
                    o[(dk, b)] = (ag, lm, lm * rows_pg[(dk, b)], len(ps))
                    if ls:
                        tot += lm * rows_pg[(dk, b)]
                    if ps:
                        agw += ag * rows_pg[(dk, b)]; n_ag += rows_pg[(dk, b)]
                o[(dk, "all")] = (agw / n_ag if n_ag else float("nan"), float("nan"), tot, sum(o[(dk, b)][3] for b in bands))
            return o

        def ctr(o):
            c = {}
            for focus, b in (("OP15-058", "all"), ("OP11-041", "4-6")):
                oth = [o[(dk, b)][2] for dk in decks if dk != focus]
                c["%s|%s" % (focus, b)] = o[(focus, b)][2] - float(np.mean(oth))
                otha = [o[(dk, b)][0] for dk in decks if dk != focus]
                c["agree:%s|%s" % (focus, b)] = o[(focus, b)][0] - float(np.mean(otha))
            return c
        o0 = est(P)
        c0 = ctr(o0)
        by = collections.defaultdict(list)
        for p in P:
            by[(p["deck"], p["band"])].append(p)
        boots = collections.defaultdict(list)
        for _ in range(BOOT):
            sel = []
            for k, xs in by.items():
                sel.extend(xs[j] for j in rng.integers(0, len(xs), len(xs)))
            for k, v in ctr(est(sel)).items():
                boots[k].append(v)
        for dk in decks:
            d2["by_deck"][dk] = {b: {"agree": round(o0[(dk, b)][0], 4), "loss_per_point": None if b == "all" else round(o0[(dk, b)][1], 5),
                                     "loss_per_game": round(o0[(dk, b)][2], 4), "points": o0[(dk, b)][3],
                                     "rows_per_game": None if b == "all" else round(rows_pg[(dk, b)], 3)} for b in bands + ("all",)}
        d2["contrast"] = {k: {"diff": round(v, 4), "ci": [round(float(np.nanpercentile(boots[k], 2.5)), 4),
                                                          round(float(np.nanpercentile(boots[k], 97.5)), 4)]} for k, v in c0.items()}
        out["deep"] = d2
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k].get("contrast") for k in ("theory", "deep") if k in out}, ensure_ascii=False))
    return 0


def cmd_playsplit(a):
    """`2026-10-14_play_gap_split.md` §1: `playgap` の 2 本の物差しを、決着前・決着後（その局で最初に決着のフラグが立ったターン以降）に分ける。"""
    rowlist = []
    _pts, _cnt, dof = playgap_points(a.src, rowlist)
    decks = sorted(set(dof.values()))
    clocks = json.load(open(a.clocks, encoding="utf-8"))
    first = collections.defaultdict(lambda: 10 ** 9)
    for s_, _w, t_, d_ in clocks["settled"]:
        if d_:
            first[int(s_)] = min(first[int(s_)], int(t_))

    def ph(seed, t):
        return "post" if t >= first[seed] else "pre"
    phases = ("pre", "post")
    keys = [(b, p_) for b in TURN_BANDS for p_ in phases]
    rng = np.random.default_rng(SEED)
    FOCUS = (("OP15-058", "all"), ("OP11-041", "4-6"), ("OP16-022", "all"))

    def contrasts(m, sel):
        o = {}
        for focus, b in FOCUS:
            oth = [m[(dk, b, sel)] for dk in decks if dk != focus]
            o["%s|%s" % (focus, b)] = m[(focus, b, sel)] - float(np.mean(oth))
        return o

    def deck_means(per_unit, units):
        """per_unit[(seed,w)][(band,phase)] → {(deck, band|all, pre|post|both): 局×席の平均}"""
        m = {}
        for dk in decks:
            us = [u for u in units if dof.get(u) == dk]
            for b in TURN_BANDS + ("all",):
                for sel in phases + ("both",):
                    vals = []
                    for u in us:
                        d = per_unit.get(u, {})
                        vals.append(sum(v for (bb, pp), v in d.items() if (b == "all" or bb == b) and (sel == "both" or pp == sel)))
                    m[(dk, b, sel)] = float(np.mean(vals)) if vals else float("nan")
        return m

    def pack(m):
        return {dk: {sel: {b: round(m[(dk, b, sel)], 4) for b in TURN_BANDS + ("all",)} for sel in phases + ("both",)} for dk in decks}

    def share(m):
        return {dk: round(m[(dk, "all", "post")] / m[(dk, "all", "both")], 4) if m[(dk, "all", "both")] else None for dk in decks}
    out = {"record": a.src, "dump": a.dump, "deep": a.deep, "clocks": a.clocks,
           "settled_games": sum(1 for v in first.values() if v < 10 ** 9)}
    # ---- 物差し 1
    stub = {"cap": 99.0, "rows_out": [], "turn_harm": [], "settled": []}
    dump = load_dump(a.dump)
    dseeds = sorted(dump)
    rows_all, _al, *_r = build_rows(a.src, dump, stub)
    per = collections.defaultdict(lambda: collections.defaultdict(float))
    nrow = collections.defaultdict(collections.Counter)
    for r in rows_all:
        if r["pv"] is None or r["bv"] is None or r["nsc"] < 2:
            continue
        k = (turn_band(r["t"]), ph(r["seed"], r["t"]))
        per[(r["seed"], r["w"])][k] += (float(r["bv"]) - float(r["pv"])) * float(r["kap"])
        nrow[(r["seed"], r["w"])][k] += 1
    units = [(s_, w) for s_ in dseeds for w in (0, 1)]
    m1 = deck_means(per, units)
    c1 = {sel: contrasts(m1, sel) for sel in phases + ("both",)}
    boots = collections.defaultdict(list)
    for _ in range(BOOT):
        ss = [dseeds[k] for k in rng.integers(0, len(dseeds), len(dseeds))]
        mb = deck_means(per, [(s_, w) for s_ in ss for w in (0, 1)])
        for sel in phases + ("both",):
            for k, v in contrasts(mb, sel).items():
                boots[(sel, k)].append(v)
    out["theory"] = {"by_deck": pack(m1), "rows_per_game": pack(deck_means(nrow, units)), "post_share": share(m1),
                     "contrast": {sel: {k: {"diff": round(v, 4), "ci": [round(float(np.percentile(boots[(sel, k)], 2.5)), 4),
                                                                       round(float(np.percentile(boots[(sel, k)], 97.5)), 4)]}
                                        for k, v in c1[sel].items()} for sel in phases + ("both",)}}
    # ---- 物差し 2
    games = [json.loads(line) for line in open(a.deep, encoding="utf-8")]
    P = [dict(p, seed=g["seed"], ph=ph(g["seed"], p["t"])) for g in games for p in g["points"] if not p.get("unmapped")]
    rcnt = collections.defaultdict(collections.Counter)
    for s_, w, t in rowlist:
        rcnt[(s_, w)][(turn_band(t), ph(s_, t))] += 1
    allu = list(dof)
    rpg = {}
    for dk in decks:
        us = [u for u in allu if dof[u] == dk]
        for k in keys:
            rpg[(dk,) + k] = float(np.mean([rcnt[u].get(k, 0) for u in us]))
    by = collections.defaultdict(list)
    for p in P:
        by[(p["deck"], p["band"], p["ph"])].append(p)
    few = {"|".join(k): len(v) for k, v in by.items() if len(v) < 20}

    def est(byx):
        o = {}
        for dk in decks:
            for p_ in phases:
                pool = [x for b in TURN_BANDS for x in byx.get((dk, b, p_), [])]
                pool_l = [x["loss"] for x in pool if "loss" in x]
                fallback = float(np.mean(pool_l)) if pool_l else 0.0
                for b in TURN_BANDS:
                    xs = byx.get((dk, b, p_), [])
                    ls = [x["loss"] for x in xs if "loss" in x]
                    lm = float(np.mean(ls)) if len(xs) >= 20 and ls else fallback
                    ag = float(np.mean([x["agree"] for x in xs])) if xs else float("nan")
                    o[(dk, b, p_)] = (lm * rpg[(dk, b, p_)], ag, len(xs), lm)
        m = {}
        for dk in decks:
            for b in TURN_BANDS + ("all",):
                for sel in phases + ("both",):
                    bs = TURN_BANDS if b == "all" else (b,)
                    ps = phases if sel == "both" else (sel,)
                    m[(dk, b, sel)] = sum(o[(dk, bb, pp)][0] for bb in bs for pp in ps)
        return o, m
    o2, m2 = est(by)
    c2 = {sel: contrasts(m2, sel) for sel in phases + ("both",)}
    boots2 = collections.defaultdict(list)
    for _ in range(BOOT):
        bb = {k: [v[j] for j in rng.integers(0, len(v), len(v))] for k, v in by.items()}
        _o, mb = est(bb)
        for sel in phases + ("both",):
            for k, v in contrasts(mb, sel).items():
                boots2[(sel, k)].append(v)
    agree = {}
    for dk in decks:
        for p_ in phases:
            w_ = [(o2[(dk, b, p_)][1], rpg[(dk, b, p_)]) for b in TURN_BANDS if o2[(dk, b, p_)][2] > 0]
            agree["%s|%s" % (dk, p_)] = round(sum(x * y for x, y in w_) / sum(y for _, y in w_), 4) if w_ and sum(y for _, y in w_) else None
    out["deep"] = {"points": {"%s|%s|%s" % k: len(v) for k, v in sorted(by.items())}, "few_points": few,
                   "cells": {"%s|%s|%s" % k: {"loss_per_game": round(v[0], 4), "agree": None if math.isnan(v[1]) else round(v[1], 4),
                                              "points": v[2], "loss_per_point": round(v[3], 5), "rows_per_game": round(rpg[k], 3)}
                             for k, v in o2.items()},
                   "by_deck": pack(m2), "post_share": share(m2), "agree": agree,
                   "contrast": {sel: {k: {"diff": round(v, 4), "ci": [round(float(np.percentile(boots2[(sel, k)], 2.5)), 4),
                                                                     round(float(np.percentile(boots2[(sel, k)], 97.5)), 4)]}
                                      for k, v in c2[sel].items()} for sel in phases + ("both",)}}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({"theory": out["theory"]["post_share"], "deep": out["deep"]["post_share"]}, ensure_ascii=False))
    return 0


def cmd_theorydeep(a):
    """`2026-10-14_theory_vs_deep.md` §1: 深い探索の点（`playdeep` の束の番号つき）ごとに、記録の手・深い探索の手・理論の最善（値段の写し）を並べる。"""
    _pts, _cnt, dof = playgap_points(a.src)
    decks = sorted(set(dof.values()))
    stub = {"cap": 99.0, "rows_out": [], "turn_harm": [], "settled": []}
    rows_all, _al, *_r = build_rows(a.src, load_dump(a.dump), stub)
    drow = {(r["seed"], r["step"]): r for r in rows_all}
    cnt = collections.Counter()
    P = []
    for line in open(a.deep, encoding="utf-8"):
        g = json.loads(line)
        for p in g["points"]:
            cnt["points"] += 1
            if p.get("unmapped") or p.get("gi_played") is None or p.get("gi_deep") is None:
                cnt["unmapped"] += 1
                continue
            r = drow.get((g["seed"], p["step"]))
            if r is None or r["nsc"] < 2:
                cnt["no_theory_row"] += 1
                continue
            pr = [c[1] for c in r["cands"]]
            if len(pr) != p["groups"]:
                cnt["len_mismatch"] += 1
                continue
            gp, gd = p["gi_played"], p["gi_deep"]
            priced = [k for k, v in enumerate(pr) if v is not None]
            mx = max(pr[k] for k in priced)
            best = [k for k in priced if abs(pr[k] - mx) < 1e-12]
            agree = gp == gd
            rec = {"seed": g["seed"], "deck": p["deck"], "band": p["band"], "agree": agree, "tie": len(best) > 1,
                   "no_price": pr[gd] is None or pr[gp] is None}
            if gd in best:
                rec["to"] = "deep"
            elif gp in best:
                rec["to"] = "record"
            else:
                rec["to"] = "third"
            if not agree and not rec["no_price"]:
                rec["diff"] = (pr[gd] - pr[gp]) * float(r["kap"])
            # 記録の手が写しの打った手と同じか（照合）
            cnt["played_matches_dump"] += int(abs((pr[gp] if pr[gp] is not None else float("nan")) - float(r["pv"])) < 1e-12) if r["pv"] is not None else 0
            P.append(rec)
    cnt["used"] = len(P)
    rng = np.random.default_rng(SEED)

    def shares(xs):
        xs = [x for x in xs if not x["no_price"]]
        if not xs:
            return None
        n = len(xs)
        return {k: sum(1 for x in xs if x["to"] == k) / n for k in ("deep", "record", "third")}

    def blk(xs, boot=False):
        o = {"points": len(xs), "no_price": sum(1 for x in xs if x["no_price"]), "ties": sum(1 for x in xs if x["tie"])}
        sh = shares(xs)
        if sh is None:
            return o
        o.update({k: round(v, 4) for k, v in sh.items()})
        o["deep_minus_record"] = round(sh["deep"] - sh["record"], 4)
        if boot:
            xs2 = [x for x in xs if not x["no_price"]]
            bd, bm = [], []
            for _ in range(BOOT):
                ss = [xs2[j] for j in rng.integers(0, len(xs2), len(xs2))]
                s2 = shares(ss)
                bd.append(s2["deep"]); bm.append(s2["deep"] - s2["record"])
            o["deep_ci"] = [round(float(np.percentile(bd, 2.5)), 4), round(float(np.percentile(bd, 97.5)), 4)]
            o["deep_minus_record_ci"] = [round(float(np.percentile(bm, 2.5)), 4), round(float(np.percentile(bm, 97.5)), 4)]
        ds = [x["diff"] for x in xs if "diff" in x]
        if ds:
            o["diff"] = {"n": len(ds), "mean": round(float(np.mean(ds)), 5), "median": round(float(np.median(ds)), 5),
                         "pos_share": round(float(np.mean([d > 1e-12 for d in ds])), 4),
                         "q25": round(float(np.percentile(ds, 25)), 5), "q75": round(float(np.percentile(ds, 75)), 5)}
        return o
    out = {"record": a.src, "dump": a.dump, "deep": a.deep, "count": dict(cnt), "by_deck": {}}
    for dk in decks:
        out["by_deck"][dk] = {}
        for b in TURN_BANDS + ("all",):
            for ag in (True, False):
                xs = [x for x in P if x["deck"] == dk and (b == "all" or x["band"] == b) and x["agree"] == ag]
                out["by_deck"][dk]["%s|%s" % (b, "agree" if ag else "disagree")] = blk(xs, boot=(b == "all"))
    for ag in (True, False):
        out["all|%s" % ("agree" if ag else "disagree")] = blk([x for x in P if x["agree"] == ag], boot=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out["count"]))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("clocks", "analyze", "run", "turns", "deckcal", "paircal", "clockside", "guardside", "defturns", "parts", "reread", "speed", "namipower", "reach", "counter", "playdeep", "playgap", "playsplit", "theorydeep", "-h", "--help"):
        argv = ["run"] + argv                     # 入口の既定: `--in <記録> --out <json>`
    ap = argparse.ArgumentParser(description="理論の目で記録を読む（1 回きり）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clocks")
    c.add_argument("--in", dest="src", required=True)
    c.add_argument("--games", type=int, default=0)
    c.add_argument("--out", required=True)
    z = sub.add_parser("analyze")
    z.add_argument("--in", dest="src", required=True)
    z.add_argument("--dump", required=True)
    z.add_argument("--clocks", required=True)
    z.add_argument("--boot", type=int, default=BOOT)
    z.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--in", dest="src", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--work", default=None)
    r.add_argument("--games", type=int, default=0)
    r.add_argument("--boot", type=int, default=BOOT)
    t = sub.add_parser("turns")
    t.add_argument("--in", dest="src", required=True)
    t.add_argument("--work", default=None, help="前の報告の作業の置き場（dump.jsonl・clocks.json）")
    t.add_argument("--dump", default=None)
    t.add_argument("--clocks", default=None)
    t.add_argument("--boot", type=int, default=BOOT)
    t.add_argument("--out", required=True)
    dc = sub.add_parser("deckcal")
    dc.add_argument("--in", dest="src", required=True)
    dc.add_argument("--work", default=None)
    dc.add_argument("--dump", default=None)
    dc.add_argument("--clocks", default=None)
    dc.add_argument("--focus", default="OP15-058", help="切り分けるデッキ（リーダー）")
    dc.add_argument("--don-cap", default="rule10", choices=("rule10", "room"),
                    help="ドンの検算の計画の上限: rule10＝min(10, 場＋2k)（2026-10-12 までの式）／room＝min(ドン‼デッキの枚数, 場＋2k)（候補）")
    dc.add_argument("--boot", type=int, default=BOOT)
    dc.add_argument("--out", required=True)
    pc = sub.add_parser("paircal")
    pc.add_argument("--in", dest="src", required=True)
    pc.add_argument("--clocks", required=True)
    pc.add_argument("--min-games", type=int, default=MIN_GAMES)
    pc.add_argument("--boot", type=int, default=BOOT)
    pc.add_argument("--out", required=True)
    cs = sub.add_parser("clockside")
    cs.add_argument("--in", dest="src", required=True)
    cs.add_argument("--clocks", required=True)
    cs.add_argument("--boot", type=int, default=BOOT)
    cs.add_argument("--out", required=True)
    gs = sub.add_parser("guardside")
    gs.add_argument("--in", dest="src", required=True)
    gs.add_argument("--dump", required=True, help="`guard` 欄つきの写し（`theory_bridge.py` を `OPCG_PRICE_DUMP` で・2026-10-12 以降）")
    gs.add_argument("--out", required=True)
    dt = sub.add_parser("defturns")
    dt.add_argument("--in", dest="src", required=True)
    dt.add_argument("--out", required=True)
    pt = sub.add_parser("parts")
    pt.add_argument("--in", dest="src", required=True)
    pt.add_argument("--clocks", required=True)
    pt.add_argument("--dump", required=True, help="`guard` 欄つきの写し")
    pt.add_argument("--out", required=True)
    rr = sub.add_parser("reread")
    rr.add_argument("--in", dest="src", required=True)
    rr.add_argument("--out", required=True)
    sp = sub.add_parser("speed")
    sp.add_argument("--in", dest="src", required=True)
    sp.add_argument("--clocks", required=True)
    sp.add_argument("--out", required=True)
    npw = sub.add_parser("namipower")
    npw.add_argument("--in", dest="src", required=True)
    npw.add_argument("--out", required=True)
    rc = sub.add_parser("reach")
    rc.add_argument("--in", dest="src", required=True)
    rc.add_argument("--clocks", required=True, help="`OPCG_REACH_DUMP=1` で作った時計の行")
    rc.add_argument("--out", required=True)
    cn = sub.add_parser("counter")
    cn.add_argument("--in", dest="src", required=True)
    cn.add_argument("--clocks", required=True, help="`OPCG_REACH_DUMP=1` で作った時計の行")
    cn.add_argument("--out", required=True)
    pd = sub.add_parser("playdeep")
    pd.add_argument("--in", dest="src", required=True)
    pd.add_argument("--out", required=True, help="局ごとの jsonl（在る局は飛ばす）")
    pd.add_argument("--per", type=int, default=150, help="デッキ × 帯ごとの点の数")
    pd.add_argument("--mult", type=int, default=4, help="記録のシミュレーションの何倍で探索するか")
    pd.add_argument("--shard", default=None, help="i/n（局を n 本に分けた i 本目）")
    pg = sub.add_parser("playgap")
    pg.add_argument("--in", dest="src", required=True)
    pg.add_argument("--dump", default=None, help="今の既定で取り直した `OPCG_PRICE_DUMP` の写し（物差し 1）")
    pg.add_argument("--deep", default=None, help="`playdeep` の jsonl（物差し 2）")
    pg.add_argument("--out", required=True)
    ps = sub.add_parser("playsplit")
    ps.add_argument("--in", dest="src", required=True)
    ps.add_argument("--dump", required=True)
    ps.add_argument("--deep", required=True)
    ps.add_argument("--clocks", required=True, help="決着のフラグ（`settled`）の在る時計の行")
    ps.add_argument("--out", required=True)
    td = sub.add_parser("theorydeep")
    td.add_argument("--in", dest="src", required=True)
    td.add_argument("--dump", required=True)
    td.add_argument("--deep", required=True, help="`playdeep` の jsonl（束の番号つき・2026-10-14 以降）")
    td.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"clocks": cmd_clocks, "analyze": cmd_analyze, "run": cmd_run, "turns": cmd_turns, "deckcal": cmd_deckcal,
            "paircal": cmd_paircal, "clockside": cmd_clockside, "guardside": cmd_guardside,
            "defturns": cmd_defturns, "parts": cmd_parts, "reread": cmd_reread, "speed": cmd_speed, "namipower": cmd_namipower, "reach": cmd_reach, "counter": cmd_counter,
            "playdeep": cmd_playdeep, "playgap": cmd_playgap, "playsplit": cmd_playsplit, "theorydeep": cmd_theorydeep}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
