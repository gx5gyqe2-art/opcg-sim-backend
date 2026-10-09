"""`survival_split.py` の時計ごとの測りを集計する（2026-10-08・`docs/reports/2026-10-08_survival_split.md`・読み取り専用）。

実行例:
  python tests/scripts/survival_split_sum.py --recs split_real.jsonl --out split_real_sum.json
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

S_BANDS = ((-1.0, 1e-9, "0"), (1e-9, 0.25, "(0,.25]"), (0.25, 0.5, "(.25,.5]"), (0.5, 0.75, "(.5,.75]"),
           (0.75, 0.95, "(.75,.95]"), (0.95, 1.01, "(.95,1]"))
ALTS = ("A", "A1", "D", "AD")


def band(S):
    for lo, hi, name in S_BANDS:
        if lo < S <= hi:
            return name
    return "0"


def r4(x):
    return None if x is None else round(float(x), 4)


def oe(rs):
    """見張った窓（地平の内・観測のある段）で倒れる確率の和（予測）と、倒した数（観測）。差し替えの解き直しも。"""
    if not rs:
        return {"n": 0}
    o = {"n": len(rs), "obs": int(sum(r["ev"] for r in rs)), "exp": r4(sum(r["F_W"] for r in rs))}
    o["obs_rate"], o["exp_rate"] = r4(o["obs"] / len(rs)), r4(o["exp"] / len(rs))
    o["obs_win_only_rate"] = r4(sum(r["ev"] for r in rs) / max(1, sum(r["won"] for r in rs)))
    gap = o["obs"] - o["exp"]
    for k in ALTS:
        v = [r.get("F_" + k) for r in rs]
        if any(x is None for x in v):
            ok = [r for r in rs if r.get("F_" + k) is not None]
            o[k] = {"n": len(ok), "exp": r4(sum(r["F_" + k] for r in ok)),
                    "exp_base_same": r4(sum(r["F_W"] for r in ok)), "obs": int(sum(r["ev"] for r in ok))}
        else:
            o[k] = {"n": len(rs), "exp": r4(sum(v))}
        sub = o[k]
        base = sub.get("exp_base_same", o["exp"])
        obs = sub.get("obs", o["obs"])
        g = obs - base
        sub["closed"] = r4((sub["exp"] - base) / g) if abs(g) > 1e-9 else None
    o["gap"] = r4(gap)
    hs = [r for r in rs if r.get("hitsW_obs") is not None and r.get("hitsW_A") is not None]
    if hs:
        o["hits_cuts_window"] = {"n": len(hs),
                                 "hits_model": r4(np.mean([r["hitsW_model"] for r in hs])),
                                 "hits_dp_actual_attacks": r4(np.mean([r["hitsW_A"] for r in hs])),
                                 "hits_obs": r4(np.mean([r["hitsW_obs"] for r in hs])),
                                 "cuts_model": r4(np.mean([r["cutsW_model"] for r in hs])),
                                 "cuts_dp_actual_attacks": r4(np.mean([r["cutsW_A"] for r in hs])),
                                 "cuts_obs": r4(np.mean([r["cutsW_obs"] for r in hs])),
                                 "steps": r4(np.mean([r["W"] for r in hs]))}
    return o


def hazard(rs, smax=6):
    """段ごとの条件つきの倒れる割合（段 s まで見張れて、段 s より前に倒れていない時計）: 予測（最初の行から）・連鎖（段 s に実際に
    来た盤面の今のターンの予測）・観測。"""
    out = {}
    for s in range(1, smax + 1):
        at = [r for r in rs if r["W"] >= s]
        if not at:
            continue
        h0 = []
        for r in at:
            F = sum(r["pd0"][:s - 1])
            p = r["pd0"][s - 1]
            h0.append(p / (1.0 - F) if 1.0 - F > 1e-9 else 1.0)
        obs = [1.0 if (r["ev"] and r["act"] == s) else 0.0 for r in at]
        q = [r["steps"][s - 1].get("q1") for r in at]
        qa = [(hh, qq, oo) for hh, qq, oo in zip(h0, q, obs) if qq is not None]
        out[str(s)] = {"n": len(at), "pred_from_start": r4(np.mean(h0)), "obs": r4(np.mean(obs)),
                       "n_chain": len(qa),
                       "pred_from_start_chain_rows": r4(np.mean([a for a, _, _ in qa])) if qa else None,
                       "pred_now": r4(np.mean([b for _, b, _ in qa])) if qa else None,
                       "obs_chain_rows": r4(np.mean([c for _, _, c in qa])) if qa else None}
    return out


def later_vs_first(rs):
    """段 1 と段 2 以降で（予測の和・観測の数）を分ける。"""
    o = {}
    for name, cond in (("step1", lambda s: s == 1), ("later", lambda s: s >= 2)):
        e = ob = 0.0
        n = 0
        en = 0.0
        for r in rs:
            for s in range(1, r["W"] + 1):
                if not cond(s):
                    continue
                if r["ev"] and r["act"] < s:
                    break
                F = sum(r["pd0"][:s - 1])
                p = r["pd0"][s - 1]
                e += p / (1.0 - F) if 1.0 - F > 1e-9 else 1.0
                ob += 1.0 if (r["ev"] and r["act"] == s) else 0.0
                q = r["steps"][s - 1].get("q1")
                en += q if q is not None else 0.0
                n += 1
        o[name] = {"at_risk_steps": n, "pred_from_start": r4(e), "pred_now": r4(en), "obs": int(ob),
                   "obs_over_pred": r4(ob / e) if e > 0 else None, "obs_over_now": r4(ob / en) if en > 0 else None}
    return o


def sources(rs):
    o = {}
    allc = collections.Counter()
    killc = collections.Counter()
    kill_any_new = 0
    nk = 0
    for r in rs:
        for k, v in (r.get("src") or {}).items():
            allc[k] += v
        ks = r.get("kill_src")
        if ks is not None:
            nk += 1
            for k in ks:
                killc[k] += 1
            kill_any_new += int("new" in ks)
    ta, tk = sum(allc.values()), sum(killc.values())
    o["all_steps"] = {k: r4(v / ta) for k, v in sorted(allc.items())} if ta else {}
    o["all_steps_n"] = ta
    o["kill_turn"] = {k: r4(v / tk) for k, v in sorted(killc.items())} if tk else {}
    o["kill_turn_n"] = tk
    o["kill_turns"] = nk
    o["kill_turn_has_new"] = r4(kill_any_new / nk) if nk else None
    return o


def now_table(rs):
    """今のターン（段 1）: 計画の攻撃で言う倒れる確率・実際の攻撃で解き直した倒れる確率・観測。受けた本数・切った札も。"""
    sel = [r for r in rs if r.get("now_act") and r["steps"] and r["steps"][0].get("n_atk") is not None]
    if not sel:
        return {"n": 0}
    o = {"n": len(sel)}
    obs_kill = [1.0 if (r["ev"] and r["act"] == 1) else 0.0 for r in sel]
    o["q_model"] = r4(np.mean([r["now_model"]["q"] for r in sel]))
    o["q_actual_attacks"] = r4(np.mean([r["now_act"]["q"] for r in sel]))
    o["obs_kill"] = r4(np.mean(obs_kill))
    life = [r["steps"][0]["life_lost"] for r in sel]
    hits_obs = [(ll if ll is not None else 0) + (1 if k else 0) for ll, k in zip(life, obs_kill)]
    o["hits_model_actual_attacks"] = r4(np.mean([r["now_act"]["hits"] for r in sel]))
    o["hits_obs"] = r4(np.mean(hits_obs))
    o["cuts_model_actual_attacks"] = r4(np.mean([r["now_act"]["cuts"] for r in sel]))
    o["cuts_obs"] = r4(np.mean([r["steps"][0]["counters"] for r in sel]))
    o["blocks_obs"] = r4(np.mean([r["steps"][0]["blocks"] for r in sel]))
    o["n_attacks_leader"] = r4(np.mean([r["steps"][0]["n_atk"] for r in sel]))
    o["n_attacks_hit_x"] = r4(np.mean([r["steps"][0]["n_hit_x"] for r in sel]))
    # 帯: 計画で言う倒れる確率
    bands = {}
    for lo, hi, name in ((-1, 0.05, "<=.05"), (0.05, 0.5, "(.05,.5]"), (0.5, 0.95, "(.5,.95]"), (0.95, 2, ">.95")):
        ss = [r for r, k in zip(sel, obs_kill) if lo < r["now_act"]["q"] <= hi]
        if ss:
            bands[name] = {"n": len(ss), "q_actual_attacks": r4(np.mean([r["now_act"]["q"] for r in ss])),
                           "obs_kill": r4(np.mean([1.0 if (r["ev"] and r["act"] == 1) else 0.0 for r in ss]))}
    o["by_q_actual"] = bands
    return o


def summarise(recs):
    out = {"n": len(recs)}
    groups = {"all": recs, "S_lt_0.5": [r for r in recs if r["S"] < 0.5], "S_ge_0.5": [r for r in recs if r["S"] >= 0.5],
              "S_gt_0.95": [r for r in recs if r["S"] > 0.95]}
    for b in [x[2] for x in S_BANDS]:
        groups["band " + b] = [r for r in recs if band(r["S"]) == b]
    out["oe"] = {k: oe(v) for k, v in groups.items()}
    out["hazard"] = {k: hazard(groups[k]) for k in ("all", "S_lt_0.5", "S_ge_0.5", "S_gt_0.95")}
    out["first_vs_later"] = {k: later_vs_first(groups[k]) for k in ("all", "S_lt_0.5", "S_ge_0.5", "S_gt_0.95")}
    out["sources"] = {k: sources(groups[k]) for k in ("all", "S_lt_0.5", "S_ge_0.5")}
    out["now"] = {k: now_table(groups[k]) for k in ("all", "S_ge_0.5")}
    out["D_dropped_mean"] = {k: r4(np.mean([r.get("D_dropped", 0) for r in groups[k]])) if groups[k] else None
                             for k in ("all", "S_ge_0.5")}
    out["hand_cards_mean"] = {k: r4(np.mean([r["hand_cards"] for r in groups[k]])) if groups[k] else None
                              for k in ("all", "S_lt_0.5", "S_ge_0.5")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recs", nargs="+", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    recs = [json.loads(l) for p in a.recs for l in open(os.path.expanduser(p))]
    res = summarise(recs)
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        open(os.path.expanduser(a.out), "w").write(txt)
    print(txt)
    return 0




def chain_summary(rows):
    """段ごとの 2×2 の和（段 2 以降）: 最初の行からの予測 `h0`・守り手は実際＋攻撃は計画 `q_F`・＋新しい攻め手 `q_Fnew`・
    守り手も攻撃も実際 `q_A`・計画の今の予測 `q_now`・観測。差は順に「守り手の状態の読み」「新しい攻め手」「ほかの攻撃の読み」「今の打ち」。"""
    groups = {"all": rows, "S_lt_0.5": [r for r in rows if r["S0"] < 0.5], "S_ge_0.5": [r for r in rows if r["S0"] >= 0.5],
              "S_gt_0.95": [r for r in rows if r["S0"] > 0.95]}
    out = {}
    for k, rs in groups.items():
        rs = [r for r in rs if None not in (r["q_F"], r["q_Fnew"], r["q_A"], r.get("q_A0"), r.get("q_Aold"), r.get("hA_r0"), r.get("hA1_r0"))]
        if not rs:
            continue
        o = {"n": len(rs)}
        for f in ("h0", "q_F", "q_Fnew", "q_A", "q_now", "q_A0", "q_Aold", "hA_r0", "hA1_r0"):
            o[f] = r4(sum(r[f] for r in rs))
        o["obs"] = int(sum(r["obs"] for r in rs))
        gap = o["obs"] - o["h0"]
        o["gap"] = r4(gap)
        if abs(gap) > 1e-9:
            o["share"] = {"defender_state": r4((o["q_F"] - o["h0"]) / gap),
                          "new_attackers": r4((o["q_Fnew"] - o["q_F"]) / gap),
                          "other_attacks": r4((o["q_A"] - o["q_Fnew"]) / gap),
                          "in_the_moment": r4((o["obs"] - o["q_A"]) / gap)}
            o["share_reverse"] = {"new_attackers": r4((o["hA1_r0"] - o["h0"]) / gap),
                                  "other_attacks": r4((o["hA_r0"] - o["hA1_r0"]) / gap),
                                  "defender_state": r4((o["q_A"] - o["hA_r0"]) / gap),
                                  "in_the_moment": r4((o["obs"] - o["q_A"]) / gap)}
            o["share_actual_world"] = {"new_attackers": r4((o["q_A"] - o["q_Aold"]) / gap),
                                       "don_attached": r4((o["q_A"] - o["q_A0"]) / gap)}
        for f2 in ("q_A_don", "q_A_life", "q_A_blk", "q_A_hand"):
            ok = [r for r in rs if r.get(f2) is not None]
            o[f2] = {"n": len(ok), "q": r4(sum(r[f2] for r in ok)), "q_A_same": r4(sum(r["q_A"] for r in ok)),
                     "share_of_gap": r4((sum(r["q_A"] for r in ok) - sum(r[f2] for r in ok)) / gap) if abs(gap) > 1e-9 else None}
        for f in ("n_F", "n_A", "n_new", "sumx_F", "sumx_A", "cards0", "cards_s", "csum0", "csum_s", "life0", "life_s",
                  "don0", "don_s", "blk0", "blk_s", "k_A", "k_all", "lifeF_s", "cardsF_s"):
            o["mean_" + f] = r4(np.mean([r[f] for r in rs]))
        out[k] = o
    return out


def chain_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", nargs="+", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    rows = [json.loads(l) for p in a.chain for l in open(os.path.expanduser(p))]
    res = chain_summary(rows)
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        open(os.path.expanduser(a.out), "w").write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(chain_main(sys.argv[2:]) if sys.argv[1:2] == ["chain"] else main())
