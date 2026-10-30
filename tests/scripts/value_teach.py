"""WP `value-teach`（`docs/search_efficiency_plan.md` §7.71・台帳 2+485）の材料作りと試験台の数え。

読み取りだけ（エンジン・出荷・探索は変えない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。
訓練の足しそのもの（案 A の sidecar・案 B の損失）は `opcg_sim/learned/train/value_teach.py`。

1. ``pairs``: 打ち比べの記録（`window_read_all`・`main_read`・`matchup_kr`・`matchup_5`・`rocks_bounce`・`blunder_audit`）の
   各点を当て直し、手 a・手 b を当てた盤面（その手の続きの自分の選択と窓を出荷の設定で片付けた盤面＝次の判断点）を
   打った席の視点で符号化し、打ち続けた勝率（価値の目盛り 2w−1）と組にして npz に書く（案 B の教材）。
   **試験台の局（紫カイドウ 24 点・6キングだけ 53 点）と同じ局の点は外す**（局の単位）。``--shard i/n``。
2. ``rocks``: 試験台 2。6キングだけの 53 点で、ネットごとに出荷の設定の根の読み（`Game.decide`）を回し、
   ロックス海賊団を選ぶか・根の評価の差（戻す − 訪問最多の手）・ネットの値の差（戻した後の盤面 − 訪問最多の手の後の盤面）。
3. ``summary``: 試験台 1（`offdist_value.py evalnet`／`timeval` の出力）・試験台 2・評価帯をまとめる。

    python tests/scripts/value_teach.py pairs --workers 4 [--shard 0/4]
    python tests/scripts/value_teach.py merge
    python tests/scripts/value_teach.py rocks --nets r27.npz ctrl.npz a.npz b.npz --tag all
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blunder_audit as BA  # noqa: E402
import don_read as DRd  # noqa: E402
import main_read as MR  # noqa: E402
import matchup_kr as KR  # noqa: E402
import move_judge as MJ  # noqa: E402
import window_read_all as W  # noqa: E402

from opcg_sim.loop import driver as DR, engine as E  # noqa: E402

OUT_DIR = os.path.join("n1_results", "value_teach")
#: 配り方の番号のずらし（… offdist_value 14000・rocks_bounce 15000 と重ねない）。
OFFSET = 16000
K6, EV = "EB04-031", "OP17-056"
N_TOK = 22
_G = {}


# ---------------------------------------------------------------- 試験台の局（漏れの防止）

def kaido_key(p):
    return ("kr", int(p["idx"])) if p["src"] == "kr" else ("m5", p["mu"], int(p["idx"]))


def rocks_key(p):
    return ("kr", int(p["idx"])) if p["src"] == "kr" else ("m5", "kaido_rocks", int(p["idx"]))


def rocks_points():
    return [p for p in BA.read_jsonl(os.path.join("n1_results", "rocks_bounce", "points.jsonl")) if p["group"] == "k6_only"]


def bench_games():
    """試験台 1・2 の点を取った局の鍵（この局の点は対の教材に入れない）。"""
    kaido = [p for p in BA.read_jsonl(os.path.join(DRd.OUT_DIR, "points.jsonl")) if p["judged"] and p["item"] != "mihawk_leader"]
    return {kaido_key(p) for p in kaido} | {rocks_key(p) for p in rocks_points()}


# ---------------------------------------------------------------- 1. pairs

def _wr(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)


def _ref(r):
    return r["ref"] if isinstance(r.get("ref"), dict) else None


def collect_jobs():
    """(源, 局の鍵, 席, step, a, b, a の勝率, b の勝率, 打ち続けた本数, 6キングの uuid) の列。"""
    jobs = []
    # window_read_all（窓の手）: a＝今の手・b＝短い木の手
    wpts = {p["pid"]: p for p in BA.read_jsonl(os.path.join(W.OUT_DIR, "points.jsonl"))}
    wev = W.load_evals(W.OUT_DIR)
    for r in KR.read_glob(os.path.join(W.OUT_DIR, "judge*.jsonl"), lambda x: x["pid"]).values():
        if not r.get("ok"):
            continue
        p, e = wpts[r["pid"]], wev.get(r["pid"])
        if e is None:
            continue
        wa, na = _wr(r["ref"]["a"]); wb, nb = _wr(r["ref"]["b"])
        jobs.append(dict(src="window_read_all", g=list(p["g"]), seat=p["seat"], step=p["step"],
                         a=e["legal"][e["cur"]], b=e["legal"][e["prim"]], wa=wa, wb=wb, n=min(na, nb)))
    # main_read（メインの手）: a＝今の手・別の読みの手（big／next）ごとに 1 対
    mjobs = {p["pid"]: (p, alts) for p, _e, alts in MR.judge_jobs(MR.OUT_DIR)}
    for r in KR.read_glob(os.path.join(MR.OUT_DIR, "judge*.jsonl"), lambda x: x["pid"]).values():
        if not r.get("ok") or r["pid"] not in mjobs:
            continue
        p, alts = mjobs[r["pid"]]
        wa, na = _wr(r["res"]["a"])
        for nm, mv in sorted(alts.items()):
            src = r.get("same_" + nm) or nm
            if src == "a":
                continue
            wb, nb = _wr(r["res"][src])
            jobs.append(dict(src="main_read", g=list(p["g"]), seat=p["seat"], step=p["step"], a=p["a"], b=mv,
                             wa=wa, wb=wb, n=min(na, nb)))
    # matchup_kr／matchup_5（記事の手との打ち比べ）
    for name, pat, gk in (("matchup_kr", "n1_results/matchup_kr/judge_points_shard*.jsonl", lambda r: ["kr", r["idx"]]),
                          ("matchup_5", "n1_results/matchup_5/judge_points_shard*.jsonl", lambda r: ["m5", r["mu"], r["idx"]])):
        for f in sorted(glob.glob(pat)):
            for r in BA.read_jsonl(f):
                if not r.get("ok") or _ref(r) is None:
                    continue
                wa, na = _wr(r["ref"]["a"]); wb, nb = _wr(r["ref"]["b"])
                jobs.append(dict(src=name, g=gk(r), seat=r["seat"], step=r["step"], a=r["a"], b=r["b"],
                                 wa=wa, wb=wb, n=min(na, nb)))
    # rocks_bounce（b＝ロックス海賊団で 6キングを戻す）
    for f in sorted(glob.glob("n1_results/rocks_bounce/judge_shard*.jsonl")):
        for r in BA.read_jsonl(f):
            if not r.get("ok"):
                continue
            wa, na = _wr(r["ref"]["a"]); wb, nb = _wr(r["ref"]["b"])
            jobs.append(dict(src="rocks_bounce", g=list(rocks_key(r)), seat="p2", step=r["step"], a=r["a"], b=r["b"],
                             wa=wa, wb=wb, n=min(na, nb), k6=r["k6_uuid"]))
    # blunder_audit（自己対戦の点・次点の手との打ち比べ・審判は当時のネット）
    for r in BA.read_jsonl(os.path.join(BA.OUT_DIR, "points.jsonl")):
        ref = r.get("ref") if isinstance(r.get("ref"), dict) else None
        if not r.get("ok") or not ref or not r.get("b"):
            continue
        net, rr = next(iter(ref.items()))
        if not isinstance(rr, dict) or rr.get("diff") is None:
            continue
        wa, na = _wr(rr["a"]); wb, nb = _wr(rr["b"])
        jobs.append(dict(src="blunder_audit", g=["ba", r["seed"]], seat=r["seat"], step=r["step"], a=r["a"], b=r["b"],
                         wa=wa, wb=wb, n=min(na, nb), judge_net=net))
    jobs = [j for j in jobs if j["wa"] is not None and j["wb"] is not None and BA.key(j["a"]) != BA.key(j["b"])]
    for i, j in enumerate(jobs):
        j["jid"] = i
    return jobs


def _init_pairs():
    W._boot()
    _G["games"] = W.load_all()
    _G["ba"] = {g["seed"]: g for g in BA.read_jsonl(os.path.join(BA.OUT_DIR, "games.jsonl"))}


def _replay(j):
    k = tuple(j["g"])
    if k[0] == "ba":
        return BA.replay(_G["ba"][k[1]], j["step"])
    return W.replay({"g": list(k), "step": j["step"]}, _G["games"][k])


def after_move(game, seat, mv, ws, k6=None):
    """手を当て、続きの自分の選択と窓を片付けた盤面（決着したら None）。`k6` はロックス海賊団の対象を 6キングに固定。"""
    g = game.fork()
    g.apply_search_move(seat, json.dumps(mv, ensure_ascii=False))
    if k6:
        pend = json.loads(g.pending_json())
        if pend.get("player_id") == seat and pend.get("action") == "SEARCH_AND_SELECT" and "ロックス海賊団" in (pend.get("message") or ""):
            BA.apply(g, seat, {"kind": "game", "action_type": "RESOLVE_EFFECT_SELECTION",
                               "payload": {"selected_uuids": [k6], "index": 0, "accepted": True, "position": "BOTTOM", "declared_value": 0}})
    W.settle(g, seat, ws)
    return None if g.winner() is not None else g


def encode(g, who):
    enc = json.loads(g.encode(who))
    ci = np.zeros(N_TOK, np.int16)
    src = np.asarray(enc["card_idx"], np.int64)[:N_TOK]
    ci[:len(src)] = src
    return (np.asarray(enc["scalars"], np.float16), ci,
            np.asarray(enc["tokens"], np.float16).reshape(N_TOK, -1))


def pair_job(j):
    row = {"jid": j["jid"], "src": j["src"], "ok": False}
    try:
        game = _replay(j)
        pend = json.loads(game.pending_json())
        if pend.get("player_id") != j["seat"]:
            row["fail"] = "seat"
            return row
        ws = MJ.world_seed(sum(map(ord, json.dumps(j["g"]))), j["step"], OFFSET)
        ga = after_move(game, j["seat"], j["a"], ws)
        gb = after_move(game, j["seat"], j["b"], ws, j.get("k6"))
        if ga is None or gb is None:
            row["fail"] = "ended"
            return row
        row["a"] = encode(ga, j["seat"])
        row["b"] = encode(gb, j["seat"])
        row["ok"] = True
    except Exception as exc:  # noqa: BLE001
        row["fail"] = "%s: %s" % (type(exc).__name__, str(exc)[:80])
    return row


def cmd_pairs(a):
    jobs = collect_jobs()
    bench = bench_games()
    n_all = len(jobs)
    leak = [j for j in jobs if tuple(j["g"]) in bench]
    jobs = [j for j in jobs if tuple(j["g"]) not in bench]
    si, sn = KR.shard_of(a)
    jobs = [j for j in jobs if j["jid"] % sn == si]
    os.makedirs(a.out, exist_ok=True)
    print(f"PAIRS shard={a.shard} jobs {len(jobs)}（全 {n_all}・試験台の局で外した {len(leak)}）", flush=True)
    t0 = time.time()
    keep, fails = [], {}
    with get_context("fork").Pool(a.workers, initializer=_init_pairs) as pool:
        for i, r in enumerate(pool.imap_unordered(pair_job, jobs, chunksize=4)):
            if r["ok"]:
                keep.append(r)
            else:
                k = (r["src"], (r.get("fail") or "")[:30])
                fails[k] = fails.get(k, 0) + 1
            if (i + 1) % 50 == 0:
                print(f"[{i + 1}/{len(jobs)}] ok {len(keep)} ({time.time() - t0:.0f}s)", flush=True)
    byj = {j["jid"]: j for j in jobs}
    keep.sort(key=lambda r: r["jid"])
    js = [byj[r["jid"]] for r in keep]
    out = {"sc_a": np.array([r["a"][0] for r in keep]), "ci_a": np.array([r["a"][1] for r in keep]),
           "tok_a": np.array([r["a"][2] for r in keep]),
           "sc_b": np.array([r["b"][0] for r in keep]), "ci_b": np.array([r["b"][1] for r in keep]),
           "tok_b": np.array([r["b"][2] for r in keep]),
           "t_a": np.array([2.0 * j["wa"] - 1.0 for j in js], np.float32),
           "t_b": np.array([2.0 * j["wb"] - 1.0 for j in js], np.float32),
           "game": np.array([json.dumps(j["g"]) for j in js]), "src": np.array([j["src"] for j in js]),
           "jid": np.array([j["jid"] for j in js]), "n": np.array([j["n"] for j in js])}
    path = os.path.join(a.out, f"pairs_shard{si}.npz" if a.shard else "pairs.npz")
    np.savez_compressed(path, **out)
    meta = {"shard": a.shard, "jobs": len(jobs), "ok": len(keep), "all": n_all, "leak_excluded": len(leak),
            "fails": {f"{k[0]}|{k[1]}": v for k, v in sorted(fails.items())},
            "by_src": {s: int((out["src"] == s).sum()) for s in sorted(set(out["src"].tolist()))}}
    json.dump(meta, open(path[:-4] + ".json", "w"), ensure_ascii=False, indent=1)
    print("PAIRS_DONE", json.dumps(meta, ensure_ascii=False), flush=True)


def cmd_merge(a):
    parts = [dict(np.load(p, allow_pickle=True)) for p in sorted(glob.glob(os.path.join(a.out, "pairs_shard*.npz")))]
    out = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
    np.savez_compressed(os.path.join(a.out, "pairs.npz"), **out)
    d = out["t_b"] - out["t_a"]
    print("MERGED", len(d), {s: int((out["src"] == s).sum()) for s in sorted(set(out["src"].tolist()))},
          "mean |t_b−t_a|", round(float(np.abs(d).mean()), 3))


def bench_jobs():
    """試験台の点を対の形に（紫カイドウ 24 点＝a 使う・b 控える／6キングだけ 53 点＝a 訪問最多・b 戻す）。"""
    jobs = []
    for p in BA.read_jsonl(os.path.join(DRd.OUT_DIR, "points.jsonl")):
        if not p["judged"] or p["item"] == "mihawk_leader":
            continue
        dec = p["b"] if p["why"] in ("early", "big_next") else p["a"]
        acc = p["a"] if p["why"] in ("early", "big_next") else p["b"]
        wd = p["wb"] if p["why"] in ("early", "big_next") else p["wa"]
        wu = p["wa"] if p["why"] in ("early", "big_next") else p["wb"]
        jobs.append(dict(src="kaido24", g=list(kaido_key(p)), seat="p1", step=p["step"], a=acc, b=dec, wa=wu, wb=wd,
                         n=16, pid=p["pid"], why=p["why"]))
    for r in KR.read_glob("n1_results/rocks_bounce/judge*.jsonl", lambda x: (x["src"], x["idx"], x["step"])).values():
        if r.get("ok") and r["group"] == "k6_only":
            jobs.append(dict(src="rocks53", g=list(rocks_key(r)), seat="p2", step=r["step"], a=r["a"], b=r["b"],
                             wa=r["ref"]["wa"], wb=r["ref"]["wb"], n=16, k6=r["k6_uuid"], ot=r["ot"], chosen_b=r["chosen_b"]))
    for i, j in enumerate(jobs):
        j["jid"] = i
    return jobs


def cmd_bench(a):
    jobs = bench_jobs()
    with get_context("fork").Pool(a.workers, initializer=_init_pairs) as pool:
        rows = list(pool.imap_unordered(pair_job, jobs, chunksize=1))
    byj = {j["jid"]: j for j in jobs}
    for src in ("kaido24", "rocks53"):
        keep = sorted([r for r in rows if r["ok"] and r["src"] == src], key=lambda r: r["jid"])
        js = [byj[r["jid"]] for r in keep]
        out = {"sc_a": np.array([r["a"][0] for r in keep]), "ci_a": np.array([r["a"][1] for r in keep]),
               "tok_a": np.array([r["a"][2] for r in keep]),
               "sc_b": np.array([r["b"][0] for r in keep]), "ci_b": np.array([r["b"][1] for r in keep]),
               "tok_b": np.array([r["b"][2] for r in keep]),
               "t_a": np.array([2.0 * j["wa"] - 1.0 for j in js], np.float32),
               "t_b": np.array([2.0 * j["wb"] - 1.0 for j in js], np.float32),
               "game": np.array([json.dumps(j["g"]) for j in js]), "src": np.array([j["src"] for j in js]),
               "jid": np.array([j["jid"] for j in js]), "n": np.array([j["n"] for j in js]),
               "info": np.array([json.dumps({k: j.get(k) for k in ("pid", "why", "ot", "chosen_b", "step")}) for j in js])}
        os.makedirs(a.out, exist_ok=True)
        np.savez_compressed(os.path.join(a.out, f"bench_{src}.npz"), **out)
        fails = [r.get("fail") for r in rows if not r["ok"] and r["src"] == src]
        print("BENCH", src, "ok", len(keep), "fail", fails, flush=True)


def net_values(path, sc, ci, tok, bs=256):
    """npz のネットの生の価値（numpy の forward・`--ablate` は npz の meta のまま）。"""
    from opcg_sim.learned import n_rel as NL
    from opcg_sim.learned import n_rel_feat as NR
    from opcg_sim.learned.train import n_rel_train as NT
    from opcg_sim.learned.train.n_eff_feat import build_eff_tables
    from opcg_sim.learned.vocab import load_db
    if "tables" not in _G:
        stats, ab, abm, pwr, isl, vocab = build_eff_tables()
        _G["tables"] = (stats, ab, abm, pwr, isl)
        _G["rt"] = NR.RelTable(NR.profile_table(load_db(), vocab))
    net = NL.NRelNet.load(path, tables=_G["tables"])
    out = []
    for s in range(0, len(sc), bs):
        a_, c_, t_ = sc[s:s + bs].astype(np.float32), ci[s:s + bs].astype(np.int64), tok[s:s + bs].astype(np.float32)
        out.append(net.value(a_, c_, t_, *NT.relations_or_zeros(net, c_, t_, _G["rt"])))
    return np.concatenate(out)


def cmd_netval(a):
    """試験台の対で、ネットの値の差（b − a）と打ち続けた差（価値の目盛り）を比べる（探索なし）。"""
    res = {}
    for src in ("kaido24", "rocks53"):
        d = dict(np.load(os.path.join(a.out, f"bench_{src}.npz"), allow_pickle=True))
        dt = d["t_b"] - d["t_a"]
        res[src] = {"n": int(len(dt)), "truth_mean": round(float(dt.mean()), 4), "nets": {}}
        for net in a.nets:
            dv = net_values(net, d["sc_b"], d["ci_b"], d["tok_b"]) - net_values(net, d["sc_a"], d["ci_a"], d["tok_a"])
            res[src]["nets"][os.path.basename(net)] = {
                "dv_mean": round(float(dv.mean()), 4), "ratio": round(float(dv.mean() / dt.mean()), 3) if dt.mean() else None,
                "corr": round(float(np.corrcoef(dv, dt)[0, 1]), 3), "b_higher": int((dv > 0).sum()),
                "dv": [round(float(x), 4) for x in dv]}
            print(src, os.path.basename(net), "ネットの差", res[src]["nets"][os.path.basename(net)]["dv_mean"],
                  "打ち続けた差", res[src]["truth_mean"], "相関", res[src]["nets"][os.path.basename(net)]["corr"],
                  "b を高く見る点", int((dv > 0).sum()), "/", len(dv), flush=True)
    # 案 B の対の holdout（訓練に入れていない局・7 分の 1）と、源ごとの当てはまり
    from opcg_sim.learned.train import value_teach as VTT
    PRS = VTT.load_pairs([os.path.join(a.out, "pairs.npz")])
    res["pairs"] = {}
    for net in a.nets:
        dv = net_values(net, PRS["sc_b"], PRS["ci_b"], PRS["tok_b"]) - net_values(net, PRS["sc_a"], PRS["ci_a"], PRS["tok_a"])
        dt = PRS["t_b"] - PRS["t_a"]
        r_ = {}
        for nm, m in (("hold", PRS["hold"]), ("train", ~PRS["hold"])):
            r_[nm] = {"n": int(m.sum()), "corr": round(float(np.corrcoef(dv[m], dt[m])[0, 1]), 3),
                      "slope": round(float(np.dot(dv[m], dt[m]) / np.dot(dt[m], dt[m])), 3),
                      "d_mse": round(float(np.mean((dv[m] - dt[m]) ** 2)), 4),
                      "sign_agree": round(float(np.mean(np.sign(dv[m][dt[m] != 0]) == np.sign(dt[m][dt[m] != 0]))), 3)}
        res["pairs"][os.path.basename(net)] = r_
        print("pairs", os.path.basename(net), json.dumps(r_), flush=True)
    json.dump(res, open(os.path.join(a.out, f"netval_{a.tag}.json"), "w"), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- 1b. 時点ごとのドンの値（試験台 1 の割り引き）

def tvs_point(p):
    """`offdist_value.tv_point` と同じ 4 時点の盤面（盤面を作るのは常に r27＝ネットに依らない）で、
    ドン 1 枚違いの対（そのまま・1 枚少ない）の符号化を残す（ネットの値は後で `tvval` が付ける）。"""
    import offdist_value as OV
    path = os.path.join(_G["tv_dir"], f"pid{p['pid']:03d}.npz")
    if os.path.exists(path):
        return {"pid": p["pid"], "ok": True, "skip": True}
    who = "p1"
    rows = {"t": [], "base": [], "less": []}

    def add(t, g, prefer):
        b, l = OV.don_minus_one(g, who, prefer)
        if b is not None:
            rows["t"].append(t); rows["base"].append(encode(b, who)); rows["less"].append(encode(l, who))

    game, g = OV.replay_point(p)
    ga = OV.own_turn_head(p, g)
    if ga is not None:
        add("a_head", ga, "active")
    seed = DRd.game_seed(p)
    dec = OV.decline_move(p)
    t_now = int(game.turn_count or 0)
    target = t_now + (2 if game.turn_player() == who else 1)
    errs = 0
    for i in range(_G["k"]):
        ws = MJ.world_seed(seed, p["step"], OV.OFFSET_TV + i)
        try:
            g1 = game.fork_world(who, "fair", int(ws))
            g1.apply_search_move(who, json.dumps(dec, ensure_ascii=False))
            W.settle(g1, who, ws)
            add("b_close", g1, "rested")
            gc = OV.play_to(g1.fork(), who, ws, lambda gg, pend, mv: mv is not None and pend["player_id"] != who
                            and mv.get("action_type") == "TURN_END")
            if gc.winner() is None:
                add("c_oppend", gc, "rested")
            gd = OV.play_to(g1.fork(), who, ws, lambda gg, pend, mv: mv is None and int(gg.turn_count or 0) >= target
                            and pend["player_id"] == who and gg.turn_player() == who and not gg.has_interaction)
            if gd.winner() is None:
                add("d_next", gd, "active")
        except Exception:  # noqa: BLE001
            errs += 1
    out = {"t": np.array(rows["t"])}
    for nm in ("base", "less"):
        out["sc_" + nm] = np.array([x[0] for x in rows[nm]]); out["ci_" + nm] = np.array([x[1] for x in rows[nm]])
        out["tok_" + nm] = np.array([x[2] for x in rows[nm]])
    np.savez_compressed(path, **out)
    return {"pid": p["pid"], "ok": True, "n": len(rows["t"]), "errs": errs}


def _init_tvs(k, tv_dir):
    import offdist_value as OV
    OV._init_tv_k(k, 0, None)
    _G["k"], _G["tv_dir"] = int(k), tv_dir


def cmd_tvstates(a):
    import offdist_value as OV
    pts = OV.kaido_points()
    tv_dir = os.path.join(a.out, "tv_states")
    os.makedirs(tv_dir, exist_ok=True)
    print(f"TVSTATES todo {len(pts)} k={a.k} -> {tv_dir}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init_tvs, initargs=(a.k, tv_dir)) as pool:
        for i, r in enumerate(pool.imap_unordered(tvs_point, pts, chunksize=1)):
            print(f"[{i + 1}/{len(pts)}] {json.dumps(r)} ({time.time() - t0:.0f}s)", flush=True)
    print("TVSTATES_DONE", flush=True)


#: 打ち続けた差（窓を閉じた直後のドン 1 枚・価値の目盛り・24 点の平均・§7.69 の報告と RESULT.json）
TV_TRUTH = 0.338
TV_TIMES = ("a_head", "b_close", "c_oppend", "d_next")


def cmd_tvval(a):
    files = sorted(glob.glob(os.path.join(a.out, "tv_states", "pid*.npz")))
    res = {"points": len(files), "truth": TV_TRUTH, "nets": {}}
    for net in a.nets:
        per = {t: [] for t in TV_TIMES}
        for f in files:
            d = dict(np.load(f, allow_pickle=True))
            if not len(d["t"]):
                continue
            dv = net_values(net, d["sc_base"], d["ci_base"], d["tok_base"]) - net_values(net, d["sc_less"], d["ci_less"], d["tok_less"])
            for t in TV_TIMES:
                m = d["t"] == t
                if m.any():
                    per[t].append(float(dv[m].mean()))
        row = {t: {"mean": round(float(np.mean(v)), 4), "ratio": round(float(np.mean(v)) / TV_TRUTH, 3), "n": len(v)}
               for t, v in per.items() if v}
        res["nets"][os.path.basename(net)] = row
        print(os.path.basename(net), " ".join(f"{t} {x['mean']:.3f}({x['ratio']:.2f})" for t, x in row.items()), flush=True)
    json.dump(res, open(os.path.join(a.out, f"tvval_{a.tag}.json"), "w"), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- 2. rocks（試験台 2）

def _init_rocks(nets):
    E.engine()
    for n in nets:
        E.load_net(n)
    W._boot()
    _G["games"] = W.load_all()
    _G["nets"] = nets


def rocks_point(p):
    row = {"src": p["src"], "idx": p["idx"], "step": p["step"], "ot": p["ot"], "ok": False, "nets": {}}
    try:
        game = W.replay({"g": list(rocks_key(p)), "step": p["step"]}, _G["games"][rocks_key(p)])
    except Exception as exc:  # noqa: BLE001
        row["fail"] = str(exc)[:80]
        return row
    turn = int(game.turn_count or 0)
    for net in _G["nets"]:
        spec = BA.ship_spec(net)
        out = json.loads(game.fork().decide("p2", spec.decide_opts(int(p["seed"]), turn, "p2", DR.Carry().get(turn, "p2"))))
        rows = BA.ranked(out)
        hit = [r for r in rows if BA.pkey(r["mv"]) == BA.pkey(p["b"])]
        top = rows[0] if rows else None
        chose_b = BA.pkey(out.get("move")) == BA.pkey(p["b"])
        row["nets"][net] = {"chose_b": chose_b, "q_top": top["q"] if top else None, "q_b": hit[0]["q"] if hit else None,
                            "top_is_b": bool(top and BA.pkey(top["mv"]) == BA.pkey(p["b"])),
                            "share_b": (hit[0]["n"] / max(sum(r["n"] for r in rows), 1.0)) if hit else 0.0,
                            "top": BA.desc(game, top["mv"]) if top else None}
    row["ok"] = True
    return row


def cmd_rocks(a):
    pts = rocks_points()
    with get_context("fork").Pool(a.workers, initializer=_init_rocks, initargs=(a.nets,)) as pool:
        rows = list(pool.imap_unordered(rocks_point, pts, chunksize=1))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"rocks_{a.tag}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print_rocks(rows, a.nets)


def print_rocks(rows, nets):
    ok = [r for r in rows if r["ok"]]
    for net in nets:
        xs = [r["nets"][net] for r in ok if net in r["nets"]]
        d = [x["q_b"] - x["q_top"] for x in xs if x["q_b"] is not None and x["q_top"] is not None]
        print(f"{os.path.basename(net)} 点 {len(xs)} 戻すを選ぶ {sum(x['chose_b'] for x in xs)} "
              f"根の評価の差（戻す−訪問最多）{np.mean(d):+.4f}（{len(d)} 点）戻すの訪問の割合 {np.mean([x['share_b'] for x in xs]):.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["pairs", "merge", "bench", "netval", "rocks", "tvstates", "tvval"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--nets", nargs="*", default=[KR.NET])
    ap.add_argument("--tag", default="r27")
    a = ap.parse_args(argv)
    {"pairs": cmd_pairs, "merge": cmd_merge, "bench": cmd_bench, "netval": cmd_netval, "rocks": cmd_rocks,
     "tvstates": cmd_tvstates, "tvval": cmd_tvval}[a.cmd](a)


if __name__ == "__main__":
    main()
