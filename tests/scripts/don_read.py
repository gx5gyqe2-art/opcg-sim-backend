"""ドンを減らす効果を「使うか・控えるか」の読みの診断（WP `don-read` 段 1・`docs/search_efficiency_plan.md` §7.66・台帳 2+475）。

読み取りだけの計測（エンジン・ネット・既定の挙動は変えない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。
試験台は `matchup_kr`（§7.63）と `matchup_5`（§7.65）の打ち比べの点（差は `move_judge.paired` の b − a＝記事の手 − CPU の手）と、
同じ型の打ち比べていない点（紫カイドウのリーダー効果を 6キングの前に使った点）。

各点・両方の手（a＝CPU の手・b＝記事の手）について、配り直した世界 K 本で:

- **今の読み**: 出荷の設定の決め方そのもの（窓の根畳み〔16 本〕・メインの点は根の木の値）。
- **短い木**: 手を当てて窓を閉じた盤面（`Game.resolve_windows`）から、その世界のまま（引き直さない）木を回した根の値
  （§2.5.12 の 3 と同じ形・回数は出荷の 1,280 ÷ 候補 2 ＝ 640）。
- **次のターンの頭**: 手を当て、今のターンの残りと相手のターンを両席とも出荷の設定で打ち、自分の次のターンの最初の判断の盤面で
  ネットの値（生の値）と審判（160 回）の根の値。終局なら ±1。
- **ドン 1 枚の値**: 「次のターンの頭」の盤面から自分のアクティブのドンを 1 枚ドンデッキへ戻した盤面を作り、ネットの値と審判の値の差。

    python tests/scripts/don_read.py points
    python tests/scripts/don_read.py read --workers 4 [--shard i/n] [--k 16]
    python tests/scripts/don_read.py summary

再開できる（終えた点は飛ばす）。
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blunder_audit as BA  # noqa: E402
import matchup_5 as M5  # noqa: E402
import matchup_kr as KR  # noqa: E402
import move_judge as MJ  # noqa: E402

from opcg_sim.loop import driver as DR, engine as E  # noqa: E402

OUT_DIR = os.path.join("n1_results", "don_read")
NET = KR.NET
#: 配り方の番号のずらし（matchup_kr 7000・matchup_5 8000 と重ねない）。
OFFSET = 9000
#: 短い木の回数（出荷の 1,280 ÷ 候補 2＝§2.5.12 の 3 と同じ割り方）。
TREE_SIMS = E.PLAY_SIMS // 2
#: 打ち比べていない点（6キングの前に使った点）の数の上限（コーパスごと）。
N_EXTRA = 20
MAX_STEPS = 300
_G = {}


# ---------------------------------------------------------------- 点

def _m5_games():
    rows = {}
    for path in sorted(glob.glob(os.path.join(M5.OUT_DIR, "games*.jsonl"))):
        for g in BA.read_jsonl(path):
            rows[(g["mu"], g["idx"])] = g
    return rows


def game_seed(p):
    return (KR.SEED0 + p["idx"]) if p["src"] == "kr" else (M5.MU[p["mu"]][2] + p["idx"])


def replay(p, g):
    return KR.replay(g, p["step"]) if p["src"] == "kr" else M5.replay(g, p["step"])


def _judged(src):
    d = KR.OUT_DIR if src == "kr" else M5.OUT_DIR
    out = []
    for path in sorted(glob.glob(os.path.join(d, "judge_points_shard*.jsonl"))):
        for x in BA.read_jsonl(path):
            if x["item"] not in ("5", "kr_leader", "mihawk_leader") or not x.get("ok"):
                continue
            r = x["ref"]
            out.append({"src": src, "mu": x.get("mu") or "kaido_rocks_official", "idx": x["idx"], "step": x["step"],
                        "item": x["item"], "why": x["why"], "turn": x["turn"], "own_turn": x.get("own_turn"),
                        "a": x["a"], "b": x["b"], "a_desc": x["a_desc"], "b_desc": x["b_desc"],
                        "judged": True, "diff": r["diff"], "ci": r["ci"], "wa": r["wa"], "wb": r["wb"]})
    return out


def _extra(src, games, judged_keys):
    """打ち比べていない「6キングの前に使った」点（記事は控える・勝負の決まった局面は外す）。"""
    d = KR.OUT_DIR if src == "kr" else M5.OUT_DIR
    item = "5" if src == "kr" else "kr_leader"
    rows = []
    for x in BA.read_jsonl(os.path.join(d, "cands_all.jsonl")):
        if x["item"] != item or x["why"] != "early" or x["qa"] is None or abs(x["qa"]) >= 0.9:
            continue
        mu = x.get("mu") or "kaido_rocks_official"
        if (src, mu, x["idx"], x["step"]) in judged_keys:
            continue
        g = games.get(x["idx"] if src == "kr" else (mu, x["idx"]))
        if g is None:
            continue
        pt = g["points"][x["step"]]
        a = g["moves"][x["step"]]
        others = [t["mv"] for t in pt["top"] if BA.key(t["mv"]) != BA.key(a)]
        if not others or not (a.get("payload") or {}).get("accepted", False):
            continue
        rows.append({"src": src, "mu": mu, "idx": x["idx"], "step": x["step"], "item": item, "why": "early",
                     "turn": x["turn"], "own_turn": x.get("own_turn"), "a": a, "b": others[0],
                     "judged": False, "qa": x["qa"]})
    rows.sort(key=lambda r: E.search_seed(int(r["idx"]) ^ 0x0D0E_EAD0, int(r["step"]), r["mu"]))
    return rows[:N_EXTRA]


def cmd_points(a):
    os.makedirs(a.out, exist_ok=True)
    pts = []
    for src in ("kr", "m5"):
        games = KR.load_games(KR.OUT_DIR) if src == "kr" else _m5_games()
        j = _judged(src)
        keys = {(p["src"], p["mu"], p["idx"], p["step"]) for p in j}
        pts += j + _extra(src, games, keys)
    for n, p in enumerate(pts):
        p["pid"] = n
    with open(os.path.join(a.out, "points.jsonl"), "w", encoding="utf-8") as f:
        for p in pts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("POINTS", len(pts), "judged", sum(p["judged"] for p in pts),
          {k: sum(1 for p in pts if p["item"] == k) for k in ("5", "kr_leader", "mihawk_leader")}, flush=True)


# ---------------------------------------------------------------- 読み

def _init(k):
    E.engine()
    _G["ship"] = BA.ship_spec(NET)
    _G["ref"] = E.SeatSpec(NET, sims=MJ.REFEREE_SIMS)
    _G["k"] = int(k)
    _G["games"] = {"kr": KR.load_games(KR.OUT_DIR), "m5": _m5_games()}


def net_value(g, seat="p1"):
    enc = json.loads(g.encode(seat))
    enc["tok"] = enc.pop("tokens", None) or []
    return float(json.loads(E.engine().net_eval(json.dumps(enc), "[]"))["value"])


def root_value(out, mover):
    st = out.get("stats") or {}
    ns, qs = st.get("N") or [], st.get("Q") or []
    tot = sum(ns)
    if not tot:
        return None
    v = sum(n * q for n, q in zip(ns, qs)) / tot
    return v if mover == "p1" else -v


def terminal(g):
    w = g.winner()
    return None if w is None else (1.0 if w == "p1" else -1.0)


def short_tree(g1, ws):
    """窓を閉じた盤面から、その世界のまま木を回した根の値（p1 から見た）。"""
    g = g1.fork()
    ship = _G["ship"]
    omni = json.dumps(dict(ship.opts, search_seed=int(ws), omniscient=True))
    # 手の続きの自分の選択（対象など）は窓の根畳みと同じ決め方で（その世界のまま）選ぶ。
    for _ in range(6):
        pend = json.loads(g.pending_json())
        if g.winner() is not None or not pend or pend["player_id"] != "p1" or not (g.has_interaction or g.phase != "MAIN"):
            break
        out = json.loads(g.decide("p1", omni))
        if out.get("kind") != "window" or out.get("move") is None:
            break
        BA.apply(g, "p1", out["move"])
    info = json.loads(g.resolve_windows(json.dumps(dict(ship.opts, search_seed=int(ws)))))
    t = terminal(g)
    if t is not None:
        return {"v": t, "net": t, "open": False}
    pend = json.loads(g.pending_json())
    if not pend:
        return {"v": None, "net": None, "open": info["open"]}
    mover = pend["player_id"]
    opts = {"net": ship.net, "sims": TREE_SIMS, "omniscient": True, "search_seed": int(ws) ^ 0x7EE5}
    out = json.loads(g.decide(mover, json.dumps(opts)))
    v = root_value(out, mover)
    return {"v": v, "net": net_value(g), "open": info["open"], "kind": out.get("kind"), "mover": mover,
            "don_pair": don_pair(g, ws, with_ref=False)}


def don_counts(g, seat="p1"):
    h = json.loads(g.hidden_json())
    d = h["players"][seat]["don"]
    return {"active": len(d["active"]), "rested": len(d["rested"]), "attached": len(d["attached"]), "deck": len(d["deck"])}


def don_pair(g, ws, with_ref=True):
    """同じ盤面で自分の場のドン（アクティブ優先・無ければレスト）を 1 枚ドンデッキへ戻した対
    （ネットの値・審判の値の差＝元 − 戻した方）。"""
    if g.has_interaction:
        return None
    h = json.loads(g.hidden_json())
    d = h["players"]["p1"]["don"]
    zone = "active" if d["active"] else ("rested" if d["rested"] else None)
    if zone is None:
        return None
    eng = E.engine()
    try:
        base = eng.Game.from_hidden(json.dumps(h))
        x = d[zone].pop()
        x["is_rest"] = False
        d["deck"].append(x)
        less = eng.Game.from_hidden(json.dumps(h))
        vb, vl = net_value(base), net_value(less)
        if not with_ref:
            return {"net_same": round(net_value(g) - vb, 6), "net": vb - vl, "net_base": vb, "zone": zone}
        ref = _G["ref"]
        turn = int(g.turn_count or 0)
        rb = root_value(json.loads(base.decide("p1", ref.decide_opts(int(ws), turn, "p1"))), "p1")
        rl = root_value(json.loads(less.decide("p1", ref.decide_opts(int(ws), turn, "p1"))), "p1")
    except Exception as exc:  # noqa: BLE001
        return {"err": "%s: %s" % (type(exc).__name__, str(exc)[:100])}
    return {"net_same": round(net_value(g) - vb, 6), "net": vb - vl, "ref": None if rb is None or rl is None else rb - rl,
            "net_base": vb, "ref_base": rb}


def next_turn_head(g1, ws):
    """今のターンの残りと相手のターンを両席とも出荷の設定で打ち、自分の次のターンの最初の判断の盤面へ。"""
    g = g1.fork()
    ship = _G["ship"]
    t0 = int(g.turn_count or 0)
    target = t0 + (2 if g.turn_player() == "p1" else 1)
    carries = {"p1": DR.Carry(), "p2": DR.Carry()}
    steps = 0
    while g.winner() is None and steps < MAX_STEPS:
        pend = json.loads(g.pending_json())
        if not pend:
            break
        name = pend["player_id"]
        turn = int(g.turn_count or 0)
        if turn >= target and name == "p1" and g.turn_player() == "p1" and not g.has_interaction:
            break
        c = carries[name]
        out = json.loads(g.decide(name, ship.decide_opts(int(ws), turn, name, c.get(turn, name))))
        c.put(out)
        mv = out.get("move")
        if mv is None:
            break
        BA.apply(g, name, mv)
        steps += 1
    t = terminal(g)
    if t is not None:
        return {"v": t, "net": t, "end": True, "steps": steps}, None
    ref = _G["ref"]
    turn = int(g.turn_count or 0)
    out = json.loads(g.fork().decide("p1", ref.decide_opts(int(ws), turn, "p1")))
    row = {"v": root_value(out, "p1"), "net": net_value(g), "end": False, "steps": steps, "turn": turn,
           "don": don_counts(g), "hand": len(json.loads(g.hidden_json())["players"]["p1"]["hand"])}
    return row, g


def read_point(p):
    t0 = time.time()
    row = {"pid": p["pid"], "ok": False}
    try:
        game = replay(p, _G["games"][p["src"]][p["idx"] if p["src"] == "kr" else (p["mu"], p["idx"])])
    except Exception as exc:  # noqa: BLE001
        row["fail"] = "replay: %s" % type(exc).__name__
        return row
    seed = game_seed(p)
    ship = _G["ship"]
    turn = int(game.turn_count or 0)
    out = json.loads(game.fork().decide("p1", ship.decide_opts(seed, turn, "p1", DR.Carry().get(turn, "p1"))))
    rows = BA.ranked(out)
    cur = {}
    for nm in ("a", "b"):
        hit = [r for r in rows if BA.key(r["mv"]) == BA.key(p[nm])] or [r for r in rows if BA.pkey(r["mv"]) == BA.pkey(p[nm])]
        cur[nm] = {"n": hit[0]["n"], "q": hit[0]["q"]} if hit else None
    row["now"] = {"kind": out.get("kind"), "a": cur["a"], "b": cur["b"], "net0": net_value(game),
                  "don0": don_counts(game), "turn_player": game.turn_player()}
    per = {"a": [], "b": []}
    fold = []
    for i in range(_G["k"]):
        ws = MJ.world_seed(seed, p["step"], OFFSET + i)
        if out.get("kind") == "window":
            try:
                gw = game.fork_world("p1", "fair", int(ws))
                ow = json.loads(gw.decide("p1", json.dumps(dict(ship.opts, search_seed=int(ws), omniscient=True))))
                rw = BA.ranked(ow)
                fold.append({nm: next((r["q"] for r in rw if BA.key(r["mv"]) == BA.key(p[nm])), None) for nm in ("a", "b")})
            except Exception as exc:  # noqa: BLE001
                fold.append({"err": str(exc)[:80]})
        for nm in ("a", "b"):
            try:
                g1 = game.fork_world("p1", "fair", int(ws))
                g1.apply_search_move("p1", json.dumps(p[nm], ensure_ascii=False))
                tr = short_tree(g1, ws)
                nx, gh = next_turn_head(g1, ws)
                dp = don_pair(gh, ws) if (gh is not None and nm == "b") else None
                per[nm].append({"tree": tr, "next": nx, "don_pair": dp})
            except Exception as exc:  # noqa: BLE001
                per[nm].append({"err": "%s: %s" % (type(exc).__name__, str(exc)[:120])})
    row.update(per=per, fold=fold, ok=True, sec=round(time.time() - t0, 1))
    return row


def cmd_read(a):
    pts = BA.read_jsonl(os.path.join(a.out, "points.jsonl"))
    si, sn = KR.shard_of(a)
    jobs = [p for p in pts if p["pid"] % sn == si]
    if a.only_judged:
        jobs = [p for p in jobs if p["judged"]]
    if a.limit:
        jobs = jobs[:a.limit]
    rpath = os.path.join(a.out, f"reads_shard{si}.jsonl" if a.shard else "reads.jsonl")
    done = set()
    for path in glob.glob(os.path.join(a.out, "reads*.jsonl")):
        done |= {x["pid"] for x in BA.read_jsonl(path) if x.get("ok")}
    jobs = [p for p in jobs if p["pid"] not in done]
    print(f"READ shard={a.shard} todo {len(jobs)} (done {len(done)}) k={a.k} -> {rpath}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.k,)) as pool:
        for i, r in enumerate(pool.imap_unordered(read_point, jobs, chunksize=1)):
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(jobs)}] pid={r['pid']} ok={r['ok']} sec={r.get('sec')} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- 集計

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _ci(ds):
    ds = [d for d in ds if d is not None]
    n = len(ds)
    if n < 2:
        return None, [None, None], n
    m = sum(ds) / n
    sd = (sum((d - m) ** 2 for d in ds) / (n - 1)) ** 0.5
    h = 1.96 * sd / n ** 0.5
    return m, [m - h, m + h], n


def _pair_diff(per, part, key):
    """世界ごとの b − a（両方値が出た世界だけ）。"""
    ds = []
    for x, y in zip(per["a"], per["b"]):
        va = (x.get(part) or {}).get(key) if "err" not in x else None
        vb = (y.get(part) or {}).get(key) if "err" not in y else None
        if va is not None and vb is not None:
            ds.append(vb - va)
    return ds


def _r(x, n=3):
    return None if x is None else round(x, n)


def _corr(xs, ys):
    pr = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pr)
    if n < 3:
        return None
    mx, my = sum(x for x, _ in pr) / n, sum(y for _, y in pr) / n
    sxy = sum((x - mx) * (y - my) for x, y in pr)
    sxx = sum((x - mx) ** 2 for x, _ in pr)
    syy = sum((y - my) ** 2 for _, y in pr)
    return sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else None


def cmd_summary(a):
    pts = {p["pid"]: p for p in BA.read_jsonl(os.path.join(a.out, "points.jsonl"))}
    reads = {}
    for path in sorted(glob.glob(os.path.join(a.out, "reads*.jsonl"))):
        for x in BA.read_jsonl(path):
            if x.get("ok"):
                reads[x["pid"]] = x
    table = []
    for pid, r in sorted(reads.items()):
        p = pts[pid]
        now = r["now"]
        qa = (now.get("a") or {}).get("q")
        qb = (now.get("b") or {}).get("q")
        d_now = None if qa is None or qb is None else qb - qa
        row = {"pid": pid, "src": p["src"], "mu": p["mu"], "idx": p["idx"], "step": p["step"], "item": p["item"], "why": p["why"],
               "turn": p["turn"], "own_turn": p.get("own_turn"), "judged": p["judged"], "diff": p.get("diff"), "ci": p.get("ci"),
               "kind": now["kind"], "turn_player": now.get("turn_player"), "don0": now.get("don0"),
               "now": _r(d_now)}
        for part, key, nm in (("tree", "v", "tree"), ("tree", "net", "tree_net"), ("next", "v", "next"), ("next", "net", "next_net")):
            m, ci, n = _ci(_pair_diff(r["per"], part, key))
            row[nm] = _r(m)
            row[nm + "_ci"] = [_r(ci[0]), _r(ci[1])]
        # §2.5.12 の 3 の形（世界 2 本）
        ds2 = _pair_diff({"a": r["per"]["a"][:2], "b": r["per"]["b"][:2]}, "tree", "v")
        row["tree2"] = _r(_mean(ds2))
        # 次のターンの頭の場のドン（a と b）
        for nm in ("a", "b"):
            dons = [(x.get("next") or {}).get("don") for x in r["per"][nm] if "err" not in x]
            dons = [d["active"] + d["rested"] + d["attached"] for d in dons if d]
            row["don_next_" + nm] = _r(_mean(dons), 2)
        dps = [x.get("don_pair") for x in r["per"]["b"] if "err" not in x and x.get("don_pair") and "err" not in x["don_pair"]]
        row["don1_net"] = _r(_mean([d["net"] for d in dps]))
        row["don1_ref"] = _r(_mean([d["ref"] for d in dps]))
        row["don1_n"] = len(dps)
        # 窓を閉じた盤面（今の読みが値を取る所）でのドン 1 枚の値（ネットの値・a と b の両方の世界）
        tps = [(x.get("tree") or {}).get("don_pair") for nm in ("a", "b") for x in r["per"][nm] if "err" not in x]
        tps = [d for d in tps if d and "err" not in d]
        row["don1_close_net"] = _r(_mean([d["net"] for d in tps]))
        row["errs"] = sum(1 for nm in ("a", "b") for x in r["per"][nm] if "err" in x)
        table.append(row)

    def pick(d):
        return None if d is None else ("b" if d > 0 else ("a" if d < 0 else "tie"))

    def article_decline(row):
        # 記事の手（b）が「控える」なら b を選ぶ＝控える
        return row["item"] != "mihawk_leader" and row["why"] in ("early", "big_next")

    groups = {
        "judged_all": [t for t in table if t["judged"]],
        "judged_kaido_decline": [t for t in table if t["judged"] and article_decline(t)],
        "judged_kaido_early": [t for t in table if t["judged"] and t["why"] == "early" and t["item"] != "mihawk_leader"],
        "judged_kaido_use": [t for t in table if t["judged"] and t["why"] == "bk_on"],
        "judged_mihawk": [t for t in table if t["judged"] and t["item"] == "mihawk_leader"],
        "extra_early": [t for t in table if not t["judged"]],
        "early_all": [t for t in table if t["why"] == "early"],
    }
    agg = {}
    for gname, rows in groups.items():
        g = {"n": len(rows)}
        for read in ("now", "tree", "tree2", "next", "next_net", "tree_net"):
            vals = [t[read] for t in rows]
            g[read] = {"picks_article": sum(1 for v in vals if v is not None and v > 0),
                       "picks_cpu": sum(1 for v in vals if v is not None and v < 0),
                       "mean": _r(_mean(vals)),
                       "corr_with_diff": _r(_corr(vals, [t["diff"] for t in rows]))}
            # 読みが記事の手を選んだ点の打ち比べの差の平均・CPU の手を選んだ点の平均
            sel = [t["diff"] for t in rows if t[read] is not None and t[read] > 0 and t["diff"] is not None]
            oth = [t["diff"] for t in rows if t[read] is not None and t[read] <= 0 and t["diff"] is not None]
            g[read]["gain_if_followed"] = _r(sum(sel) / len(rows)) if rows and any(t["diff"] is not None for t in rows) else None
            g[read]["diff_when_article"] = _r(_mean(sel))
            g[read]["diff_when_cpu"] = _r(_mean(oth))
        g["diff_mean"] = _r(_mean([t["diff"] for t in rows]))
        g["don1_net"] = _r(_mean([t["don1_net"] for t in rows]))
        g["don1_ref"] = _r(_mean([t["don1_ref"] for t in rows]))
        g["don1_close_net"] = _r(_mean([t["don1_close_net"] for t in rows]))
        agg[gname] = g
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"groups": agg, "table": table}, f, ensure_ascii=False, indent=1)
    for gname, g in agg.items():
        print(gname, json.dumps(g, ensure_ascii=False))
    for t in table:
        print(json.dumps({k: t[k] for k in ("pid", "mu", "item", "why", "own_turn", "turn_player", "diff", "now", "tree", "tree2", "next",
                                             "next_net", "don_next_a", "don_next_b", "don1_net", "don1_ref", "don1_close_net", "errs")}, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["points", "read", "summary"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-judged", action="store_true")
    a = ap.parse_args(argv)
    {"points": cmd_points, "read": cmd_read, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
