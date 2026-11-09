"""窓の決め方全体に短い先読みを当てる——段 1 の診断（WP `window-read-all`・`docs/search_efficiency_plan.md` §7.67・台帳 2+477）。

読み取りだけの計測（エンジン・ネット・既定の挙動は変えない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。
`don_read.py`（§7.66 段 1）を窓の種類全体と両方の席に一般化したもの。

記録（両席 r27・出荷の設定）: `n1_results/hold_bias/games.jsonl`（random × real・150 局）・`matchup_5/games_shard*.jsonl`・`matchup_kr/games_shard*.jsonl`。
窓の種類（判断する側から見て）:

- ``counter``＝カウンター／``blocker``＝ブロック（どちらも自分のライフ 2 以上。1 以下は今も短い木なので ``battle_life1`` として数だけ）
- ``confirm_own``＝自分の手番の任意効果の確認／``confirm_opp``＝相手の手番の任意効果の確認（どちらもドンを代金にしないもの）
- ``don_confirm``＝ドンを代金にする確認（使うと自分の場のドンが減るかレストになる・§7.66 と同じ型）
- ``select``＝「最大 n 枚」などの選択（SEARCH_AND_SELECT）／``trigger``＝ライフのトリガーの確認

1. ``points``: 種類ごとに局の記録だけで決まる順位で点を抜く（候補 2 つ以上・勝負の決まった局面〔記録の評価の絶対値 0.9 以上〕は除く・
   hold_bias とマッチアップの記録を交互に）。``points.jsonl``。
2. ``eval``: 各点で (a) 今の決め方（出荷の設定の窓の根畳み）の手と値、(b) 全候補について、配り直した世界 4 本で窓を閉じた盤面から
   その世界のまま木を 160 回と 640 回回した値（世界 2 本の値も出す）、(c) 今の手と短い木の手が違えば両方を次の自分のターンの頭まで
   進めた値（世界 4 本）。1 点の費用（秒）も記録。``evals*.jsonl``・``--shard i/n``。
3. ``judge``: 短い木（640 回・世界 2 本＝§2.5.12 の 3 と同じ形）の手が今の手と違う点を、1 点 16 通りで打ち比べる
   （公平な配り直し・今のターンの残りは両席とも出荷の設定、その先は審判 r27・160 回＝§7.60 と同じ）。差は「短い木の手 − 今の手」。
   ``judge*.jsonl``・``--shard i/n``。
4. ``summary``: 種類ごとの表（手が変わる割合・打ち比べの平均と 95% 区間・相関・費用）。``summary.json``。
"""
import argparse
import glob
import json
import os
import sys
import time
import zlib
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blunder_audit as BA  # noqa: E402
import don_read as DRd  # noqa: E402
import matchup_5 as M5  # noqa: E402
import matchup_kr as KR  # noqa: E402
import move_judge as MJ  # noqa: E402

from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402

OUT_DIR = os.path.join("n1_results", "window_read_all")
NET = KR.NET
HB_GAMES = os.path.join("n1_results", "hold_bias", "games.jsonl")
#: 配り方の番号のずらし（hold_bias 3000・small_play 4000・matchup_kr 7000・matchup_5 8000・don_read 9000 と重ねない）。
OFFSET_READ = 10000
OFFSET_JUDGE = 11000
TYPES = ["counter", "blocker", "confirm_own", "confirm_opp", "don_confirm", "select", "trigger"]
N_PER_TYPE = 50
TREE_SIMS = (160, 640)
WORLDS = 4
NEXT_WORLDS = 4
DECIDED = 0.9
MAX_STEPS = 300
_G = {}


# ---------------------------------------------------------------- 記録

def load_all():
    games = {}
    for g in BA.read_jsonl(HB_GAMES):
        games[("hb", g["seed"])] = g
    for g in KR.load_games(KR.OUT_DIR).values():
        games[("kr", g["idx"])] = g
    for path in sorted(glob.glob(os.path.join(M5.OUT_DIR, "games_shard*.jsonl"))):
        for g in BA.read_jsonl(path):
            games[("m5", g["mu"], g["idx"])] = g
    return games


def gkey(p):
    return tuple(p["g"])


def game_seed(p):
    k = p["g"]
    if k[0] == "hb":
        return int(k[1])
    if k[0] == "kr":
        return KR.SEED0 + int(k[1])
    return M5.MU[k[1]][2] + int(k[2])


def replay(p, g):
    k = p["g"]
    if k[0] == "hb":
        return BA.replay(g, p["step"])
    if k[0] == "kr":
        return KR.replay(g, p["step"])
    return M5.replay(g, p["step"])


def _boot():
    E.engine()
    if "db" not in BA._G:
        BA._G["db"] = D.load_db()
    _G["ship"] = BA.ship_spec(NET)
    _G["ref"] = E.SeatSpec(NET, sims=MJ.REFEREE_SIMS)
    DRd._G["ship"], DRd._G["ref"] = _G["ship"], _G["ref"]


# ---------------------------------------------------------------- 1. points

def raw_type(p):
    """記録の点 → 種類の候補（再生前・ドンの代金とライフ 1 以下は再生してから分ける）。"""
    if p.get("kind") != "window" or int(p.get("n_cand") or 0) < 2:
        return None
    dt = p.get("dtype") or ""
    if dt in ("counter", "blocker"):
        return dt
    if dt.endswith("CONFIRM_OPTIONAL"):
        return "confirm_own" if dt.startswith("own") else "confirm_opp"
    if dt.endswith("SEARCH_AND_SELECT"):
        return "select"
    if dt.endswith("CONFIRM_TRIGGER"):
        return "trigger"
    return None


def chosen_q(p):
    mv = p.get("move")
    for t in p.get("top") or []:
        if BA.key(t["mv"]) == BA.key(mv):
            return t["q"]
    return (p.get("top") or [{}])[0].get("q")


def field_don(game, seat):
    d = json.loads(game.hidden_json())["players"][seat]["don"]
    return len(d["active"]) + len(d["rested"]) + len(d["attached"]), len(d["active"])


def is_don_cost(game, seat):
    """確認で「使う」を当てると自分の場のドンが減るかアクティブのドンが減るか。"""
    legal = json.loads(game.legal_json(seat))
    acc = [m for m in legal if (m.get("payload") or {}).get("accepted") is True]
    if not acc:
        return False
    tot0, act0 = field_don(game, seat)
    g = game.fork()
    try:
        BA.apply(g, seat, acc[0])
    except Exception:  # noqa: BLE001
        return False
    tot1, act1 = field_don(g, seat)
    return tot1 < tot0 or act1 < act0


def cmd_points(a):
    os.makedirs(a.out, exist_ok=True)
    _boot()
    games = load_all()
    cands = {t: {"hb": [], "mu": []} for t in TYPES + ["battle_life1"]}
    for k, g in games.items():
        for p in g["points"]:
            t = raw_type(p)
            if t is None:
                continue
            q = chosen_q(p)
            if q is None or abs(q) >= DECIDED:
                continue
            src = "hb" if k[0] == "hb" else "mu"
            r = E.search_seed(zlib.crc32(json.dumps(list(k)).encode()) ^ 0x77A1_D0A0, int(p["step"]), p["seat"])
            cands[t][src].append((r, k, p))
    for t in cands:
        for s in cands[t]:
            cands[t][s].sort(key=lambda x: x[0])
    counts = {t: {s: len(v) for s, v in d.items()} for t, d in cands.items()}
    print("CANDS", counts, flush=True)
    out = []
    seen = {t: 0 for t in TYPES}
    life1 = {"counter": 0, "blocker": 0, "checked": 0}
    pools = {}
    for t in ("counter", "blocker", "confirm_own", "confirm_opp", "select", "trigger"):
        pools[t] = [cands[t]["hb"], cands[t]["mu"]]
    # 確認の 2 種からドンの代金の点を分ける（両方の流れを順に見る）
    pos = {t: [0, 0] for t in pools}

    def take(t, side):
        lst = pools[t][side]
        i = pos[t][side]
        if i >= len(lst):
            return None
        pos[t][side] += 1
        return lst[i]

    def classify(k, p):
        g = games[k]
        pt = {"g": list(k), "step": p["step"], "seat": p["seat"], "turn": p["turn"]}
        try:
            game = replay(pt, g)
        except Exception:  # noqa: BLE001
            return None, None
        pend = json.loads(game.pending_json())
        if pend.get("player_id") != p["seat"]:
            return None, None
        return game, pt

    for t in ("counter", "blocker", "select", "trigger", "confirm_own", "confirm_opp"):
        side = 0
        tries = 0
        while tries < 4000:
            if t in ("counter", "blocker", "select", "trigger") and seen[t] >= a.n:
                break
            if t in ("confirm_own", "confirm_opp") and seen[t] >= a.n and seen["don_confirm"] >= a.n:
                break
            x = take(t, side)
            if x is None:
                x = take(t, 1 - side)
                if x is None:
                    break
            side = 1 - side
            tries += 1
            _, k, p = x
            game, pt = classify(k, p)
            if game is None:
                continue
            typ = t
            if t in ("counter", "blocker"):
                life = len(json.loads(game.hidden_json())["players"][p["seat"]]["life"])
                life1["checked"] += 1
                if life <= 1:
                    life1[t] += 1
                    continue
            if t in ("confirm_own", "confirm_opp"):
                if is_don_cost(game, p["seat"]):
                    typ = "don_confirm"
            if seen[typ] >= a.n:
                continue
            seen[typ] += 1
            pt.update(type=typ, src=k[0], dtype=p["dtype"], a_rec=p["move"], q_rec=chosen_q(p), n_cand=p["n_cand"])
            out.append(pt)
    for n, pt in enumerate(out):
        pt["pid"] = n
    with open(os.path.join(a.out, "points.jsonl"), "w", encoding="utf-8") as f:
        for pt in out:
            f.write(json.dumps(pt, ensure_ascii=False) + "\n")
    meta = {"cands": counts, "picked": seen, "battle_life1_skipped": life1}
    json.dump(meta, open(os.path.join(a.out, "points_meta.json"), "w"), ensure_ascii=False, indent=1)
    print("POINTS", len(out), seen, life1, flush=True)


# ---------------------------------------------------------------- 2. eval

def _init_eval():
    _boot()
    _G["games"] = load_all()


def settle(g, who, ws):
    """手の続きの自分の選択（対象など）は窓の根畳みと同じ決め方で（その世界のまま）選び、窓を閉じる。"""
    ship = _G["ship"]
    omni = json.dumps(dict(ship.opts, search_seed=int(ws), omniscient=True))
    for _ in range(6):
        pend = json.loads(g.pending_json())
        if g.winner() is not None or not pend or pend["player_id"] != who or not (g.has_interaction or g.phase != "MAIN"):
            break
        out = json.loads(g.decide(who, omni))
        if out.get("kind") != "window" or out.get("move") is None:
            break
        BA.apply(g, who, out["move"])
    info = json.loads(g.resolve_windows(json.dumps(dict(ship.opts, search_seed=int(ws)))))
    return info


def net_value(g, who):
    enc = json.loads(g.encode(who))
    enc["tok"] = enc.pop("tokens", None) or []
    return float(json.loads(E.engine().net_eval(json.dumps(enc), "[]"))["value"])


def root_value(out, mover, who):
    st = out.get("stats") or {}
    ns, qs = st.get("N") or [], st.get("Q") or []
    tot = sum(ns)
    if not tot:
        return None
    v = sum(n * q for n, q in zip(ns, qs)) / tot
    return v if mover == who else -v


def tree_values(game, who, mv, ws):
    """1 つの世界で: 手を当て・窓を閉じ・その世界のまま木（TREE_SIMS 各回数）。値は who から見た。"""
    g = game.fork_world(who, "fair", int(ws))
    g.apply_search_move(who, json.dumps(mv, ensure_ascii=False))
    info = settle(g, who, ws)
    w = g.winner()
    if w is not None:
        t = 1.0 if w == who else -1.0
        return {"net": t, **{str(s): t for s in TREE_SIMS}, "open": False}
    row = {"net": net_value(g, who), "open": info["open"]}
    pend = json.loads(g.pending_json())
    if not pend:
        return row
    mover = pend["player_id"]
    for s in TREE_SIMS:
        out = json.loads(g.fork().decide(mover, json.dumps({"net": _G["ship"].net, "sims": s, "omniscient": True,
                                                             "search_seed": int(ws) ^ 0x7EE5})))
        row[str(s)] = root_value(out, mover, who)
        row["kind" + str(s)] = out.get("kind")
    return row


def next_head(game, who, mv, ws):
    """手を当て、今のターンの残りと相手のターンを両席とも出荷の設定で打ち、who の次のターンの最初の判断の盤面の値。"""
    g = game.fork_world(who, "fair", int(ws))
    g.apply_search_move(who, json.dumps(mv, ensure_ascii=False))
    ship = _G["ship"]
    t0 = int(g.turn_count or 0)
    target = t0 + (2 if g.turn_player() == who else 1)
    carries = {"p1": DR.Carry(), "p2": DR.Carry()}
    steps = 0
    while g.winner() is None and steps < MAX_STEPS:
        pend = json.loads(g.pending_json())
        if not pend:
            break
        name = pend["player_id"]
        turn = int(g.turn_count or 0)
        if turn >= target and name == who and g.turn_player() == who and not g.has_interaction:
            break
        c = carries[name]
        out = json.loads(g.decide(name, ship.decide_opts(int(ws), turn, name, c.get(turn, name))))
        c.put(out)
        m = out.get("move")
        if m is None:
            break
        BA.apply(g, name, m)
        steps += 1
    w = g.winner()
    if w is not None:
        t = 1.0 if w == who else -1.0
        return {"v": t, "net": t, "end": True}
    turn = int(g.turn_count or 0)
    out = json.loads(g.fork().decide(who, _G["ref"].decide_opts(int(ws), turn, who)))
    return {"v": root_value(out, who, who), "net": net_value(g, who), "end": False}


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pick(vals, cur):
    """値の最大の候補（同点は今の手）。vals は候補ごとの値（None は除外）。"""
    best, bv = cur, vals[cur] if vals[cur] is not None else -9.0
    for i, v in enumerate(vals):
        if v is not None and v > bv + 1e-9:
            best, bv = i, v
    return best


def eval_point(p):
    t0 = time.time()
    who = p["seat"]
    row = {"pid": p["pid"], "type": p["type"], "ok": False}
    try:
        game = replay(p, _G["games"][gkey(p)])
    except Exception as exc:  # noqa: BLE001
        row["fail"] = "replay: %s" % type(exc).__name__
        return row
    seed = game_seed(p)
    ship = _G["ship"]
    turn = int(game.turn_count or 0)
    out = json.loads(game.fork().decide(who, ship.decide_opts(seed, turn, who, DR.Carry().get(turn, who))))
    if out.get("kind") != "window":
        row["fail"] = "kind=%s" % out.get("kind")
        return row
    legal = out["stats"]["legal"]
    m = len(legal)
    cur = next((i for i, mv in enumerate(legal) if BA.key(mv) == BA.key(out.get("move"))), 0)
    row.update(m=m, cur=cur, legal=legal, q_now=out["stats"]["Q"], n_now=out["stats"]["N"],
               same_as_record=BA.key(out.get("move")) == BA.key(p["a_rec"]),
               descs=[BA.desc(game, mv) for mv in legal], life=len(json.loads(game.hidden_json())["players"][who]["life"]),
               turn_player=game.turn_player())
    t1 = time.time()
    per = []  # per[w][j] = tree_values
    for w in range(WORLDS):
        ws = MJ.world_seed(seed, p["step"], OFFSET_READ + w)
        rw = []
        for mv in legal:
            try:
                rw.append(tree_values(game, who, mv, ws))
            except Exception as exc:  # noqa: BLE001
                rw.append({"err": "%s: %s" % (type(exc).__name__, str(exc)[:100])})
        per.append(rw)
    row["per"] = per
    row["sec_tree"] = round(time.time() - t1, 2)
    picks = {}
    for s in TREE_SIMS:
        for W in (2, WORLDS):
            vals = [_avg([per[w][j].get(str(s)) for w in range(W)]) for j in range(m)]
            picks[f"{s}w{W}"] = {"pick": pick(vals, cur), "vals": vals}
    picks["netw4"] = {"pick": pick([_avg([per[w][j].get("net") for w in range(WORLDS)]) for j in range(m)], cur)}
    row["picks"] = picks
    prim = picks["640w2"]["pick"]
    row["prim"] = prim
    t2 = time.time()
    if prim != cur:
        nx = {"cur": [], "tree": []}
        for w in range(NEXT_WORLDS):
            ws = MJ.world_seed(seed, p["step"], OFFSET_READ + 100 + w)
            for nm, j in (("cur", cur), ("tree", prim)):
                try:
                    nx[nm].append(next_head(game, who, legal[j], ws))
                except Exception as exc:  # noqa: BLE001
                    nx[nm].append({"err": "%s: %s" % (type(exc).__name__, str(exc)[:100])})
        row["next"] = nx
    row["sec_next"] = round(time.time() - t2, 2)
    row["ok"] = True
    row["sec"] = round(time.time() - t0, 1)
    return row


def _done(out_dir, prefix):
    done = set()
    for path in glob.glob(os.path.join(out_dir, prefix + "*.jsonl")):
        done |= {x["pid"] for x in BA.read_jsonl(path) if x.get("ok") or x.get("fail")}
    return done


def cmd_eval(a):
    pts = BA.read_jsonl(os.path.join(a.out, "points.jsonl"))
    si, sn = KR.shard_of(a)
    jobs = [p for p in pts if p["pid"] % sn == si]
    if a.limit:
        jobs = jobs[:a.limit]
    rpath = os.path.join(a.out, f"evals_shard{si}.jsonl" if a.shard else "evals.jsonl")
    done = _done(a.out, "evals")
    jobs = [p for p in jobs if p["pid"] not in done]
    print(f"EVAL shard={a.shard} todo {len(jobs)} (done {len(done)}) -> {rpath}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init_eval) as pool:
        for i, r in enumerate(pool.imap_unordered(eval_point, jobs, chunksize=1)):
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(jobs)}] pid={r['pid']} type={r['type']} ok={r['ok']} m={r.get('m')} cur={r.get('cur')} "
                  f"prim={r.get('prim')} sec={r.get('sec')} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- 3. judge

def load_evals(out_dir):
    ev = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "evals*.jsonl"))):
        for x in BA.read_jsonl(path):
            if x.get("ok"):
                ev[x["pid"]] = x
    return ev


def judge_jobs(out_dir):
    pts = {p["pid"]: p for p in BA.read_jsonl(os.path.join(out_dir, "points.jsonl"))}
    ev = load_evals(out_dir)
    return [(pts[pid], e) for pid, e in sorted(ev.items()) if e["prim"] != e["cur"]]


def judge_point(job):
    p, e = job
    t0 = time.time()
    who = p["seat"]
    row = {"pid": p["pid"], "type": p["type"], "ok": False}
    try:
        game = replay(p, _G["games"][gkey(p)])
    except Exception as exc:  # noqa: BLE001
        row["fail"] = "replay: %s" % type(exc).__name__
        return row
    seed = game_seed(p)
    a, b = e["legal"][e["cur"]], e["legal"][e["prim"]]
    wa, wb, errs = [], [], {}
    for i in range(_G["k"]):
        ws = MJ.world_seed(seed, p["step"], OFFSET_JUDGE + i)
        ra, ea = BA.rollout_ship(game, who, a, ws, _G["ref"], _G["ship"], both=True)
        rb, eb = BA.rollout_ship(game, who, b, ws, _G["ref"], _G["ship"], both=True)
        wa.append(ra)
        wb.append(rb)
        for x in (ea, eb):
            if x:
                errs[x] = errs.get(x, 0) + 1
    # 差は b − a ＝「短い木の手 − 今の手」
    row["ref"] = dict(MJ.paired(wa, wb), a=wa, b=wb, errs=errs or None)
    row["ok"] = row["ref"]["diff"] is not None
    row["sec"] = round(time.time() - t0, 1)
    return row


def _init_judge(k):
    _init_eval()
    _G["k"] = int(k)


def cmd_judge(a):
    jobs = judge_jobs(a.out)
    si, sn = KR.shard_of(a)
    jobs = [j for j in jobs if j[0]["pid"] % sn == si]
    rpath = os.path.join(a.out, f"judge_shard{si}.jsonl" if a.shard else "judge.jsonl")
    done = _done(a.out, "judge")
    jobs = [j for j in jobs if j[0]["pid"] not in done]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"JUDGE shard={a.shard} todo {len(jobs)} (done {len(done)}) k={a.k} -> {rpath}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init_judge, initargs=(a.k,)) as pool:
        for i, r in enumerate(pool.imap_unordered(judge_point, jobs, chunksize=1)):
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            ref = r.get("ref") or {}
            print(f"[{i + 1}/{len(jobs)}] pid={r['pid']} type={r['type']} ok={r['ok']} diff={ref.get('diff')} "
                  f"sec={r.get('sec')} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- 4. summary

def _ci(ds):
    ds = [d for d in ds if d is not None]
    n = len(ds)
    if n == 0:
        return {"n": 0, "mean": None, "ci": [None, None]}
    m = sum(ds) / n
    if n < 2:
        return {"n": n, "mean": round(m, 4), "ci": [None, None]}
    sd = (sum((d - m) ** 2 for d in ds) / (n - 1)) ** 0.5
    h = 1.96 * sd / n ** 0.5
    return {"n": n, "mean": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)]}


def _r(x, n=3):
    return None if x is None else round(x, n)


def cmd_summary(a):
    pts = {p["pid"]: p for p in BA.read_jsonl(os.path.join(a.out, "points.jsonl"))}
    ev = load_evals(a.out)
    jd = {}
    for path in sorted(glob.glob(os.path.join(a.out, "judge*.jsonl"))):
        for x in BA.read_jsonl(path):
            if x.get("ok"):
                jd[x["pid"]] = x
    fails = {}
    for path in sorted(glob.glob(os.path.join(a.out, "evals*.jsonl"))):
        for x in BA.read_jsonl(path):
            if not x.get("ok"):
                fails[x.get("fail")] = fails.get(x.get("fail"), 0) + 1
    by = {}
    rows = []
    for pid, e in sorted(ev.items()):
        t = e["type"]
        ch = {k: v["pick"] != e["cur"] for k, v in e["picks"].items()}
        j = jd.get(pid)
        diff = j["ref"]["diff"] if j else None
        # 短い木の値の差（短い木の手 − 今の手）・次のターンの頭の値の差
        v = e["picks"]["640w2"]["vals"]
        dtree = None if v[e["prim"]] is None or v[e["cur"]] is None else v[e["prim"]] - v[e["cur"]]
        dnext = dnet = None
        if e.get("next"):
            pa = [x.get("v") for x in e["next"]["cur"]]
            pb = [x.get("v") for x in e["next"]["tree"]]
            ds = [y - x for x, y in zip(pa, pb) if x is not None and y is not None]
            dnext = _r(sum(ds) / len(ds)) if ds else None
            na = [x.get("net") for x in e["next"]["cur"]]
            nb = [x.get("net") for x in e["next"]["tree"]]
            ds = [y - x for x, y in zip(na, nb) if x is not None and y is not None]
            dnet = _r(sum(ds) / len(ds)) if ds else None
        qn = e["q_now"]
        dnow = qn[e["prim"]] - qn[e["cur"]]
        row = {"pid": pid, "type": t, "src": pts[pid]["src"], "m": e["m"], "life": e["life"], "changed": ch, "diff": diff,
               "d_now": _r(dnow), "d_tree": _r(dtree), "d_next": dnext, "d_next_net": dnet,
               "cur": e["descs"][e["cur"]], "tree": e["descs"][e["prim"]], "same_as_record": e["same_as_record"],
               "sec": e["sec"], "sec_tree": e["sec_tree"]}
        rows.append(row)
        by.setdefault(t, []).append(row)
    out = {}
    for t in TYPES:
        rs = by.get(t, [])
        if not rs:
            continue
        g = {"n": len(rs), "m_mean": _r(sum(r["m"] for r in rs) / len(rs), 2)}
        for k in rs[0]["changed"]:
            g["change_" + k] = _r(sum(1 for r in rs if r["changed"][k]) / len(rs))
        jr = [r for r in rs if r["diff"] is not None]
        g["judged"] = _ci([r["diff"] for r in jr])
        g["judged_sig_tree_better"] = sum(1 for r in jr if r["diff"] is not None and _sig(jd[r["pid"]], 1))
        g["judged_sig_now_better"] = sum(1 for r in jr if r["diff"] is not None and _sig(jd[r["pid"]], -1))
        g["gain_per_point"] = _r(sum(r["diff"] for r in jr) / len(rs)) if jr else None
        g["corr_tree_vs_diff"] = _r(DRd._corr([r["d_tree"] for r in jr], [r["diff"] for r in jr]))
        g["corr_next_vs_diff"] = _r(DRd._corr([r["d_next"] for r in jr], [r["diff"] for r in jr]))
        g["next_agrees_with_tree"] = sum(1 for r in rs if r["d_next"] is not None and r["d_next"] > 0)
        g["next_measured"] = sum(1 for r in rs if r["d_next"] is not None)
        g["sec_tree_mean"] = _r(sum(r["sec_tree"] for r in rs) / len(rs), 2)
        g["sec_tree_per_cand"] = _r(sum(r["sec_tree"] / r["m"] for r in rs) / len(rs), 3)
        g["by_src"] = {s: {"n": sum(1 for r in rs if r["src"] == s),
                           "changed": sum(1 for r in rs if r["src"] == s and r["changed"]["640w2"]),
                           "judged": _ci([r["diff"] for r in jr if r["src"] == s])} for s in ("hb", "kr", "m5")}
        out[t] = g
    res = {"types": out, "fails": fails, "rows": rows}
    meta = os.path.join(a.out, "points_meta.json")
    if os.path.exists(meta):
        res["points_meta"] = json.load(open(meta))
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for t, g in out.items():
        print(t, json.dumps({k: v for k, v in g.items() if k != "by_src"}, ensure_ascii=False))
        print("   by_src", json.dumps(g["by_src"], ensure_ascii=False))
    print("fails", fails)


def _sig(j, sign):
    lo, hi = j["ref"]["ci"]
    if lo is None:
        return False
    return lo > 0 if sign > 0 else hi < 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["points", "eval", "judge", "summary"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n", type=int, default=N_PER_TYPE)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    {"points": cmd_points, "eval": cmd_eval, "judge": cmd_judge, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
