"""青ロックスの「ロックス海賊団で 6キングを手札に戻す」を、CPU が打った手と打ち比べる（WP `rocks-bounce`・読み取りだけ・
1 回きりの計測＝テスト無し・エンジンも出荷も変えない）。

場面: 紫カイドウ対青ロックスの記録（`matchup_kr`＝公式の構築例 200 局・`matchup_5` の kaido_rocks＝記事のデッキ 120 局）の
青ロックス（p2）のメインの判断点で、紫の場に 6キングがいて、青の手札にロックス海賊団があり、アクティブのドンが 5 枚以上のもの
（同じターンは最初の 1 点だけ）。「6キングだけ」（紫の場に 7 コスト以上がいない）と「他の大型あり」に分ける。

1. ``points``: 場面を拾って `points.jsonl` に書く（根の数字＝読んだ回数・評価・事前分布つき）。
2. ``judge``: 各点で a＝CPU の手・b＝ロックス海賊団（対象は 6キングに固定）を配り直した世界 16 本で打ち比べる
   （今のターンの残りは両席とも出荷の設定・その先は審判 r27 160 回・`blunder_audit.rollout_ship` と同じ形）。`--shard i/n`。
3. ``summary``: 群ごと・青の何ターン目ごとの「b − a」（プラス＝戻す方が良い）。

    python tests/scripts/rocks_bounce.py points
    python tests/scripts/rocks_bounce.py judge --workers 4 [--shard 0/4]
    python tests/scripts/rocks_bounce.py summary
"""
import argparse
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

OUT_DIR = os.path.join("n1_results", "rocks_bounce")
K6, EV = "EB04-031", "OP17-056"
BIG_COST = 7
#: 配り方の番号のずらし（他の計測と重ねない: matchup_kr 7000・matchup_5 8000・don_read 9000・… offdist_value 14000）。
OFFSET = 15000
_G = {}
_DB = {}


def cost(cid):
    if not _DB:
        for x in json.load(open(os.path.join("opcg_sim", "data", "opcg_cards.json"), encoding="utf-8")):
            _DB[x["number"]] = x
    try:
        return int(_DB.get(cid, {}).get("コスト"))
    except (TypeError, ValueError):
        return 0


def sources():
    """(名前, 局の表, 局 → (seed, 紫先攻か), replay) の列。"""
    kr = KR.load_games(KR.OUT_DIR)
    yield "kr", kr, (lambda g: (KR.SEED0 + g["idx"], g["kaido_first"])), (lambda g, s: KR.replay(g, s))
    m5 = M5.load_games(M5.OUT_DIR, "kaido_rocks")
    yield "m5", {k[1]: v for k, v in m5.items()}, (lambda g: (M5.game_setup("kaido_rocks", g["idx"])[0], g["me_first"])), (lambda g, s: M5.replay(g, s))


def new_game(src, g):
    return KR.new_game(g["idx"]) if src == "kr" else M5.new_game("kaido_rocks", g["idx"])


def replay(src, g, step):
    return KR.replay(g, step) if src == "kr" else M5.replay(g, step)


def hidden(game):
    return json.loads(game.hidden_json())


def cmd_points(a):
    E.engine()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for src, games, meta, _rp in sources():
        for idx in sorted(games):
            g = games[idx]
            seed, kaido_first = meta(g)
            game = new_game(src, g)
            seen = set()
            for s, mv in enumerate(g["moves"]):
                pend = json.loads(game.pending_json())
                name = pend["player_id"]
                if name == "p2" and game.turn_player() == "p2" and pend.get("action") == "MAIN_ACTION":
                    turn = int(game.turn_count or 0)
                    if turn not in seen:
                        h = hidden(game)
                        p1f = [c["card_id"] for c in h["players"]["p1"]["field"]]
                        p2 = h["players"]["p2"]
                        ev = [c["uuid"] for c in p2["hand"] if c["card_id"] == EV]
                        k6 = [c["uuid"] for c in h["players"]["p1"]["field"] if c["card_id"] == K6]
                        active = len((p2.get("don") or {}).get("active") or [])
                        if k6 and ev and active >= 5:
                            seen.add(turn)
                            p = g["points"][s]
                            top = p.get("top") or []
                            tot = sum(x["n"] for x in top) or 1.0
                            bmv = {"kind": "game", "action_type": "PLAY", "payload": {"uuid": ev[0]}}
                            hit = [x for x in top if BA.pkey(x["mv"]) == BA.pkey(bmv)]
                            bigs = [c for c in p1f if cost(c) >= BIG_COST]
                            rows.append({"src": src, "idx": idx, "seed": seed, "step": s, "turn": turn,
                                         "ot": KR.own_turn(turn, "p2", kaido_first), "kaido_first": kaido_first,
                                         "group": "k6_only" if not bigs else "with_big", "p1_field": p1f, "p1_bigs": bigs,
                                         "k6_uuid": k6[0], "a": p["a_box"] or p["move"], "b": bmv,
                                         "root_a": {"n": top[0]["n"], "share": round(top[0]["n"] / tot, 4), "q": top[0]["q"], "p": top[0].get("p")} if top else None,
                                         "root_b": {"n": hit[0]["n"], "share": round(hit[0]["n"] / tot, 4), "q": hit[0]["q"], "p": hit[0].get("p")} if hit else None,
                                         "chosen_b": bool(hit) and BA.pkey(p["a_box"] or p["move"]) == BA.pkey(bmv),
                                         "won_rocks": not g["kaido_won"] if src == "kr" else not g["me_won"]})
                BA.apply(game, name, mv)
                if game.winner() is not None:
                    break
    path = os.path.join(a.out, "points.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    c = collections.Counter((r["group"], r["ot"]) for r in rows)
    print("POINTS", len(rows), "->", path)
    for k in sorted(c):
        print(" ", k, c[k])


# ---------------------------------------------------------------- judge

def _init(k):
    E.engine()
    BA._G["ship"] = BA.ship_spec(KR.NET)
    _G["ref"] = E.SeatSpec(KR.NET, sims=MJ.REFEREE_SIMS)
    _G["k"] = int(k)


def rollout(game, move, wseed, k6_uuid, max_steps=MJ.MAX_STEPS):
    """`BA.rollout_ship(both=True)` と同じだが、手の直後にロックス海賊団の対象の問いが来たら 6キングを選ぶ。"""
    who = "p2"
    g = game.fork_world(who, "fair", int(wseed))
    t0 = int(game.turn_count or 0)
    forced = False
    try:
        g.apply_search_move(who, json.dumps(move, ensure_ascii=False))
        carries = {}
        steps = 0
        while g.winner() is None and steps < max_steps:
            pending = json.loads(g.pending_json())
            if not pending:
                break
            name = pending["player_id"]
            turn = int(g.turn_count or 0)
            if (not forced and name == who and turn == t0 and pending.get("action") == "SEARCH_AND_SELECT"
                    and "ロックス海賊団" in (pending.get("message") or "")):
                forced = True
                BA.apply(g, name, {"kind": "game", "action_type": "RESOLVE_EFFECT_SELECTION",
                                   "payload": {"selected_uuids": [k6_uuid], "index": 0, "accepted": True, "position": "BOTTOM", "declared_value": 0}})
                steps += 1
                continue
            use_ship = turn == t0
            spec = BA._G["ship"] if use_ship else _G["ref"]
            c = carries.setdefault((name, use_ship), DR.Carry())
            out = json.loads(g.decide(name, spec.decide_opts(int(wseed), turn, name, c.get(turn, name))))
            c.put(out)
            mv = out.get("move")
            if mv is None:
                break
            BA.apply(g, name, mv)
            steps += 1
    except Exception as exc:  # noqa: BLE001
        return None, "%s: %s" % (type(exc).__name__, str(exc)[:120]), forced
    w = g.winner()
    if w is None:
        return None, "no_winner", forced
    return (1 if w == who else 0), None, forced


def judge_point(job):
    g, c = job
    t0 = time.time()
    row = dict(c, ok=False, fail=None)
    try:
        game = replay(c["src"], g, c["step"])
    except Exception as exc:  # noqa: BLE001
        row["fail"] = f"replay: {type(exc).__name__}"
        return row
    pend = json.loads(game.pending_json())
    if pend.get("player_id") != "p2" or int(game.turn_count or 0) != c["turn"]:
        row["fail"] = "seat/turn"
        return row
    row["a_desc"], row["b_desc"] = BA.desc(game, c["a"]), BA.desc(game, c["b"])
    wa, wb, errs, forced_n = [], [], {}, 0
    for i in range(_G["k"]):
        ws = MJ.world_seed(c["seed"], c["step"], OFFSET + i)
        ra, ea, _ = rollout(game, c["a"], ws, c["k6_uuid"])
        rb, eb, fb = rollout(game, c["b"], ws, c["k6_uuid"])
        forced_n += bool(fb)
        wa.append(ra)
        wb.append(rb)
        for e in (ea, eb):
            if e:
                errs[e] = errs.get(e, 0) + 1
    row["ref"] = dict(MJ.paired(wa, wb), a=wa, b=wb, errs=errs or None)
    row["forced"] = forced_n
    row["ok"] = row["ref"]["diff"] is not None
    row["sec"] = round(time.time() - t0, 1)
    return row


def cmd_judge(a):
    pts = BA.read_jsonl(os.path.join(a.out, "points.jsonl"))
    if a.group:
        pts = [c for c in pts if c["group"] == a.group]
    if a.turns:
        lo, hi = (int(x) for x in a.turns.split("-"))
        pts = [c for c in pts if lo <= c["ot"] <= hi or c["group"] == "k6_only"]
    si, sn = KR.shard_of(a)
    jobs = [c for n, c in enumerate(pts) if n % sn == si]
    games = {}
    for src, gs, _m, _r in sources():
        games[src] = gs
    rpath = os.path.join(a.out, f"judge_shard{si}.jsonl" if a.shard else "judge.jsonl")
    done = set(KR.read_glob(os.path.join(a.out, "judge*.jsonl"), lambda x: (x["src"], x["idx"], x["step"])))
    jobs = [(games[c["src"]][c["idx"]], c) for c in jobs if (c["src"], c["idx"], c["step"]) not in done]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"JUDGE shard={a.shard} todo {len(jobs)} (done {len(done)}) k={a.k} -> {rpath}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.k,)) as pool:
        for i, r in enumerate(pool.imap_unordered(judge_point, jobs, chunksize=1)):
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            ref = r.get("ref") or {}
            print(f"[{i + 1}/{len(jobs)}] {r['src']} idx={r['idx']} step={r['step']} ot={r['ot']} {r['group']} ok={r['ok']} "
                  f"diff={ref.get('diff')} ci={ref.get('ci')} forced={r.get('forced')} sec={r.get('sec')} ({time.time() - t0:.0f}s)", flush=True)
    print("JUDGE_DONE", flush=True)


# ---------------------------------------------------------------- summary

def cmd_summary(a):
    rows = [r for r in KR.read_glob(os.path.join(a.out, "judge*.jsonl"), lambda x: (x["src"], x["idx"], x["step"])).values() if r.get("ok")]
    print(f"判定できた点 {len(rows)}")

    def line(label, rs):
        ds = [r["ref"]["diff"] for r in rs]
        if not ds:
            return
        mc = KR.mean_ci(ds)
        m, (lo, hi) = mc["mean"], (mc["ci"][0] if mc["ci"][0] is not None else mc["mean"], mc["ci"][1] if mc["ci"][1] is not None else mc["mean"])
        sig_b = sum(1 for r in rs if (r["ref"].get("ci") or [0, 0])[0] > 0)
        sig_a = sum(1 for r in rs if (r["ref"].get("ci") or [0, 0])[1] < 0)
        print(f"| {label} | {len(rs)} | {m:+.3f} | {lo:+.3f}〜{hi:+.3f} | {sig_b} | {sig_a} |")

    print("| 群 | 点 | 戻す − CPU の手 | 95% 区間 | 戻す方が有意に良い | CPU の手が有意に良い |")
    print("|---|---:|---:|---|---:|---:|")
    line("全体", rows)
    for grp, lab in (("k6_only", "6キングだけ"), ("with_big", "他の大型あり")):
        rs = [r for r in rows if r["group"] == grp]
        line(lab, rs)
        for ot in sorted({r["ot"] for r in rs}):
            line(f"　{lab}・青の {ot} ターン目", [r for r in rs if r["ot"] == ot])
    line("CPU が実際に戻した点", [r for r in rows if r["chosen_b"]])
    line("CPU が戻さなかった点", [r for r in rows if not r["chosen_b"]])
    live = [r for r in rows if r.get("root_a") and abs(r["root_a"]["q"]) < 0.9]
    line("勝負が決まっていない点（CPU の評価の絶対値 0.9 未満）", live)
    for grp, lab in (("k6_only", "6キングだけ"), ("with_big", "他の大型あり")):
        line(f"　{lab}・勝負が決まっていない点", [r for r in live if r["group"] == grp])
    out = {"n": len(rows), "rows": [{k: r[k] for k in ("src", "idx", "step", "ot", "group", "p1_bigs", "a_desc", "b_desc", "root_a", "root_b", "chosen_b", "forced")} | {"diff": r["ref"]["diff"], "ci": r["ref"].get("ci")} for r in rows]}
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["points", "judge", "summary"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--group", default=None)
    ap.add_argument("--turns", default=None, help="他の大型ありの群を青の何ターン目に絞る（例 4-6）。6キングだけは常に全部")
    a = ap.parse_args(argv)
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
