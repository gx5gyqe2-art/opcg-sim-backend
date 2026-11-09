"""方針の地図のずれは本当の勝率の差か——ずれた点で「勧める方針の手」と「打った手」を打ち比べる
（WP `plan-bias` 段 1・`docs/search_efficiency_plan.md` §7.64・読み取り専用）。

1 回きりの測定の道具なのでテストは足さない（CLAUDE.md の理論の道具の規約）。

1. ``play``: 両席 r27・**出荷の設定**（1,280 回・世界 4 本・枝予算なし・詰みの証明 root＝
   `blunder_audit.ship_spec`）・random × real で局を打つ。全ての判断点で打った手・根の候補（訪問の
   多い順に上位 12）・両席のライフ・手の分類（face／board／develop…・`plan_labels.move_class`）を書き、
   **自席ターンの最初の main** と**相手ターンの最初の自分の判断**では盤面の符号化（`game.encode`）も書く
   （地図用ネットの方針ヘッドをそこで引く＝`plan_bias_map.py` の行と同じ取り方）。
2. ``select``: 地図用ネット（方針ヘッド付き）で各ターンの勧める方針を引き、**決着まで自席 2 ターン以内**
   （残りターン数は局の記録から数えた真値）の**ずれた点**を拾う。比べる 2 つの手:
   - 攻め（勧め face／board／mixed と打った方針が食い違うターン）: 打った側にあって勧めに無い種類の
     最初の手（例: 勧め face・打った board → 最初に盤面を触った手）を a、その判断点の根の候補のうち
     勧めの種類（face なら相手リーダーへの攻撃・board ならキャラへの攻撃か除去）で訪問が最多の手を b。
   - 守り（勧め guard・打った take）: その相手ターンで、守る手（カウンター・ブロッカー）が選べたのに
     見送った最初の窓の手を a、その窓の根の候補で訪問が最多の守る手を b。逆（勧め take・打った guard）は
     最初に守った手を a、見送る手を b。
3. ``judge``: 点ごとに局の手の列を当て直して局面を作り、配り方 K 通り（既定 16）で a と b の先を最後まで
   打つ（`blunder_audit.rollout_ship`＝**今のターンの残りは両席とも出荷の設定**・その先は審判 r27 の
   160 回）。同じ配り方と乱数で対の差（b − a）。``--shard i/n`` で点を分けられる。
4. ``summary``: 種類別（攻め／守り・向き）の差の平均と区間、b が勝った点の割合、地図の差との関係。

配り方の番号は ``world_seed(seed, step, OFFSET + i)``（OFFSET＝7640・`blunder_audit` の 2000／
`hold_bias` などのずらしと重ねない）。局の seed 帯は ``--seed-base``（既定 7,640,000）。

    python tests/scripts/plan_bias_roll.py play --games 40 --workers 4
    python tests/scripts/plan_bias_roll.py select --net ~/nrel_pmap.npz
    python tests/scripts/plan_bias_roll.py judge --workers 4 [--shard 0/4]
    python tests/scripts/plan_bias_roll.py summary
"""
import argparse
import json
import os
import random
import sys
import time
from multiprocessing import get_context

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPCG_LOG_SILENT", "1")

from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402
from opcg_sim.loop import record_gen as RG  # noqa: E402
from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
import blunder_audit as BA  # noqa: E402  ship_spec／ranked／chosen／apply／rollout_ship／desc
import move_judge as MJ  # noqa: E402  paired／world_seed

OUT_DIR = os.path.join("n1_results", "plan_bias", "roll")
NET = "opcg_sim/data/learned/nrel_r27.npz"
OFFSET = 7640
TOP = 12
_G = {}


def _cards():
    if "cards" not in _G:
        _G["cards"] = PL.Cards(_G.setdefault("db", D.load_db()))
    return _G["cards"]


def u2c_of(game):
    h = json.loads(game.hidden_json())
    out = {}
    for p in h["players"].values():
        for z in ("leader", "stage"):
            c = p.get(z)
            if c:
                out[c["uuid"]] = c["card_id"]
        for z in ("deck", "hand", "life", "field", "trash", "temp_zone"):
            for c in p.get(z) or ():
                out[c["uuid"]] = c["card_id"]
    return out, h


def mclass(mv, u2c):
    """手 → face／board／develop／end／don／guard／pass／other（攻めは `plan_labels.move_class`）。"""
    if mv is None:
        return None
    if mv.get("kind") == "battle":
        at = mv.get("action_type") or ""
        if mv.get("card_uuid"):
            return "guard" if ("COUNTER" in at or "BLOCK" in at) else "other"
        return "pass"
    if mv.get("action_type") == "SETUP_BOX":
        mv = BA.first_primitive(mv)
    try:
        sig = RG.move_sig(mv)
    except Exception:  # noqa: BLE001
        return "other"
    return PL.move_class(sig, u2c, _cards()) or "other"


def is_guard_move(mv):
    return bool(mv) and mv.get("kind") == "battle" and bool(mv.get("card_uuid")) and \
        any(s in (mv.get("action_type") or "") for s in ("COUNTER", "BLOCK"))


# ---------------------------------------------------------------- 1. play

def play_one(args):
    seed, a = args
    E.engine()
    db = _G.setdefault("db", D.load_db())
    spec = BA.ship_spec(a.net)
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "real")
    pts, moves = [], []
    seen = set()

    def obs(game, name, turn, step, out, move):
        u2c, h = u2c_of(game)
        opp = "p2" if name == "p1" else "p1"
        own = game.turn_player() == name
        kind = out.get("kind")
        rows = BA.ranked(out) if kind in ("main", "window") else []
        ch = BA.chosen(out, rows) if rows else None
        rec = {"step": step, "seat": name, "turn": turn, "own": own, "kind": kind,
               "act": json.loads(game.pending_json()).get("action"), "move": move,
               "cls": mclass(ch["mv"] if ch else move, u2c), "a_box": ch["mv"] if ch else None,
               "top": [{"mv": r["mv"], "n": r["n"], "q": round(r["q"], 4), "cls": mclass(r["mv"], u2c)}
                       for r in rows[:TOP]],
               "my_life": len(h["players"][name]["life"]), "opp_life": len(h["players"][opp]["life"])}
        key = (name, turn, own)
        if turn >= 1 and key not in seen and (kind == "main" or not own):
            seen.add(key)
            enc = json.loads(game.encode(name))
            rec["enc"] = {"scalars": [round(float(x), 5) for x in enc["scalars"]],
                          "tokens": [round(float(x), 5) for x in enc["tokens"]],
                          "card_idx": [int(x) for x in enc["card_idx"]]}
        pts.append(rec)
        moves.append(move)

    t0 = time.perf_counter()
    try:
        r = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, observer=obs)
        winner, turns = r["winner"], r["turns"]
    except DR.GameAborted as exc:
        winner, turns = None, str(exc)
    return {"seed": seed, "leaders": [la, lb], "winner": winner, "turns": turns,
            "sec": round(time.perf_counter() - t0, 1), "points": pts, "moves": moves}


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def shard_of(xs, shard):
    if not shard:
        return xs
    i, n = (int(x) for x in shard.split("/"))
    return [x for k, x in enumerate(xs) if k % n == i]


def cmd_play(a):
    os.makedirs(a.out, exist_ok=True)
    gpath = os.path.join(a.out, f"games{('_' + a.shard.replace('/', 'of')) if a.shard else ''}.jsonl")
    done = {g["seed"] for g in read_jsonl(gpath)}
    seeds = [s for s in shard_of([a.seed_base + i for i in range(a.games)], a.shard) if s not in done]
    print(f"PLAY todo {len(seeds)} (done {len(done)}) -> {gpath}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers) as pool:
        for i, g in enumerate(pool.imap_unordered(play_one, [(s, a) for s in seeds], chunksize=1)):
            with open(gpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(seeds)}] seed={g['seed']} pts={len(g['points'])} sec={g['sec']} "
                  f"winner={g['winner']} ({time.time() - t0:.0f}s)", flush=True)
    print("PLAY_DONE", flush=True)


# ---------------------------------------------------------------- 2. select

def turn_plans(g):
    """局 → {(seat, turn): {"plan": 0/1/2/-1, "face": n, "board": n}}・{(seat, opp_turn): 3/4}・残りターン数。"""
    pts = g["points"]
    att = {}
    for p in pts:
        if p["own"] and p["turn"] >= 1 and p["kind"] in ("main", "commit", "lethal") or (p["own"] and p.get("act") in ("MAIN", "MAIN_ACTION")):
            c = att.setdefault((p["seat"], p["turn"]), {"face": 0, "board": 0, "develop": 0})
            if p["cls"] in c:
                c[p["cls"]] += 1
    plan = {}
    for k, c in att.items():
        lab = PL.turn_label(c)
        plan[k] = PL.PLAN_CLASSES.index(lab) if lab in ("face", "board", "mixed") else -1
    # 守り: 相手ターン t の間に相手のリーダー攻撃が在り、自分のライフが減ったか
    first_life = {}
    for p in pts:
        first_life.setdefault((p["seat"], p["turn"]), p["my_life"])
        opp = "p2" if p["seat"] == "p1" else "p1"
        first_life.setdefault((opp, p["turn"]), p["opp_life"])
    guard = {}
    for (s, t), c in att.items():
        opp = "p2" if s == "p1" else "p1"
        if c["face"] <= 0:
            continue
        a0, a1 = first_life.get((opp, t)), first_life.get((opp, t + 1))
        if a0 is None:
            continue
        if a1 is None:                                   # 局がこのターンで終わった
            a1 = 0 if g["winner"] == s else a0
        guard[(opp, t)] = 3 if a0 - a1 >= 1 else 4
    # 残りターン数（決着までの自席ターン数・今のターンを含む／相手ターンなら次から）
    own_turns = {}
    for p in pts:
        if p["own"]:
            own_turns.setdefault(p["seat"], set()).add(p["turn"])

    def clock(seat, t, own):
        ts = sorted(own_turns.get(seat, ()))
        return len([x for x in ts if (x >= t if own else x > t)])
    return plan, guard, clock


def cmd_select(a):
    from opcg_sim.learned import n_rel as NL
    from opcg_sim.learned.train import n_rel_train as NT
    from opcg_sim.learned.train.n_eff_feat import build_eff_tables
    import plan_bias_map as PBM
    tabs = build_eff_tables()
    net = NL.NRelNet.load(a.net, tabs[:5])
    if not net.plan:
        raise SystemExit("このネットに方針ヘッドが無い")
    games = [g for f in sorted(os.listdir(a.out)) if f.startswith("games") for g in read_jsonl(os.path.join(a.out, f))]
    games = [g for g in games if g["winner"] in ("p1", "p2")]
    out = []
    stats = {"games": len(games), "attack_rows": 0, "defense_rows": 0}
    for g in games:
        plan, guard, clock = turn_plans(g)
        pts = g["points"]
        encs = [p for p in pts if "enc" in p]
        if not encs:
            continue
        sc = np.array([p["enc"]["scalars"] for p in encs], np.float32)
        tok = np.array([p["enc"]["tokens"] for p in encs], np.float32).reshape(len(encs), 22, -1)
        ci = np.array([p["enc"]["card_idx"][:22] for p in encs], np.int64)
        sc, tok = PBM._pad(sc, tok)
        _v, vq = net.value_with_plan(sc, ci, tok, *NT.relations_or_zeros(net, ci, tok, None))
        for p, q in zip(encs, vq):
            q = [float(x) for x in q]
            s, t = p["seat"], p["turn"]
            ck = clock(s, t, p["own"])
            z = 1.0 if g["winner"] == s else -1.0
            base = {"seed": g["seed"], "seat": s, "turn": t, "clock": ck, "vq": q, "z": z}
            if p["own"]:
                stats["attack_rows"] += 1
                pl = plan.get((s, t), -1)
                rec = int(np.argmax(q[:3]))
                if ck > a.clock or pl < 0 or pl == rec:
                    continue
                pt = attack_point(g, s, t, rec, pl)
                if pt:
                    out.append({**base, "side": "attack", "rec": PL.PLAN_CLASSES[rec], "played": PL.PLAN_CLASSES[pl],
                                "gap": q[rec] - q[pl], **pt})
            else:
                stats["defense_rows"] += 1
                pl = guard.get((s, t), -1)
                rec = 3 + int(q[4] > q[3])
                if ck > a.clock or pl < 0 or pl == rec:
                    continue
                pt = defense_point(g, s, t, rec)
                if pt:
                    out.append({**base, "side": "defense", "rec": PL.PLAN_CLASSES[rec], "played": PL.PLAN_CLASSES[pl],
                                "gap": q[rec] - q[pl], **pt})
    rng = random.Random(a.sample_seed)
    by = {}
    for x in out:
        by.setdefault((x["side"], x["rec"], x["played"]), []).append(x)
    stats["candidates"] = {"/".join(k): len(v) for k, v in sorted(by.items())}
    keep = []
    for side, cap in (("attack", a.max_attack), ("defense", a.max_defense)):
        xs = [x for x in out if x["side"] == side]
        rng.shuffle(xs)
        keep += sorted(xs[:cap], key=lambda x: (x["seed"], x["step"]))
    with open(os.path.join(a.out, "points.jsonl"), "w", encoding="utf-8") as f:
        for x in keep:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    stats["kept"] = len(keep)
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    with open(os.path.join(a.out, "select_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)


def attack_point(g, s, t, rec, pl):
    want_face = rec in (0, 2)
    want_board = rec in (1, 2)
    did_face = pl in (0, 2)
    did_board = pl in (1, 2)
    if did_board and not want_board:
        bad, good = "board", "face"
    elif did_face and not want_face:
        bad, good = "face", "board"
    elif want_board and not did_board:
        bad, good = "face", "board"
    elif want_face and not did_face:
        bad, good = "board", "face"
    else:
        return None
    for p in g["points"]:
        if p["seat"] != s or p["turn"] != t or not p["own"] or p["kind"] != "main":
            continue
        if p["cls"] != bad and (p.get("a_box") is None or mclass_box(p) != bad):
            continue
        a_mv = p["a_box"] or p["move"]
        b = next((c for c in p["top"] if c["cls"] == good), None)
        if b is None:
            return None
        return {"step": p["step"], "a": a_mv, "b": b["mv"], "a_cls": bad, "b_cls": good,
                "b_n": b["n"], "a_n": next((c["n"] for c in p["top"] if BA.key(c["mv"]) == BA.key(a_mv)), None)}
    return None


def mclass_box(p):
    for c in p["top"]:
        if BA.key(c["mv"]) == BA.key(p["a_box"]):
            return c["cls"]
    return None


def defense_point(g, s, t, rec):
    for p in g["points"]:
        if p["seat"] != s or p["turn"] != t or p["own"] or p["kind"] != "window":
            continue
        guards = [c for c in p["top"] if is_guard_move(c["mv"])]
        nog = [c for c in p["top"] if not is_guard_move(c["mv"])]
        if rec == 4 and not is_guard_move(p["move"]) and guards:
            return {"step": p["step"], "a": p["move"], "b": guards[0]["mv"], "a_cls": "pass", "b_cls": "guard",
                    "b_n": guards[0]["n"], "act": p["act"], "my_life": p["my_life"]}
        if rec == 3 and is_guard_move(p["move"]) and nog:
            return {"step": p["step"], "a": p["move"], "b": nog[0]["mv"], "a_cls": "guard", "b_cls": "pass",
                    "b_n": nog[0]["n"], "act": p["act"], "my_life": p["my_life"]}
    return None


# ---------------------------------------------------------------- 3. judge

def _init_judge(net, k):
    E.engine()
    _G["db"] = D.load_db()
    _G["ship"] = BA.ship_spec(net)
    _G["ref"] = E.SeatSpec(net, sims=MJ.REFEREE_SIMS)
    _G["k"] = int(k)
    BA._G["db"] = _G["db"]


def judge_point(job):
    g, x = job
    t0 = time.time()
    out = {"seed": x["seed"], "step": x["step"], "side": x["side"], "rec": x["rec"], "played": x["played"]}
    try:
        game = BA.replay(g, x["step"])
    except Exception as exc:  # noqa: BLE001
        out["error"] = "replay: %s: %s" % (type(exc).__name__, str(exc)[:120])
        return out
    wa, wb = [], []
    for i in range(_G["k"]):
        ws = MJ.world_seed(x["seed"], x["step"], OFFSET + i)
        wa.append(BA.rollout_ship(game, x["seat"], x["a"], ws, _G["ref"], _G["ship"])[0])
        wb.append(BA.rollout_ship(game, x["seat"], x["b"], ws, _G["ref"], _G["ship"])[0])
    out.update(MJ.paired(wa, wb), a=wa, b=wb, sec=round(time.time() - t0, 1))
    return out


def cmd_judge(a):
    src = a.src or a.out
    games = {g["seed"]: g for f in sorted(os.listdir(src)) if f.startswith("games")
             for g in read_jsonl(os.path.join(src, f))}
    pts = shard_of(read_jsonl(os.path.join(src, "points.jsonl")), a.shard)
    rpath = os.path.join(a.out, f"judge{('_' + a.shard.replace('/', 'of')) if a.shard else ''}.jsonl")
    done = {(r["seed"], r["step"]) for r in read_jsonl(rpath)}
    jobs = [({k: v for k, v in games[x["seed"]].items() if k != "points"}, x) for x in pts
            if (x["seed"], x["step"]) not in done]
    print(f"JUDGE todo {len(jobs)} (done {len(done)}) k={a.k} -> {rpath}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init_judge, initargs=(a.net_ship, a.k)) as pool:
        for i, r in enumerate(pool.imap_unordered(judge_point, jobs, chunksize=1)):
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"JUDGE {r['side']} {r['rec']}<-{r['played']} seed={r['seed']} step={r['step']} "
                  f"diff={r.get('diff')} {r.get('ci')} err={r.get('error')} sec={r.get('sec')} "
                  f"({i + 1}/{len(jobs)} {time.time() - t0:.0f}s)", flush=True)
    print("JUDGE_DONE", flush=True)


# ---------------------------------------------------------------- 4. summary

def cmd_summary(a):
    pts = {(x["seed"], x["step"]): x for x in read_jsonl(os.path.join(a.out, "points.jsonl"))}
    res = [r for f in sorted(os.listdir(a.out)) if f.startswith("judge") for r in read_jsonl(os.path.join(a.out, f))]
    seen = set()
    res = [r for r in res if not ((r["seed"], r["step"]) in seen or seen.add((r["seed"], r["step"])))]
    res = [dict(pts.get((r["seed"], r["step"]), {}), **r) for r in res if r.get("n")]

    def block(rs):
        if not rs:
            return {"points": 0}
        a_all = [v for r in rs for v in r["a"]]
        b_all = [v for r in rs for v in r["b"]]
        pooled = MJ.paired(a_all, b_all)
        ds = [r["diff"] for r in rs]
        m = float(np.mean(ds)); sd = float(np.std(ds, ddof=1)) if len(ds) > 1 else 0.0
        h = 1.96 * sd / np.sqrt(len(ds)) if len(ds) > 1 else float("inf")
        return {"points": len(rs), "pairs": pooled["n"], "win_a": pooled["wa"], "win_b": pooled["wb"],
                "diff_mean_over_points": round(m, 4), "ci_points": [round(m - h, 4), round(m + h, 4)],
                "b_better": round(float(np.mean([d > 0 for d in ds])), 3),
                "b_worse": round(float(np.mean([d < 0 for d in ds])), 3),
                "b_sig_better": sum(1 for r in rs if r["ci"][0] is not None and r["ci"][0] > 0),
                "b_sig_worse": sum(1 for r in rs if r["ci"][1] is not None and r["ci"][1] < 0),
                "map_gap_mean": round(float(np.mean([r.get("gap", 0.0) for r in rs])), 4),
                "corr_gap_diff": (round(float(np.corrcoef([r.get("gap", 0.0) for r in rs], ds)[0, 1]), 3)
                                  if len(rs) > 2 and np.std(ds) > 0 else None)}
    out = {"all": block(res)}
    for side in ("attack", "defense"):
        rs = [r for r in res if r["side"] == side]
        out[side] = block(rs)
        for k in sorted({(r["rec"], r["played"]) for r in rs}):
            out[f"{side}:{k[0]}<-{k[1]}"] = block([r for r in rs if (r["rec"], r["played"]) == k])
    out["errors"] = sum(1 for f in sorted(os.listdir(a.out)) if f.startswith("judge")
                        for r in read_jsonl(os.path.join(a.out, f)) if r.get("error"))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("play", "select", "judge", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--out", default=OUT_DIR)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--shard", default=None, help="i/n（play は局・judge は点を n 本に分けた i 本目）")
        if name == "play":
            p.add_argument("--games", type=int, default=40)
            p.add_argument("--seed-base", type=int, default=7_640_000)
            p.add_argument("--net", default=NET)
        if name == "select":
            p.add_argument("--net", required=True, help="方針ヘッド付きの地図用ネット")
            p.add_argument("--clock", type=int, default=2, help="決着までの自席ターン数がこれ以下の点だけ")
            p.add_argument("--max-attack", type=int, default=40)
            p.add_argument("--max-defense", type=int, default=40)
            p.add_argument("--sample-seed", type=int, default=7640)
        if name == "judge":
            p.add_argument("--k", type=int, default=16)
            p.add_argument("--net-ship", default=NET)
            p.add_argument("--src", default=None, help="games*.jsonl と points.jsonl の在る所（既定 --out）")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    {"play": cmd_play, "select": cmd_select, "judge": cmd_judge, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
