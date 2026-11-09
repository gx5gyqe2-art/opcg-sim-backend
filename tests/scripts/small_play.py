"""パワー 4000 以下のキャラを出した手は得か損か（WP `small-play`・`docs/search_efficiency_plan.md` §7.60・台帳 2+461）。

読み取りだけの計測（エンジン・ネット・既定の挙動は変えない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。
土台は `hold_bias.py`（局・判定の形）と `blunder_audit.py`（`replay`・`rollout_ship`）。局は §7.59 の 150 局
（`n1_results/hold_bias/games.jsonl`・両席 r27・出荷の設定・random × real）をそのまま使う。

1. ``scan``: 全局を探索なしで当て直し、記録の PLAY の uuid からカードを引く（カード DB の元のパワー・コスト・効果）。
   出した後の場の体数・次のターンまで（相手の次のターンの終わりまで）に場を離れたか（行き先）を記録だけで数える
   （`cards.jsonl`＝uuid の表・`plays.jsonl`＝実際に当てた全ての PLAY）。
2. ``judge``: メインの判断点（``kind == "main"``・続き・証明・マリガンは除く）で、打った手 a が
   **パワー 4000 以下のキャラの登場**の点。比べる手 b＝根の候補（記録の上位 3）のうち「パワー 4000 以下のキャラを出す」で
   ない最上位の手（無ければターン終了）。点は pure な順位で 400 まで。判定は `hold_bias` と同じ（公平な配り直し・a と b の対・
   1 点 8 通り・今のターンの残りは両席とも出荷の設定、その先は審判〔r27・160 回・世界 1 本〕）。
3. ``summary``: 「出した − 出さなかった」の平均と 95% 区間を、全体とパワー・コスト・効果・段階・b の種類・場の体数・手札で割る。

    python tests/scripts/small_play.py scan
    python tests/scripts/small_play.py judge --workers 4 --shard 0/8      # points_shard0.jsonl
    python tests/scripts/small_play.py summary

再開できる（終えた点は飛ばす）。
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blunder_audit as BA  # noqa: E402
import hold_bias as HB  # noqa: E402
import move_judge as MJ  # noqa: E402

from opcg_sim.loop import decks as D, engine as E  # noqa: E402

GAMES = os.path.join("n1_results", "hold_bias", "games.jsonl")
OUT_DIR = os.path.join("n1_results", "small_play")
NET = HB.NET
#: 判定の配り方の番号のずらし（blunder_audit の 0〜2000 番台・hold_bias の 3000 番台と重ねない）。
OFFSET = 4000
SMALL = 4000
CAP = 400
TURN_END = {"kind": "game", "action_type": "TURN_END", "payload": {}}
_G = {}


# ---------------------------------------------------------------- カード

def card_table():
    """number → {power, cost, type, effect}（カード DB の元の値・パワーの空欄は 0）。"""
    if "cards" not in _G:
        raw = json.load(open(os.path.join("opcg_sim", "data", "opcg_cards.json"), encoding="utf-8"))
        out = {}
        for x in raw:
            t = (x.get("効果(テキスト)") or "").strip()
            if x.get("種類") != "キャラクター":
                eff = None
            elif not t:
                eff = "vanilla"
            elif "登場時" in t:
                eff = "on_play"
            elif re.search("ブロッカー", t):
                eff = "blocker"
            else:
                eff = "other"
            out[x["number"]] = {"type": x.get("種類"), "power": int(x.get("パワー") or 0), "cost": int(x.get("コスト") or 0),
                                "effect": eff, "name": x.get("name"), "text": t[:200]}
        _G["cards"] = out
    return _G["cards"]


def uuid_map(game):
    """uuid → card_id（その時点で見える全ての札・両席の山札・手札・ライフ・場・トラッシュ）。"""
    h = json.loads(game.hidden_json())
    out = {}

    def walk(o):
        if isinstance(o, dict):
            if "uuid" in o and "card_id" in o:
                out[o["uuid"]] = o["card_id"]
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(h)
    return out


def is_small(mv, umap):
    """手が「パワー 4000 以下のキャラの登場」か。"""
    if not mv or mv.get("action_type") != "PLAY":
        return False
    c = card_table().get(umap.get((mv.get("payload") or {}).get("uuid")))
    return bool(c and c["type"] == "キャラクター" and c["power"] <= SMALL)


def b_type(mv, umap):
    at = mv.get("action_type")
    p = mv.get("payload") or {}
    if at == "TURN_END":
        return "turn_end"
    if at == "ATTACK" or (at == "DON_BOX" and p.get("target_ids")):
        return "attack"
    if at == "PLAY":
        c = card_table().get(umap.get(p.get("uuid"))) or {}
        return "char" if c.get("type") == "キャラクター" else "event_stage"
    if at in ("DON_BOX", "ATTACH_DON"):
        return "attach"
    if at == "ACTIVATE_MAIN":
        return "activate"
    return str(at).lower()


def pick(p, umap):
    """(a, b, b_src)。a は打った手（箱の代表）・b は上位のうち小さいキャラの登場でない最上位（無ければターン終了）。"""
    a = p["a_box"] or p["move"]
    for t in p["top"]:
        if BA.key(t["mv"]) != BA.key(a) and not is_small(t["mv"], umap):
            return a, t["mv"], "top"
    return a, TURN_END, "turn_end_fill"


# ---------------------------------------------------------------- 1. scan

def field_of(game, seat):
    h = json.loads(game.hidden_json())
    return [c["uuid"] for c in h["players"][seat]["field"]]


def where(game, seat, uuid):
    h = json.loads(game.hidden_json())
    for seat2 in ("p1", "p2"):
        pl = h["players"][seat2]
        for z in ("field", "hand", "trash", "deck", "life"):
            if any(c.get("uuid") == uuid for c in (pl.get(z) or []) if isinstance(c, dict)):
                return z if seat2 == seat else "opp_" + z
    return "unknown"


def scan_one(g):
    E.engine()
    db = _G.setdefault("db", D.load_db())
    p1, p2 = D.build_pair(db, g["leaders"][0], g["leaders"][1], g["seed"], "real")
    game = BA.DR.new_game(g["seed"], p1, p2)
    umap = uuid_map(game)
    plays, live = [], []
    for s, mv in enumerate(g["moves"]):
        name = json.loads(game.pending_json())["player_id"]
        rec = None
        if mv.get("action_type") == "PLAY":
            u = (mv.get("payload") or {}).get("uuid")
            cid = umap.get(u)
            c = card_table().get(cid) or {}
            if c.get("type") == "キャラクター":
                h = json.loads(game.hidden_json())
                p = g["points"][s]
                rec = {"seed": g["seed"], "step": s, "seat": name, "turn": int(game.turn_count or 0), "uuid": u,
                       "card": cid, "power": c["power"], "cost": c["cost"], "effect": c["effect"], "kind": p["kind"],
                       "field_before": len(h["players"][name]["field"]), "hand_before": len(h["players"][name]["hand"])}
        BA.apply(game, name, mv)
        umap.update(uuid_map(game))
        if rec is not None:
            rec["on_field"] = rec["uuid"] in field_of(game, name)
            rec["field_after"] = len(field_of(game, name))
            plays.append(rec)
            if rec["on_field"]:
                live.append(rec)
        # 次のターン（相手の次のターン）の終わりまで場に残ったか
        t = int(game.turn_count or 0)
        for r in list(live):
            if r["uuid"] not in field_of(game, r["seat"]):
                r["left"] = where(game, r["seat"], r["uuid"])
                r["left_turn"] = t
                r["left_by_opp"] = t > r["turn"]  # 相手の次のターンに離れた（同じターンなら自分の手番の中）
                live.remove(r)
            elif t > r["turn"] + 1:
                r["left"] = None
                live.remove(r)
        if game.winner() is not None:
            break
    for r in live:
        r["left"] = None
        r["game_ended"] = True
    return {"seed": g["seed"], "umap": umap, "plays": plays}


def cmd_scan(a):
    os.makedirs(a.out, exist_ok=True)
    games = BA.read_jsonl(a.games)
    with get_context("fork").Pool(a.workers) as pool:
        res = pool.map(scan_one, games, chunksize=4)
    with open(os.path.join(a.out, "cards.jsonl"), "w", encoding="utf-8") as f:
        for r in sorted(res, key=lambda r: r["seed"]):
            f.write(json.dumps({"seed": r["seed"], "umap": r["umap"]}) + "\n")
    with open(os.path.join(a.out, "plays.jsonl"), "w", encoding="utf-8") as f:
        for r in sorted(res, key=lambda r: r["seed"]):
            for x in r["plays"]:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("SCAN games", len(res), "plays", sum(len(r["plays"]) for r in res), flush=True)


def load_umaps(out):
    return {x["seed"]: x["umap"] for x in BA.read_jsonl(os.path.join(out, "cards.jsonl"))}


# ---------------------------------------------------------------- 2. judge

def _rank(seed, step):
    """点の並び（局を足しても既に選んだ点の順位が変わらない・pure・hold_bias と別のずらし）。"""
    return E.search_seed(int(seed) ^ 0x5A11_9DA7, int(step), "p1")


def job_order(games, umaps, cap=CAP):
    pool = []
    for s in sorted(games):
        g = games[s]
        um = umaps[s]
        light = {k: v for k, v in g.items() if k != "points"}
        for p in g["points"]:
            if p["kind"] != "main" or p["dtype"] == "mulligan":
                continue
            if is_small(p["a_box"] or p["move"], um):
                pool.append((light, p))
    pool.sort(key=lambda j: _rank(j[0]["seed"], j[1]["step"]))
    print(f"POOL {len(pool)} take {min(cap, len(pool))}", flush=True)
    return pool[:cap]


def _init(k):
    HB._init(k)
    _G["k"] = int(k)
    _G["umaps"] = load_umaps(_G["out"])


def judge_point(job):
    g, p = job
    t0 = time.time()
    seed, step, name = g["seed"], p["step"], p["seat"]
    um = _G["umaps"][seed]
    a, b, b_src = pick(p, um)
    ca = card_table()[um[a["payload"]["uuid"]]]
    row = {"seed": seed, "step": step, "seat": name, "turn": p["turn"], "a_rank": p["a_rank"], "n_cand": p["n_cand"],
           "my_life": p["my_life"], "opp_life": p["opp_life"], "my_hand": p["my_hand"], "opp_hand": p["opp_hand"],
           "card": um[a["payload"]["uuid"]], "power": ca["power"], "cost": ca["cost"], "effect": ca["effect"],
           "b_src": b_src, "b_type": b_type(b, um), "a": a, "b": b, "ok": False, "why": None}
    try:
        game = BA.replay(g, step)
    except Exception as exc:  # noqa: BLE001
        row["why"] = f"replay: {type(exc).__name__}"
        return row
    pend = json.loads(game.pending_json())
    if pend.get("player_id") != name or int(game.turn_count or 0) != p["turn"]:
        row["why"] = "seat/turn"
        return row
    row["field_before"] = len(field_of(game, name))
    row["a_desc"], row["b_desc"] = BA.desc(game, a), BA.desc(game, b)
    row["top"] = [{"d": BA.desc(game, t["mv"]), "n": t["n"], "q": t["q"]} for t in p["top"][:3]]
    wa, wb, errs = [], [], {}
    for i in range(_G["k"]):
        ws = MJ.world_seed(seed, step, OFFSET + i)
        ra, ea = BA.rollout_ship(game, name, a, ws, HB._G["ref"], BA._G["ship"], both=True)
        rb, eb = BA.rollout_ship(game, name, b, ws, HB._G["ref"], BA._G["ship"], both=True)
        wa.append(ra)
        wb.append(rb)
        for e in (ea, eb):
            if e:
                errs[e] = errs.get(e, 0) + 1
    row["ref"] = dict(MJ.paired(wa, wb), a=wa, b=wb, errs=errs or None)
    row["ok"] = row["ref"]["diff"] is not None
    if not row["ok"]:
        row["why"] = "no_pairs"
    row["sec"] = round(time.time() - t0, 1)
    return row


def read_points(out):
    rows = {}
    for path in sorted(glob.glob(os.path.join(out, "points*.jsonl"))):
        for x in BA.read_jsonl(path):
            k = (x["seed"], x["step"])
            if k not in rows or (x["ok"] and not rows[k]["ok"]):
                rows[k] = x
    return list(rows.values())


def cmd_judge(a):
    _G["out"] = a.out
    games = {g["seed"]: g for g in BA.read_jsonl(a.games)}
    jobs = job_order(games, load_umaps(a.out), a.cap)
    si, sn = (int(x) for x in a.shard.split("/")) if a.shard else (0, 1)
    jobs = [j for idx, j in enumerate(jobs) if idx % sn == si]
    if a.limit:
        jobs = jobs[:a.limit]
    rpath = os.path.join(a.out, f"points_shard{si}.jsonl" if a.shard else "points.jsonl")
    done = {(x["seed"], x["step"]) for x in read_points(a.out)}
    jobs = [j for j in jobs if (j[0]["seed"], j[1]["step"]) not in done]
    print(f"JUDGE shard={a.shard} todo {len(jobs)} (done overall {len(done)}) k={a.k} -> {rpath}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.k,)) as pool:
        for i, row in enumerate(pool.imap_unordered(judge_point, jobs, chunksize=1)):
            row["k"] = a.k
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            r = row.get("ref") or {}
            print(f"JUDGE seed={row['seed']} step={row['step']} {row['card']} p={row['power']} b={row['b_type']} "
                  f"why={row['why']} d={r.get('diff')} sec={row.get('sec')} ({i + 1}/{len(jobs)} {time.time() - t0:.0f}s)",
                  flush=True)
    print("JUDGE_DONE", flush=True)


# ---------------------------------------------------------------- 3. summary

def pband(x):
    return "0-2000" if x <= 2000 else str(x)


def cband(x):
    return str(x) if x <= 4 else "5+"


def fband(x):
    return str(x) if x <= 4 else "5"


def hband(x):
    return "0-3" if x <= 3 else "4-5" if x <= 5 else "6+"


def gain(x):
    """出した − 出さなかった（a − b）。"""
    return -x["ref"]["diff"]


def split(rows, keyf):
    by = {}
    for x in rows:
        by.setdefault(keyf(x), []).append(x)
    return {k: dict(HB.mean_ci([gain(x) for x in v]), sig_loss=None) for k, v in sorted(by.items(), key=lambda kv: str(kv[0]))}


def cmd_summary(a):
    rows = read_points(a.out)
    ok = [x for x in rows if x["ok"]]
    plays = BA.read_jsonl(os.path.join(a.out, "plays.jsonl"))
    games = BA.read_jsonl(a.games)
    s = {"games": len(games), "rows": len(rows), "ok": len(ok),
         "not_ok": {}, "sec_per_point": round(sum(x["sec"] for x in ok) / len(ok), 1) if ok else None,
         "all": HB.mean_ci([gain(x) for x in ok]),
         "wins": {"a": round(sum(x["ref"]["wa"] for x in ok) / max(len(ok), 1), 4),
                  "b": round(sum(x["ref"]["wb"] for x in ok) / max(len(ok), 1), 4)}}
    for x in rows:
        if not x["ok"]:
            s["not_ok"][x["why"]] = s["not_ok"].get(x["why"], 0) + 1
    cuts = {"power": lambda x: pband(x["power"]), "cost": lambda x: cband(x["cost"]), "effect": lambda x: x["effect"],
            "phase": lambda x: HB.phase(x["turn"]), "b_type": lambda x: x["b_type"],
            "field_after": lambda x: fband(x["field_before"] + 1), "hand_before": lambda x: hband(x["my_hand"]),
            "a_rank": lambda x: "top1" if x["a_rank"] == 0 else "not_top1",
            "effect|power": lambda x: x["effect"] + "|" + pband(x["power"]),
            "effect|phase": lambda x: x["effect"] + "|" + HB.phase(x["turn"])}
    s["cuts"] = {}
    for name, f in cuts.items():
        d = split(ok, f)
        for v in d.values():
            v["sig_loss"] = bool(v["ci"][1] is not None and v["ci"][1] < 0)
            v["sig_gain"] = bool(v["ci"][0] is not None and v["ci"][0] > 0)
        s["cuts"][name] = d
    s["big_loss"] = sum(1 for x in ok if x["ref"]["diff"] >= 0.20 and (x["ref"]["ci"][0] or -1) > 0)
    s["big_gain"] = sum(1 for x in ok if x["ref"]["diff"] <= -0.20 and (x["ref"]["ci"][1] or 1) < 0)
    # 記録だけの数え（全局・実際に当てた PLAY のうち小さいキャラ）
    small = [x for x in plays if x["power"] <= SMALL]
    seats = len(games) * 2
    cen = {"seats": seats, "char_plays": len(plays), "small_plays": len(small),
           "small_per_seat": round(len(small) / seats, 2),
           "small_per_seat_by_kind": {}, "by_effect": {}}
    for k in sorted({x["kind"] for x in small}):
        cen["small_per_seat_by_kind"][k] = round(sum(1 for x in small if x["kind"] == k) / seats, 2)

    def removal(v):
        landed = [x for x in v if x.get("on_field")]
        gone = [x for x in landed if x.get("left")]
        by_opp = [x for x in gone if x.get("left_by_opp")]
        dest = {}
        for x in gone:
            dest[x["left"]] = dest.get(x["left"], 0) + 1
        return {"n": len(v), "per_seat": round(len(v) / seats, 2), "landed": len(landed),
                "gone_by_next_turn": len(gone), "gone_rate": round(len(gone) / max(len(landed), 1), 3),
                "gone_on_opp_turn": len(by_opp), "gone_on_opp_turn_rate": round(len(by_opp) / max(len(landed), 1), 3),
                "dest": dest}
    cen["all"] = removal(small)
    for e in ("vanilla", "on_play", "blocker", "other"):
        cen["by_effect"][e] = removal([x for x in small if x["effect"] == e])
    cen["by_power"] = {k: removal([x for x in small if pband(x["power"]) == k]) for k in ("0-2000", "3000", "4000")}
    cen["big_chars"] = removal([x for x in plays if x["power"] > SMALL])
    s["census"] = cen
    txt = json.dumps(s, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("ALL", s["all"], "ok", len(ok), s["not_ok"])
    for name, d in s["cuts"].items():
        for k, v in d.items():
            print(f"{name:12s} {k:22s} n={v['n']:4d} mean={v['mean']} ci={v['ci']}")
    print(json.dumps(cen, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--games", default=GAMES)
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("judge")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--shard", default=None, help="i/n: 全体の並びの番号 mod n == i の点だけ（出力 points_shard{i}.jsonl）")
    p.add_argument("--limit", type=int, default=0, help="試し: シャードの先頭から n 点だけ")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cap", type=int, default=CAP)
    p.add_argument("--games", default=GAMES)
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("summary")
    p.add_argument("--games", default=GAMES)
    p.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)
    {"scan": cmd_scan, "judge": cmd_judge, "summary": cmd_summary}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
