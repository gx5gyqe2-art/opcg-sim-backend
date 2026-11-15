"""負け方の内訳（WP `ceiling-map` 段 B・`docs/search_efficiency_plan.md` §7.92）。

読み取りだけの計測（エンジン・ネット・既定の挙動は変えない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。
土台は `blunder_audit.py`（局の記録・当て直し・手の同一性）と `move_judge.py`（固定の審判が最後まで打つ対の差）。

1. ``play``: 両席 r29f4・**出荷の設定**（1,280 回・世界 8 本・枝予算なし・詰みの証明 root・公平・守りの対話の木 all）・
   random × real で局を打ち、全ての判断点で打った手・決め方・根の候補・盤面の要約・decide に渡した持ち越しを記録する
   （`games.jsonl`）。候補が 2 つ以上の判断点を 2 つの層に分けて、局ごとに無作為に引く（`pick`）:
   - 層 T（**読む回数に依る**判断）: 木が決めた点（`main`）と、短い木で読んだ窓。局ごとに上限 ``PER_T`` 点。
   - 層 S（読む回数に依らない判断）: 静的な値で比べる窓・詰みの証明が決めた点。局ごとに上限 ``PER_S`` 点（替わらないことの確かめ）。
   続き（`commit`＝前の木の決定の残りの手順）は引かない——親の木の決定の箱に含まれ、審判は箱全体を当てるので二重に数えない。
2. ``read``: 引いた点ごとに記録の手を当て直して局面を作り、同じ持ち越し・同じ探索の種で
   (i) 1 倍の読み（記録の手が再現するかの確かめ）、(ii) **10 倍の読み**（読む回数だけ 10 倍）、
   (iii) 1 倍の読みを探索の種だけ替えて（乱数だけで替わる割合＝雑音の床）を回す（`reads.jsonl`）。
3. ``judge``: 10 倍の読みの手が打った手と違う点で、固定の審判（r29f4・160 回・世界 1 本）が両方の手の先を同じ配り方
   K 通りで最後まで打ち、対の差（10 倍の手 − 打った手＝**打った手の損**）を出す（`judge.jsonl`）。
4. ``summary``: シャードをつないで、切り口ごとの「点の数 × 替わる割合 × 1 点あたりの損」と取り分・検算（`summary.json`）。

    python tests/scripts/loss_map.py play --games 15 --seed-base 7341000 --out n1_results/loss_map/w01 --workers 4
    python tests/scripts/loss_map.py read --out n1_results/loss_map/w01 --workers 4
    python tests/scripts/loss_map.py judge --out n1_results/loss_map/w01 --workers 4
    python tests/scripts/loss_map.py summary --shards 'n1_results/loss_map/w*' --out n1_results/loss_map

再開できる（終えた局・点は飛ばす）。
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time
from multiprocessing import get_context
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPCG_LOG_SILENT", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402
import blunder_audit as BA  # noqa: E402  ship_spec／ranked／chosen／dtype／key／replay／desc
import move_judge as MJ  # noqa: E402  rollout／paired／world_seed

OUT_DIR = os.path.join("n1_results", "loss_map")
MULT = 10
PER_T = 40
PER_S = 4
#: 審判の配り方の番号のずらし（`blunder_audit` 0〜／1000〜／2000〜・`search_ceiling` 7900〜 と重ねない）。
JUDGE_OFFSET = 9200
#: 雑音の床の読みの探索の種のずらし。
NOISE_XOR = 0x10557A9000000001
_G: Dict[str, Any] = {}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def root_value(rows) -> Optional[float]:
    """根の値（候補の値の訪問での加重平均・手を打つ側から見た −1〜+1）。"""
    tot = sum(r["n"] for r in rows)
    if not rows:
        return None
    if tot <= 0:
        return max(r["q"] for r in rows)
    return sum(r["n"] * r["q"] for r in rows) / tot


def stratum(p: Dict[str, Any]) -> Optional[str]:
    """T＝読む回数に依る判断・S＝依らない判断・None＝引かない（候補 1 つ・続き）。"""
    if (p.get("n_cand") or 0) < 2 or p.get("kind") == "commit":
        return None
    if p.get("kind") == "main" or (p.get("kind") == "window" and (p.get("wread") or {}).get("tree")):
        return "T"
    return "S"


# ---------------------------------------------------------------- 1. play

def play_one(args):
    seed, net = args
    E.engine()
    db = D.load_db()
    spec = BA.ship_spec(net)
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "real")
    pts, moves = [], []
    carries = {"p1": DR.Carry(), "p2": DR.Carry()}

    def obs(game, name, turn, step, out, move):
        pend = json.loads(game.pending_json())
        own = game.turn_player() == name
        kind = out.get("kind")
        rows = BA.ranked(out)
        if kind in ("main", "window"):
            n_cand = len(rows)
        else:
            try:
                n_cand = len(json.loads(game.legal_json(name)))
            except Exception:  # noqa: BLE001
                n_cand = None
        cin = carries[name].get(turn, name)
        cin = {"commit": list(cin["commit"]), "resact_pending": bool(cin["resact_pending"])}
        carries[name].put(out)
        h = json.loads(game.hidden_json())
        opp = "p2" if name == "p1" else "p1"
        ch = BA.chosen(out, rows) if rows else None
        me, op = h["players"][name], h["players"][opp]

        def don(pl):
            d = pl.get("don") or {}
            a, r = d.get("active"), d.get("rested")
            return [len(a) if isinstance(a, list) else a, len(r) if isinstance(r, list) else r]
        rv = root_value(rows) if kind in ("main", "window") else None
        rec = {"step": step, "seat": name, "turn": turn, "own": own, "act": pend.get("action"), "kind": kind,
               "dtype": BA.dtype(pend, own, move), "n_cand": n_cand, "move": move,
               "a_box": ch["mv"] if ch else None, "a_rank": rows.index(ch) if ch else None,
               "top": [{"mv": r["mv"], "n": r["n"], "q": round(r["q"], 4)} for r in rows[:3]],
               "root_v": None if rv is None else round(rv, 4), "q_a": round(ch["q"], 4) if ch else None,
               "wread": out.get("window"),
               "carry_in": cin if (cin["commit"] or cin["resact_pending"]) else None,
               "my_life": len(me["life"]), "opp_life": len(op["life"]),
               "my_hand": len(me["hand"]), "opp_hand": len(op["hand"]),
               "my_field": len(me["field"]), "opp_field": len(op["field"]),
               "my_deck": len(me["deck"]), "opp_deck": len(op["deck"]),
               "my_don": don(me), "opp_don": don(op)}
        pts.append(rec)
        moves.append(move)

    t0 = time.perf_counter()
    try:
        r = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, observer=obs)
        winner, turns, void = r["winner"], r["turns"], r["winner"] is None
    except DR.GameAborted as exc:
        winner, turns, void = None, None, str(exc)
    rng = random.Random(seed * 7 + 3)
    by = {"T": [], "S": []}
    for p in pts:
        s = stratum(p)
        if s:
            by[s].append(p["step"])
    pick = {"T": sorted(rng.sample(by["T"], min(PER_T, len(by["T"])))),
            "S": sorted(rng.sample(by["S"], min(PER_S, len(by["S"]))))}
    return {"seed": seed, "leaders": [la, lb], "winner": winner, "turns": turns, "void": void,
            "sec": round(time.perf_counter() - t0, 1), "n_points": len(pts),
            "n_T": len(by["T"]), "n_S": len(by["S"]), "pick": pick, "points": pts, "moves": moves}


def cmd_play(a):
    os.makedirs(a.out, exist_ok=True)
    gpath = os.path.join(a.out, "games.jsonl")
    done = {g["seed"] for g in read_jsonl(gpath)}
    seeds = [a.seed_base + i for i in range(a.games) if a.seed_base + i not in done]
    print(f"PLAY todo {len(seeds)} (done {len(done)})", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers) as pool:
        for i, g in enumerate(pool.imap_unordered(play_one, [(s, a.net) for s in seeds], chunksize=1)):
            with open(gpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(seeds)}] seed={g['seed']} pts={g['n_points']} T={g['n_T']} S={g['n_S']} "
                  f"sec={g['sec']} winner={g['winner']} void={g['void']} ({time.time() - t0:.0f}s)", flush=True)
    print("PLAY_DONE", flush=True)


# ---------------------------------------------------------------- 2. read

def _init(k=0):
    E.engine()
    BA._G["db"] = D.load_db()
    _G["spec"] = BA.ship_spec(E.DEFAULT_NET)
    _G["ref"] = E.SeatSpec(E.DEFAULT_NET, sims=MJ.REFEREE_SIMS)
    _G["k"] = int(k)


def _decide(game, name, opts):
    t0 = time.process_time()
    out = json.loads(game.decide(name, json.dumps(opts, ensure_ascii=False)))
    rows = BA.ranked(out)
    ch = BA.chosen(out, rows) if rows else None
    box = ch["mv"] if ch else out.get("move")
    return {"out": out, "rows": rows, "ch": ch, "box": box, "key": BA.key(box), "cpu": round(time.process_time() - t0, 2)}


def read_point(job):
    g, p, strat = job
    res = {"seed": g["seed"], "step": p["step"], "stratum": strat}
    try:
        game = BA.replay(g, p["step"])
        name = json.loads(game.pending_json())["player_id"]
        if name != p["seat"]:
            res["error"] = "replay: seat %s != %s" % (name, p["seat"])
            return res
        base = json.loads(_G["spec"].decide_opts(g["seed"], p["turn"], name, p.get("carry_in")))
        base["world_threads"] = 1
        a_mv = p.get("a_box") or p["move"]
        a_key = BA.key(a_mv)
        r1 = _decide(game, name, base)
        hi = dict(base, sims=int(base["sims"]) * _G["mult"])
        r10 = _decide(game, name, hi)
        nz = dict(base, search_seed=E.search_seed(g["seed"] ^ NOISE_XOR, p["turn"], name))
        rn = _decide(game, name, nz)
        rows = r10["rows"]
        tot = sum(r["n"] for r in rows) or 1.0
        a10 = next((r for r in rows if BA.key(r["mv"]) == a_key), None)
        rv10 = root_value(rows) if r10["out"].get("kind") in ("main", "window") else None
        res.update({
            "repro": r1["key"] == a_key, "kind1": r1["out"].get("kind"), "cpu1": r1["cpu"],
            "kind10": r10["out"].get("kind"), "cpu10": r10["cpu"], "key_a": a_key, "key_b": r10["key"],
            "changed": r10["key"] != a_key, "b": r10["box"],
            "desc_a": BA.desc(game, a_mv), "desc_b": BA.desc(game, r10["box"]),
            "root_v10": None if rv10 is None else round(rv10, 4),
            "q10_b": round(r10["ch"]["q"], 4) if r10["ch"] else None,
            "n10_b": round(r10["ch"]["n"] / tot, 4) if r10["ch"] else None,
            "q10_a": round(a10["q"], 4) if a10 else None, "n10_a": round(a10["n"] / tot, 4) if a10 else None,
            "wread10": r10["out"].get("window"),
            "noise_changed": rn["key"] != a_key, "noise_key": rn["key"],
        })
    except Exception as exc:  # noqa: BLE001
        res["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:160])
    return res


def _slim(g):
    return {k: v for k, v in g.items() if k != "points"}


def cmd_read(a):
    games = read_jsonl(os.path.join(a.out, "games.jsonl"))
    rpath = os.path.join(a.out, "reads.jsonl")
    done = {(r["seed"], r["step"]) for r in read_jsonl(rpath)}
    jobs = []
    for g in games:
        if g.get("void"):
            continue
        by = {p["step"]: p for p in g["points"]}
        gs = _slim(g)
        for strat in ("T", "S"):
            for s in g["pick"][strat]:
                if (g["seed"], s) not in done:
                    jobs.append((gs, by[s], strat))
    # 重い点（木の点）が後ろに偏らないように混ぜる（途中の保存がどの局にも均等に進む）
    random.Random(1).shuffle(jobs)
    _G["mult"] = a.mult
    print(f"READ todo {len(jobs)} (done {len(done)}) mult={a.mult}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init, maxtasksperchild=40) as pool:
        for i, r in enumerate(pool.imap_unordered(read_point, jobs, chunksize=1)):
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0 or r.get("error"):
                print(f"READ {i + 1}/{len(jobs)} seed={r['seed']} step={r['step']} {r.get('stratum')} changed={r.get('changed')} "
                      f"repro={r.get('repro')} cpu10={r.get('cpu10')} {r.get('error') or ''} ({time.time() - t0:.0f}s)", flush=True)
    print("READ_DONE", flush=True)


# ---------------------------------------------------------------- 3. judge

def judge_point(job):
    g, p, r = job
    t0 = time.time()
    out = {"seed": g["seed"], "step": p["step"], "key_b": r["key_b"]}
    try:
        game = BA.replay(g, p["step"])
    except Exception as exc:  # noqa: BLE001
        out["error"] = "replay: %s: %s" % (type(exc).__name__, str(exc)[:120])
        return out
    ma, mb = p.get("a_box") or p["move"], r["b"]
    wa, wb, errs = [], [], {}
    for i in range(_G["k"]):
        ws = MJ.world_seed(g["seed"], p["step"], JUDGE_OFFSET + i)
        ra, ea = MJ.rollout(game, p["seat"], ma, ws, _G["ref"])
        rb, eb = MJ.rollout(game, p["seat"], mb, ws, _G["ref"])
        wa.append(ra)
        wb.append(rb)
        for e in (ea, eb):
            if e:
                errs[e] = errs.get(e, 0) + 1
    out.update(MJ.paired(wa, wb), a=wa, b=wb, errs=errs or None, k=_G["k"], sec=round(time.time() - t0, 1))
    return out


def cmd_judge(a):
    games = read_jsonl(os.path.join(a.out, "games.jsonl"))
    reads = [r for r in read_jsonl(os.path.join(a.out, "reads.jsonl")) if r.get("changed") and not r.get("error")]
    jpath = os.path.join(a.out, "judge.jsonl")
    done = {(r["seed"], r["step"]) for r in read_jsonl(jpath)}
    gby = {g["seed"]: g for g in games}
    jobs = []
    for r in reads:
        if (r["seed"], r["step"]) in done:
            continue
        g = gby[r["seed"]]
        p = next(x for x in g["points"] if x["step"] == r["step"])
        jobs.append((_slim(g), p, r))
    random.Random(2).shuffle(jobs)
    print(f"JUDGE todo {len(jobs)} (done {len(done)}) k={a.k}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.k,), maxtasksperchild=20) as pool:
        for i, r in enumerate(pool.imap_unordered(judge_point, jobs, chunksize=1)):
            with open(jpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"JUDGE {i + 1}/{len(jobs)} seed={r['seed']} step={r['step']} diff={r.get('diff')} {r.get('ci')} "
                  f"sec={r.get('sec')} {r.get('error') or ''} ({time.time() - t0:.0f}s)", flush=True)
    print("JUDGE_DONE", flush=True)


# ---------------------------------------------------------------- 4. summary

#: §7.90 段 2 の 10 倍の対戦（512 ペア）: +29.9［+15.1, +44.8］Elo。
ARENA_ELO = (29.9, 15.1, 44.8)
KIND_JA = {"main:play": "展開", "main:attack": "攻撃", "main:attach": "ドンの付与（付与して攻撃の箱を含む）",
           "main:activate_main": "起動", "main:turn_end": "ターン終了", "main:setup_box": "準備の箱",
           "counter": "カウンター", "blocker": "ブロック", "mulligan": "マリガン"}


def elo_gain(e: float) -> float:
    """Elo の差 → 勝率の上乗せ（0.5 からの差）。"""
    return 1.0 / (1.0 + 10 ** (-e / 400.0)) - 0.5


def kind_of(dt: str) -> str:
    if dt in KIND_JA:
        return KIND_JA[dt]
    if dt.startswith("own_dialog:"):
        return "効果の対話（自分の手番）"
    if dt.startswith("opp_dialog:"):
        return "効果の対話（相手の手番）"
    return dt


def move_kind(mv: Optional[Dict[str, Any]]) -> str:
    """手そのものの種類（10 倍の手の行き先の表用）。"""
    at = (mv or {}).get("action_type")
    p = (mv or {}).get("payload") or {}
    if at == "DON_BOX":
        k = int(p.get("don_k") or 0)
        return "攻撃" if k <= 0 and p.get("target_ids") else ("付与して攻撃" if p.get("target_ids") else "付与")
    return {"PLAY": "展開", "ATTACK": "攻撃", "ATTACH_DON": "付与", "ACTIVATE_MAIN": "起動", "TURN_END": "ターン終了",
            "SETUP_BOX": "準備の箱"}.get(at, str(at))


def band(x, edges, labels):
    for e, lab in zip(edges, labels):
        if x <= e:
            return lab
    return labels[-1]


def features(g, p, colors) -> Dict[str, str]:
    seat_i = 0 if p["seat"] == "p1" else 1
    rv = p.get("root_v")
    f = {
        "種類": kind_of(p["dtype"]),
        "時期": "0（マリガン）" if p["turn"] <= 0 else band(p["turn"], [4, 8, 12], ["手番 1〜4", "手番 5〜8", "手番 9〜12", "手番 13〜"]),
        "自分のライフ": band(p["my_life"], [0, 1, 2, 3], ["0", "1", "2", "3", "4 以上"]),
        "相手のライフ": band(p["opp_life"], [0, 1, 2, 3], ["0", "1", "2", "3", "4 以上"]),
        "リーダーの色": colors.get(g["leaders"][seat_i], "?"),
        "有利不利": "（証明・値なし）" if rv is None else band(
            rv, [-0.6, -0.2, 0.2, 0.6], ["大きく不利（〜−0.6）", "不利（−0.6〜−0.2）", "五分（−0.2〜+0.2）", "有利（+0.2〜+0.6）", "大きく有利（+0.6〜）"]),
        "手番の持ち主": "自分の手番" if p["own"] else "相手の手番",
        "候補の数": band(p["n_cand"], [2, 5, 10], ["2", "3〜5", "6〜10", "11 以上"]),
        "決め方": {"main": "木", "lethal": "詰みの証明"}.get(p["kind"], "短い木の窓" if (p.get("wread") or {}).get("tree") else "静的な窓"),
        "手札": band(p["my_hand"], [2, 5, 8], ["0〜2 枚", "3〜5 枚", "6〜8 枚", "9 枚以上"]),
        "先攻後攻": "先攻" if (p["turn"] % 2 == 1) == p["own"] and p["turn"] > 0 else ("後攻" if p["turn"] > 0 else "（マリガン）"),
    }
    f["種類 × 時期"] = f["種類"] + " ／ " + f["時期"]
    f["種類 × 有利不利"] = f["種類"] + " ／ " + f["有利不利"]
    f["リーダー"] = g["leaders"][seat_i]
    return f


def leader_colors() -> Dict[str, str]:
    db = D.load_db()
    out = {}
    for cid in D.leader_pool(db):
        c = db.get_card(cid)
        cols = sorted(getattr(x, "name", str(x)) for x in (getattr(c, "colors", ()) or ()))
        out[cid] = "・".join(cols) if cols else "?"
    return out


def cmd_summary(a):
    import numpy as np
    dirs = sorted(d for pat in a.shards for d in glob.glob(pat) if os.path.isdir(d))
    colors = leader_colors()
    games, rows, pop = [], [], []        # rows＝読んだ点（重みつき）・pop＝母集団の点（層 T／S）
    miss = {"void_games": 0, "read_missing": 0, "read_error": 0, "not_repro": 0, "judge_missing": 0, "judge_void": 0}
    for d in dirs:
        reads = {(r["seed"], r["step"]): r for r in read_jsonl(os.path.join(d, "reads.jsonl"))}
        judge = {(r["seed"], r["step"]): r for r in read_jsonl(os.path.join(d, "judge.jsonl"))}
        for g in read_jsonl(os.path.join(d, "games.jsonl")):
            if g.get("void"):
                miss["void_games"] += 1
                continue
            gi = len(games)
            games.append({"seed": g["seed"], "leaders": g["leaders"], "winner": g["winner"], "turns": g["turns"],
                          "n_points": g["n_points"], "n_T": g["n_T"], "n_S": g["n_S"], "sec": g["sec"],
                          "n_commit": sum(1 for p in g["points"] if p["kind"] == "commit" and (p["n_cand"] or 0) >= 2)})
            by = {p["step"]: p for p in g["points"]}
            for p in g["points"]:
                s = stratum(p)
                if s:
                    pop.append((gi, s, features(g, p, colors)))
            for s in ("T", "S"):
                ok = []
                for st in g["pick"][s]:
                    r = reads.get((g["seed"], st))
                    if r is None:
                        miss["read_missing"] += 1
                        continue
                    if r.get("error"):
                        miss["read_error"] += 1
                        continue
                    if not r.get("repro"):
                        miss["not_repro"] += 1
                        continue
                    loss = est = 0.0
                    j = None
                    if r["changed"]:
                        j = judge.get((g["seed"], st))
                        if j is None or j.get("error") or j.get("diff") is None:
                            miss["judge_missing"] += 1
                            continue
                        miss["judge_void"] += sum(1 for x in j["a"] + j["b"] if x is None)
                        loss = float(j["diff"])
                        if r.get("q10_a") is not None and r.get("q10_b") is not None:
                            est = (float(r["q10_b"]) - float(r["q10_a"])) / 2.0
                    ok.append({"gi": gi, "stratum": s, "p": by[st], "r": r, "j": j, "loss": loss, "est": est})
                n_pop = g["n_T"] if s == "T" else g["n_S"]
                for x in ok:
                    x["w"] = n_pop / len(ok)
                    x["f"] = features(g, x["p"], colors)
                rows.extend(ok)
    G = len(games)
    if not G or not rows:
        print("no data")
        return
    denom = 2.0 * G                      # 1 席 1 局あたり
    rng = np.random.default_rng(20261115)
    B = a.boot
    draws = rng.integers(0, G, size=(B, G))
    cnt = np.zeros((B, G))
    for b in range(B):
        cnt[b] = np.bincount(draws[b], minlength=G)

    def per_game(sel, val):
        v = np.zeros(G)
        for x in rows:
            if sel(x):
                v[x["gi"]] += x["w"] * val(x)
        return v

    def stat(v):
        """局ごとの和 v → 1 席 1 局あたりの値と、局の引き直しの 95% 区間。"""
        m = float(v.sum() / denom)
        bs = cnt @ v / denom
        return m, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], bs

    tot_v = per_game(lambda x: True, lambda x: x["loss"])
    tot, tot_ci, tot_bs = stat(tot_v)
    est_v = per_game(lambda x: True, lambda x: x["est"])
    est, est_ci, _ = stat(est_v)

    def cell(sel, popsel):
        n_pop = sum(1 for (_gi, _s, f) in pop if popsel(f))
        xs = [x for x in rows if sel(x)]
        w = sum(x["w"] for x in xs)
        wc = sum(x["w"] for x in xs if x["r"]["changed"])
        wn = sum(x["w"] for x in xs if x["r"]["noise_changed"])
        lv = per_game(sel, lambda x: x["loss"])
        m, ci, bs = stat(lv)
        share_bs = bs / np.where(np.abs(tot_bs) < 1e-12, np.nan, tot_bs)
        ev = per_game(sel, lambda x: x["est"])
        nchg = sum(1 for x in xs if x["r"]["changed"])
        return {"点の数（1 席 1 局）": round(n_pop / denom, 2), "読んだ点": len(xs), "替わった点": nchg,
                "替わる割合": round(wc / w, 4) if w else None,
                "乱数だけで替わる割合": round(wn / w, 4) if w else None,
                "替わった 1 点の損": round(float(lv.sum()) / wc, 4) if wc else None,
                "1 点あたりの損": round(float(lv.sum()) / w, 5) if w else None,
                "損（1 席 1 局）": round(m, 5), "損の 95% 区間": [round(ci[0], 5), round(ci[1], 5)],
                "取り分": round(m / tot, 3) if abs(tot) > 1e-12 else None,
                "取り分の 95% 区間": [round(float(np.nanpercentile(share_bs, 2.5)), 3), round(float(np.nanpercentile(share_bs, 97.5)), 3)],
                "10 倍の読み自身の見立て（1 席 1 局）": round(float(ev.sum() / denom), 5),
                "見立ての取り分": round(float(ev.sum() / denom) / est, 3) if abs(est) > 1e-12 else None}

    axes = ["種類", "時期", "自分のライフ", "相手のライフ", "リーダーの色", "有利不利", "手番の持ち主", "候補の数", "決め方", "手札",
            "先攻後攻", "種類 × 時期", "種類 × 有利不利", "リーダー"]
    tables = {}
    for ax in axes:
        vals = sorted({f[ax] for (_gi, _s, f) in pop})
        t = {v: cell(lambda x, v=v: x["f"][ax] == v, lambda f, v=v: f[ax] == v) for v in vals}
        tables[ax] = dict(sorted(t.items(), key=lambda kv: -(kv[1]["損（1 席 1 局）"])))

    def top(axs, n=10):
        allc = [dict(c, 切り口=ax, 値=v) for ax in axs for v, c in tables[ax].items()]
        allc.sort(key=lambda c: -c["損（1 席 1 局）"])
        return allc[:n]

    # 手の行き先（打った手の種類 → 10 倍の手の種類）
    trans: Dict[str, Dict[str, Any]] = {}
    for x in rows:
        if not x["r"]["changed"]:
            continue
        k = move_kind(x["p"].get("a_box") or x["p"]["move"]) + " → " + move_kind(x["r"]["b"])
        t = trans.setdefault(k, {"替わった点": 0, "w": 0.0, "wl": 0.0})
        t["替わった点"] += 1
        t["w"] += x["w"]
        t["wl"] += x["w"] * x["loss"]
    trans_out = {k: {"替わった点": t["替わった点"], "替わった 1 点の損": round(t["wl"] / t["w"], 4),
                     "損（1 席 1 局）": round(t["wl"] / denom, 5), "取り分": round(t["wl"] / denom / tot, 3) if abs(tot) > 1e-12 else None}
                 for k, t in sorted(trans.items(), key=lambda kv: -kv[1]["wl"])}

    # 点ごとに「10 倍の手が得／損」と言えた数（対の差の 95% 区間が 0 を外す）
    jr = [x for x in rows if x["j"]]
    sig_pos = sum(1 for x in jr if x["j"]["ci"][0] is not None and x["j"]["ci"][0] > 0)
    sig_neg = sum(1 for x in jr if x["j"]["ci"][1] is not None and x["j"]["ci"][1] < 0)
    big = sorted(jr, key=lambda x: -x["loss"])[:30]
    T = [x for x in rows if x["stratum"] == "T"]
    S = [x for x in rows if x["stratum"] == "S"]
    kinds_sorted = list(tables["種類"].values())
    pos_sum = sum(max(0.0, c["損（1 席 1 局）"]) for c in kinds_sorted)
    top3 = sum(sorted((max(0.0, c["損（1 席 1 局）"]) for c in kinds_sorted), reverse=True)[:3])
    cross = list(tables["種類 × 時期"].values())
    cpos = sum(max(0.0, c["損（1 席 1 局）"]) for c in cross)
    ctop3 = sum(sorted((max(0.0, c["損（1 席 1 局）"]) for c in cross), reverse=True)[:3])
    exp = [elo_gain(e) for e in ARENA_ELO]
    summary = {
        "shards": dirs, "games": G, "missing": miss,
        "game_stats": {"turns_mean": round(sum(g["turns"] for g in games) / G, 2),
                       "points_mean": round(sum(g["n_points"] for g in games) / G, 1),
                       "pop_T_mean": round(sum(g["n_T"] for g in games) / G, 1),
                       "pop_S_mean": round(sum(g["n_S"] for g in games) / G, 1),
                       "commit_mean": round(sum(g["n_commit"] for g in games) / G, 1),
                       "p1_win": round(sum(g["winner"] == "p1" for g in games) / G, 3)},
        "sample": {"read_points": len(rows), "T": len(T), "S": len(S),
                   "T_changed": sum(x["r"]["changed"] for x in T), "S_changed": sum(x["r"]["changed"] for x in S),
                   "T_noise_changed": sum(x["r"]["noise_changed"] for x in T),
                   "S_noise_changed": sum(x["r"]["noise_changed"] for x in S),
                   "judged_pairs": len(jr), "pairs_10x_better_sig": sig_pos, "pairs_10x_worse_sig": sig_neg,
                   "cpu10_mean_T": round(sum(x["r"]["cpu10"] for x in T) / max(1, len(T)), 2),
                   "cpu1_mean_T": round(sum(x["r"]["cpu1"] for x in T) / max(1, len(T)), 2),
                   "judge_k": jr[0]["j"].get("k") if jr else None,
                   "judge_sd_mean": round(sum(x["j"]["sd"] or 0 for x in jr) / max(1, len(jr)), 3)},
        "total": {"損（1 席 1 局・勝率）": round(tot, 5), "95% 区間": [round(tot_ci[0], 5), round(tot_ci[1], 5)],
                  "10 倍の読み自身の見立て（1 席 1 局・勝率）": round(est, 5), "見立ての 95% 区間": [round(est_ci[0], 5), round(est_ci[1], 5)],
                  "対戦の +29.9 Elo を勝率に直した値": round(exp[0], 4), "その 95% 区間": [round(exp[1], 4), round(exp[2], 4)],
                  "損 ÷ 対戦の値": round(tot / exp[0], 3)},
        "concentration": {"種類の上位 3 個の取り分（正の損の合計に対して）": round(top3 / pos_sum, 3) if pos_sum else None,
                          "種類 × 時期の上位 3 個の取り分（同）": round(ctop3 / cpos, 3) if cpos else None},
        "top10_single_axis": top(["種類", "時期", "自分のライフ", "相手のライフ", "リーダーの色", "有利不利"]),
        "top10_kind_x_turn": top(["種類 × 時期"]),
        "top10_kind_x_adv": top(["種類 × 有利不利"]),
        "tables": tables, "transitions": trans_out,
        "biggest_points": [{"seed": games[x["gi"]]["seed"], "step": x["p"]["step"], "turn": x["p"]["turn"], "種類": x["f"]["種類"],
                            "diff": x["j"]["diff"], "ci": x["j"]["ci"], "a": x["r"]["desc_a"], "b": x["r"]["desc_b"],
                            "root_v": x["p"].get("root_v"), "my_life": x["p"]["my_life"], "opp_life": x["p"]["opp_life"]} for x in big],
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
        f.write("\n")
    # 点ごとの 1 行（切り口・重み・損）＝報告の表を作り直せる生の記録
    with open(os.path.join(a.out, "points.jsonl"), "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps({"seed": games[x["gi"]]["seed"], "step": x["p"]["step"], "seat": x["p"]["seat"], "turn": x["p"]["turn"],
                                "stratum": x["stratum"], "w": round(x["w"], 4), "changed": x["r"]["changed"],
                                "noise_changed": x["r"]["noise_changed"], "loss": x["loss"], "est": round(x["est"], 4),
                                "ci": x["j"]["ci"] if x["j"] else None, "f": x["f"],
                                "a": x["r"]["desc_a"], "b": x["r"]["desc_b"] if x["r"]["changed"] else None},
                               ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("games", "missing", "game_stats", "sample", "total", "concentration")},
                     ensure_ascii=False, indent=1))
    for c in summary["top10_single_axis"]:
        print(c["切り口"], c["値"], c["点の数（1 席 1 局）"], c["替わる割合"], c["替わった 1 点の損"], c["損（1 席 1 局）"], c["損の 95% 区間"], c["取り分"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play")
    p.add_argument("--games", type=int, default=15)
    p.add_argument("--seed-base", type=int, default=7341000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--net", default=E.DEFAULT_NET)
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("read")
    p.add_argument("--mult", type=int, default=MULT)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("judge")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("summary")
    p.add_argument("--shards", nargs="+", default=[os.path.join(OUT_DIR, "w*")])
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)
    {"play": cmd_play, "read": cmd_read, "judge": cmd_judge, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
