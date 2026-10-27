"""強豪の対面記事を物差しに、紫カイドウ対青ロックスで CPU の進行を突き合わせる（WP `matchup-kr`・
`docs/search_efficiency_plan.md` §7.63・台帳 2+466）。

読み取りだけの計測（エンジン・ネット・既定の挙動は変えない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。
土台は `blunder_audit.py`（局の記録の形・`rollout_ship`）と `hold_bias.py`（審判）。

デッキは固定: `tests/fixtures/decks/official_decks_20261022.json` の deck_113（紫カイドウ OP17-058）を **p1**、
deck_112（青ロックス OP17-039）を **p2**。局の番号 i（0〜199）の seed は 4241000＋i、i < 100 は紫カイドウ先攻・
i ≥ 100 は後攻。両席とも r27・出荷の設定（`blunder_audit.ship_spec`）。

1. ``play``: 局を打ち、全ての判断点で打った手・決め方・根の候補の上位 8（訪問・評価・事前分布）を記録する（`games*.jsonl`）。
2. ``trace``: 全局を探索なしで当て直し、手ごとの盤面の要約（手札・場・ドン・ライフ・トラッシュ）と打った手のカードを書く（`trace.jsonl`）。
3. ``items``: 記事の 10 項目の突き合わせと、比べる点（記事の手と CPU の手）の洗い出し（`items.json`・`points_pool.jsonl`）。
4. ``judge``: 比べる点を 1 点 16 通りで打ち比べる（`judge_points*.jsonl`・`--shard i/n`）。
5. ``summary``: 対面の分析と判定の集計（`summary.json`）。

    python tests/scripts/matchup_kr.py play --workers 4 [--shard 0/4]
    python tests/scripts/matchup_kr.py trace
    python tests/scripts/matchup_kr.py items
    python tests/scripts/matchup_kr.py judge --workers 4 [--shard 0/4]
    python tests/scripts/matchup_kr.py summary

再開できる（終えた局・点は飛ばす）。
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
import move_judge as MJ  # noqa: E402

from opcg_sim.loop import driver as DR, engine as E  # noqa: E402

OUT_DIR = os.path.join("n1_results", "matchup_kr")
NET = "opcg_sim/data/learned/nrel_r27.npz"
DECKS = os.path.join("tests", "fixtures", "decks", "official_decks_20261022.json")
KAIDO, ROCKS = "deck_113", "deck_112"
SEED0 = 4241000
N_GAMES = 200
#: 判定の配り方の番号のずらし（blunder_audit 0〜2000 番台・hold_bias 3000・small_play 4000・small_whiff と重ねない）。
OFFSET = 7000
_G = {}


def deck(name):
    x = json.load(open(DECKS, encoding="utf-8"))[name]
    cs = []
    for cid, n in x["cards"].items():
        cs += [cid] * int(n)
    return x["leader"], cs


def game_setup(idx):
    """局の番号 → (seed, first_player)。紫カイドウは常に p1。"""
    return SEED0 + idx, ("p1" if idx < N_GAMES // 2 else "p2")


def new_game(idx):
    seed, first = game_setup(idx)
    return DR.new_game(seed, deck(KAIDO), deck(ROCKS), first)


def replay(g, step):
    game = new_game(g["idx"])
    for s, mv in enumerate(g["moves"]):
        if s == step:
            break
        name = json.loads(game.pending_json())["player_id"]
        BA.apply(game, name, mv)
    return game


def read_jsonl(path):
    return BA.read_jsonl(path)


def read_glob(pattern, keyf):
    rows = {}
    for path in sorted(glob.glob(pattern)):
        for x in read_jsonl(path):
            rows[keyf(x)] = x
    return rows


def load_games(out):
    return read_glob(os.path.join(out, "games*.jsonl"), lambda g: g["idx"])


# ---------------------------------------------------------------- 1. play

def top_rows(out, k=8):
    """根の候補（訪問の多い順）に事前分布を添える。"""
    rows = BA.ranked(out)
    st = out.get("stats") or {}
    P = st.get("P") or []
    if out.get("kind") == "main" and out.get("groups"):
        pr = {int(g["rep"]): sum(float(P[i]) for i in g.get("idxs") or [] if i < len(P)) for g in out["groups"]}
    else:
        pr = {i: float(P[i]) for i in range(len(P))}
    return rows, [{"mv": r["mv"], "n": r["n"], "q": round(r["q"], 4), "p": round(pr.get(r["rep"], 0.0), 4)} for r in rows[:k]]


def play_one(idx):
    E.engine()
    seed, first = game_setup(idx)
    spec = BA.ship_spec(NET)
    pts, moves = [], []

    def obs(game, name, turn, step, out, move):
        pend = json.loads(game.pending_json())
        own = game.turn_player() == name
        rows, top = top_rows(out)
        ch = BA.chosen(out, rows) if rows else None
        pts.append({"step": step, "seat": name, "turn": turn, "own": own, "act": pend.get("action"), "kind": out.get("kind"),
                    "dtype": BA.dtype(pend, own, move), "n_cand": len(rows), "move": move,
                    "a_box": ch["mv"] if ch else None, "a_rank": rows.index(ch) if ch else None, "top": top,
                    "commit_left": len(out.get("commit") or [])})
        moves.append(move)

    t0 = time.perf_counter()
    try:
        r = DR.run_game(seed, {"p1": spec, "p2": spec}, deck(KAIDO), deck(ROCKS), first_player=first, observer=obs)
        winner, turns, void = r["winner"], r["turns"], r["winner"] is None
    except DR.GameAborted as exc:
        winner, turns, void = None, None, str(exc)
    return {"idx": idx, "seed": seed, "first": first, "kaido_first": first == "p1", "winner": winner,
            "kaido_won": None if winner is None else winner == "p1", "turns": turns, "void": void,
            "sec": round(time.perf_counter() - t0, 1), "points": pts, "moves": moves}


def shard_of(a):
    return (int(x) for x in a.shard.split("/")) if a.shard else (0, 1)


def cmd_play(a):
    os.makedirs(a.out, exist_ok=True)
    si, sn = shard_of(a)
    gpath = os.path.join(a.out, f"games_shard{si}.jsonl" if a.shard else "games.jsonl")
    done = set(load_games(a.out))
    # 先攻と後攻が交互に進むように並べる（途中で止めても両方の帯がそろう）
    order = [j for pair in zip(range(N_GAMES // 2), range(N_GAMES // 2, N_GAMES)) for j in pair]
    idxs = [i for n, i in enumerate(order) if n % sn == si and i not in done][:a.games or None]
    print(f"PLAY shard={a.shard} todo {len(idxs)} (done {len(done)}) -> {gpath}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers) as pool:
        for i, g in enumerate(pool.imap_unordered(play_one, idxs, chunksize=1)):
            with open(gpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(idxs)}] idx={g['idx']} first={g['first']} turns={g['turns']} kaido_won={g['kaido_won']} "
                  f"sec={g['sec']} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- 2. trace

def uuid_map(h):
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


def snap(h):
    """盤面の要約（両席）。"""
    out = {}
    for seat in ("p1", "p2"):
        pl = h["players"][seat]
        don = pl.get("don") or {}
        ld = pl["leader"]
        out[seat] = {
            "hand": [c["card_id"] for c in pl["hand"]],
            "field": [[c["card_id"], int(bool(c.get("is_rest"))), int(c.get("attached_don") or 0),
                       int(bool(c.get("is_newly_played"))), c["uuid"]] for c in pl["field"]],
            "leader": [int(bool(ld.get("is_rest"))), int(ld.get("attached_don") or 0)],
            "life": len(pl["life"]), "deck": len(pl["deck"]), "trash": [c.get("card_id") for c in pl.get("trash") or []],
            "stage": [c.get("card_id") for c in (pl.get("stage") if isinstance(pl.get("stage"), list) else
                                                   ([pl["stage"]] if pl.get("stage") else []))],
            "don": [len(don.get("active") or []), len(don.get("rested") or []), len(don.get("attached") or []),
                    len(don.get("deck") or [])]}
    return out


def mv_sum(mv, umap):
    """手の要約（カード番号つき）。"""
    if mv is None:
        return None
    p = mv.get("payload") or {}
    s = {"at": mv.get("action_type")}
    u = p.get("uuid") or mv.get("card_uuid")
    if u:
        s["card"] = umap.get(u, "?")
        s["uuid"] = u
    if p.get("target_ids"):
        s["tgt"] = [umap.get(t, "?") for t in p["target_ids"]]
    if p.get("selected_uuids"):
        s["sel"] = [umap.get(t, "?") for t in p["selected_uuids"]]
    if "accepted" in p:
        s["acc"] = bool(p["accepted"])
    if mv.get("action_type") in ("RESOLVE_EFFECT_SELECTION",) and p.get("index"):
        s["idx"] = p["index"]
    if p.get("don_k"):
        s["k"] = p["don_k"]
    if mv.get("action_type") == "SETUP_BOX" and p.get("base"):
        s["base"] = mv_sum(p["base"], umap)
    return s


def trace_one(g):
    E.engine()
    game = new_game(g["idx"])
    umap = {}
    rows = []
    for s, mv in enumerate(g["moves"]):
        h = json.loads(game.hidden_json())
        umap.update(uuid_map(h))
        pend = json.loads(game.pending_json())
        name = pend["player_id"]
        man = h.get("manager") or {}
        r = {"step": s, "seat": name, "turn": int(game.turn_count or 0), "tp": game.turn_player(), "act": pend.get("action"),
             "msg": (pend.get("message") or "")[:60], "battle": bool(man.get("active_battle")), "st": snap(h),
             "mv": mv_sum(mv, umap)}
        if pend.get("action") in ("MAIN_ACTION",) and name == "p1":
            legal = json.loads(game.legal_json(name))
            r["legal"] = [mv_sum(m, umap) for m in legal if m.get("action_type") in ("PLAY", "ACTIVATE_MAIN", "ATTACK")]
        rows.append(r)
        BA.apply(game, name, mv)
        if game.winner() is not None:
            break
    return {"idx": g["idx"], "kaido_first": g["kaido_first"], "kaido_won": g["kaido_won"], "turns": g["turns"],
            "rows": rows, "umap": umap}


def cmd_trace(a):
    games = load_games(a.out)
    gs = [games[i] for i in sorted(games)]
    path = os.path.join(a.trace_dir or a.out, "trace.jsonl")
    with get_context("fork").Pool(a.workers) as pool, open(path, "w", encoding="utf-8") as f:
        for t in pool.imap(trace_one, gs, chunksize=2):
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print("TRACE games", len(gs), "->", path, flush=True)


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["play", "trace", "items", "judge", "summary"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--cap", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--trace-dir", default=None)
    a = ap.parse_args(argv)
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
