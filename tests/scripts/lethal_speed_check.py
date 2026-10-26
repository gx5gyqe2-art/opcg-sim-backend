"""詰みの証明の**不変量の器**（`docs/search_efficiency_plan.md` §7.25・WP `lethal-speed`）。

§7.21 の打ち直しの監査 300 局（`n1_results/lethal_root/redo/points_3012000_300.jsonl.gz`・両席 r13d・
1 本 × 1,280 回・枝予算なし・`root_reply`・random × synth_roles）の**全点**で、判定の変更の前後で
「証明（win）の集合と証明した手」が同一か（unknown → no は可・win が減るのは不可）を確かめる。

盤面の正本は**記録した手**: `record` が同じ設定で局を打ち直して**打った手を全部**記録し（探索はここだけ）、
`judge` は記録した手を探索なしで当て直して各点の盤面を再現し、`Game.lethal_json`（`force`・配り方の seed は
監査と同じ `seed * 1000 + step`）で判定だけを打ち直す（1 局 1 秒台）。記録 v5 の `hidden` 経由では
`known_by`（自分が知っている札）が失われ、公平な配り方が別物になるので使わない。

  python tests/scripts/lethal_speed_check.py record  --games 300 --workers 3           # 1 回だけ（基準の wheel）
  python tests/scripts/lethal_speed_check.py judge   --tag before                      # 変更の前後で
  python tests/scripts/lethal_speed_check.py compare --a n1_results/lethal_speed/judge_before.jsonl \
                                                     --b n1_results/lethal_speed/judge_after.jsonl
  python tests/scripts/lethal_speed_check.py compare --a n1_results/lethal_root/redo/points_3012000_300.jsonl.gz \
                                                     --b n1_results/lethal_speed/judge_before.jsonl   # 再現の確認
  python tests/scripts/lethal_speed_check.py judge --tag true --true-deal                # 本当の配り方だけ（§7.28）
  python tests/scripts/lethal_speed_check.py judge --tag legacy --lethal-deal-legacy     # 守りの強い配り方の旧い形（§7.33）

`compare` の `--a` には監査の点（`rs` の欄）も渡せる。読み取り専用・テスト無し（理論の道具と同じ扱い）。
"""
import argparse
import gzip
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("OPCG_LOG_SILENT", "1")

from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402

OUT_DIR = os.path.join("n1_results", "lethal_speed")
MAIN = "MAIN_ACTION"
SEED_BASE = 3012000
GAMES = 300


def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def _is_point(game, name: str, turn: int, min_turn: int, max_life: int) -> Optional[int]:
    """監査の点の条件（自分の MAIN・対話なし・`min_turn` 以降・相手のライフ ≤ `max_life`）。相手のライフを返す。"""
    pend = json.loads(game.pending_json())
    if pend.get("action") != MAIN or game.turn_player() != name or game.has_interaction:
        return None
    if turn < min_turn:
        return None
    h = json.loads(game.hidden_json())
    opp = "p2" if name == "p1" else "p1"
    life = len(h["players"][opp]["life"])
    return life if life <= max_life else None


def _judge(game, name: str, seed: int, step: int, max_nodes: Optional[int], true_deal: bool = False,
           deal_legacy: bool = False) -> Dict[str, Any]:
    opts = {"search_seed": seed * 1000 + step, "force": True}
    if deal_legacy:
        opts["lethal_deal_legacy"] = True  # 守りの強い配り方の旧い形（§7.33・測定用）
    if true_deal:
        opts["true_deal"] = True          # 本当の配り方 1 通りだけ（計測用・§7.28 の仕分け）
    if max_nodes:
        opts["max_nodes"] = int(max_nodes)
    return json.loads(game.lethal_json(name, json.dumps(opts)))


def _decks(db, seed: int):
    la, lb = D.leader_pair(db, seed, "random")
    return D.build_pair(db, la, lb, seed, "synth_roles")


# --- record ---------------------------------------------------------------------------------

def record_one(args) -> Dict[str, Any]:
    seed, a = args
    E.engine()
    db = D.load_db()
    kw: Dict[str, Any] = {"budget": a.budget}
    if a.worlds > 1:
        kw["worlds"] = a.worlds
    if a.lethal_proof != "off":
        kw["lethal_proof"] = a.lethal_proof
    spec = E.SeatSpec(a.net, sims=a.sims, **kw)
    p1, p2 = _decks(db, seed)
    moves: List[Dict[str, Any]] = []
    points: List[Dict[str, Any]] = []

    def obs(game, name, turn, step, out, move):
        moves.append({"name": name, "turn": turn, "move": move})
        life = _is_point(game, name, turn, a.min_turn, a.max_life)
        if life is None:
            return
        points.append({"seat": name, "turn": turn, "step": step, "opp_life": life,
                       "rs": _judge(game, name, seed, step, None), "lethal": out.get("lethal"),
                       "kind": out.get("kind")})

    t0 = time.perf_counter()
    try:
        r = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, observer=obs)
        winner, turns, void = r["winner"], r["turns"], r["winner"] is None
    except DR.GameAborted as exc:
        winner, turns, void = None, None, str(exc)
    return {"seed": seed, "winner": winner, "turns": turns, "void": void, "steps": len(moves),
            "sec": round(time.perf_counter() - t0, 1), "moves": moves, "points": points}


def cmd_record(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"record_{a.seed_base}_{a.games}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["seed"])
    seeds = [a.seed_base + i for i in range(a.games) if a.seed_base + i not in done]
    print(f"todo {len(seeds)} (done {len(done)}) -> {path}", flush=True)
    import multiprocessing as mp
    with open(path, "a", encoding="utf-8") as f, mp.get_context("fork").Pool(a.workers) as pool:
        for i, g in enumerate(pool.imap_unordered(record_one, [(s, a) for s in seeds])):
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(seeds)}] seed={g['seed']} steps={g['steps']} pts={len(g['points'])} "
                  f"sec={g['sec']} winner={g['winner']}", flush=True)


# --- judge ----------------------------------------------------------------------------------

def judge_one(args) -> List[Dict[str, Any]]:
    g, a = args
    E.engine()
    db = D.load_db()
    seed = g["seed"]
    p1, p2 = _decks(db, seed)
    game = DR.new_game(seed, p1, p2)
    rows: List[Dict[str, Any]] = []
    for step, m in enumerate(g["moves"]):
        name, turn, move = m["name"], m["turn"], m["move"]
        pend = json.loads(game.pending_json())
        if not pend or pend["player_id"] != name or int(game.turn_count or 0) != turn:
            rows.append({"seed": seed, "step": step, "error": "desync"})
            break
        life = _is_point(game, name, turn, a.min_turn, a.max_life)
        if life is not None:
            rs = _judge(game, name, seed, step, a.max_nodes, a.true_deal, a.lethal_deal_legacy)
            rows.append({"seed": seed, "seat": name, "turn": turn, "step": step, "opp_life": life, "rs": rs})
        try:
            if move.get("kind") == "battle":
                game.apply_battle_action(name, move["action_type"], move.get("card_uuid"))
            else:
                game.apply_game_action(name, move["action_type"], json.dumps(move.get("payload") or {}, ensure_ascii=False))
        except Exception as exc:                      # noqa: BLE001  void の局（記録の最後の手が落ちた手）はそこまで
            rows.append({"seed": seed, "step": step, "error": f"{type(exc).__name__}: {exc}"})
            break
    return rows


def cmd_judge(a):
    src = a.record or os.path.join(a.out, f"record_{a.seed_base}_{a.games}.jsonl")
    games = [json.loads(line) for line in _open(src)]
    if a.limit:
        games = games[: a.limit]
    path = os.path.join(a.out, f"judge_{a.tag}.jsonl")
    import multiprocessing as mp
    t0 = time.perf_counter()
    n = 0
    with open(path, "w", encoding="utf-8") as f, mp.get_context("fork").Pool(a.workers) as pool:
        for rows in pool.imap_unordered(judge_one, [(g, a) for g in games]):
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += len(rows)
            f.flush()
    print(json.dumps({"judge": path, "games": len(games), "points": n, "wall_sec": round(time.perf_counter() - t0, 1),
                      "max_nodes": a.max_nodes}), flush=True)


# --- compare --------------------------------------------------------------------------------

def _load_points(path: str) -> Dict[tuple, Dict[str, Any]]:
    """監査の点（1 行 1 点・`rs` の欄）・`judge` の出力・`record` の出力（1 行 1 局・`points` の中）のどれでも読む。"""
    out = {}
    for line in _open(path):
        d = json.loads(line)
        if "moves" in d and "points" in d:
            for p in d["points"]:
                out[(d["seed"], p["seat"], p["turn"], p["step"])] = p
            continue
        if "game" in d or "rs" not in d:
            continue
        out[(d["seed"], d["seat"], d["turn"], d["step"])] = d
    return out


def _move_key(mv) -> str:
    return json.dumps(mv, sort_keys=True, ensure_ascii=False) if mv is not None else ""


def cmd_compare(a):
    A, B = _load_points(a.a), _load_points(a.b)
    keys = sorted(set(A) & set(B))
    trans: Dict[str, int] = {}
    applies: Dict[str, int] = {}
    win_lost: List[Dict[str, Any]] = []
    move_changed: List[Dict[str, Any]] = []
    win_new: List[Dict[str, Any]] = []
    applies_off_but_win: List[Dict[str, Any]] = []
    nodes = {"a": 0, "b": 0}
    secs = {"a": 0.0, "b": 0.0}
    by_status_nodes: Dict[str, Dict[str, int]] = {}
    for k in keys:
        ra, rb = A[k]["rs"], B[k]["rs"]
        sa, sb = ra["status"], rb["status"]
        trans[f"{sa}->{sb}"] = trans.get(f"{sa}->{sb}", 0) + 1
        ak = f"{ra.get('applies')}->{rb.get('applies')}"
        applies[ak] = applies.get(ak, 0) + 1
        nodes["a"] += ra.get("nodes", 0)
        nodes["b"] += rb.get("nodes", 0)
        secs["a"] += ra.get("sec", 0.0)
        secs["b"] += rb.get("sec", 0.0)
        d = by_status_nodes.setdefault(sa, {"n": 0, "nodes_a": 0, "nodes_b": 0})
        d["n"] += 1
        d["nodes_a"] += ra.get("nodes", 0)
        d["nodes_b"] += rb.get("nodes", 0)
        row = {"key": list(k), "a": {"status": sa, "move": ra.get("move"), "nodes": ra.get("nodes"), "applies": ra.get("applies")},
               "b": {"status": sb, "move": rb.get("move"), "nodes": rb.get("nodes"), "applies": rb.get("applies")},
               "opp_life": A[k].get("opp_life")}
        if sa == "win" and sb != "win":
            win_lost.append(row)
        elif sa == "win" and sb == "win" and _move_key(ra.get("move")) != _move_key(rb.get("move")):
            move_changed.append(row)
        elif sa != "win" and sb == "win":
            win_new.append(row)
        # 発動条件を締めたとき、「今の実装で証明できていた点」が判定に入らなくなるのは不可（§7.25 の 1）
        if sa == "win" and ra.get("applies") and not rb.get("applies"):
            applies_off_but_win.append(row)
    rep = {"a": a.a, "b": a.b, "points_a": len(A), "points_b": len(B), "joined": len(keys),
           "status_transitions": dict(sorted(trans.items())), "applies_transitions": dict(sorted(applies.items())),
           "win_a": sum(1 for k in keys if A[k]["rs"]["status"] == "win"),
           "win_b": sum(1 for k in keys if B[k]["rs"]["status"] == "win"),
           "win_lost": len(win_lost), "win_move_changed": len(move_changed), "win_new": len(win_new),
           "win_but_applies_off": len(applies_off_but_win),
           "nodes": nodes, "nodes_ratio": round(nodes["b"] / max(1, nodes["a"]), 4),
           "sec": {k: round(v, 1) for k, v in secs.items()}, "sec_ratio": round(secs["b"] / max(1e-9, secs["a"]), 4),
           "nodes_by_status_a": by_status_nodes,
           "invariant_ok": not win_lost and not move_changed and not applies_off_but_win}
    if a.verbose:
        rep["win_lost_rows"] = win_lost
        rep["win_move_changed_rows"] = move_changed
        rep["win_new_rows"] = win_new
        rep["win_but_applies_off_rows"] = applies_off_but_win
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if a.save:
        with open(a.save, "w", encoding="utf-8") as f:
            json.dump({**rep, "win_lost_rows": win_lost, "win_move_changed_rows": move_changed,
                       "win_new_rows": win_new, "win_but_applies_off_rows": applies_off_but_win}, f, ensure_ascii=False, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record", "judge"):
        p = sub.add_parser(name)
        p.add_argument("--seed-base", type=int, default=SEED_BASE)
        p.add_argument("--games", type=int, default=GAMES)
        p.add_argument("--out", default=OUT_DIR)
        p.add_argument("--workers", type=int, default=3)
        p.add_argument("--min-turn", type=int, default=4)
        p.add_argument("--max-life", type=int, default=3)
        if name == "record":
            p.add_argument("--sims", type=int, default=E.PLAY_SIMS)
            p.add_argument("--budget", type=int, default=E.PLAY_BUDGET)
            p.add_argument("--worlds", type=int, default=1)
            p.add_argument("--net", default=None)
            p.add_argument("--lethal-proof", default="root_reply", choices=("off", "root", "root_reply"))
        else:
            p.add_argument("--record", default=None, help="record の JSONL（既定は out/record_<base>_<games>.jsonl）")
            p.add_argument("--tag", required=True)
            p.add_argument("--limit", type=int, default=0, help="先頭 N 局だけ（開発中の速い確認）")
            p.add_argument("--max-nodes", type=int, default=None, help="根の節点の上限（§7.25 の 3 の曲線・既定は Rust 既定）")
            p.add_argument("--true-deal", action="store_true",
                           help="本当の配り方 1 通りだけで判定する（§7.28・証明したのに勝てなかった点に本当の守りが在ったかの仕分け）")
            p.add_argument("--lethal-deal-legacy", action="store_true",
                           help="守りの強い配り方を旧い形にする（§7.33・カウンターイベントを払えるかを見ずに積む）")
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--save", default=None)
    a = ap.parse_args(argv)
    {"record": cmd_record, "judge": cmd_judge, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
