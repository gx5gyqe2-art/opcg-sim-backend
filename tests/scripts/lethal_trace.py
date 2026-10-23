"""詰みの証明の**手順の読み取り器**（`docs/search_efficiency_plan.md` §7.30・WP `lethal-own-hidden`）。

証明（公平な配り方 8 通りの全部で勝てる）したのに**本当の配り方では勝てない**点の原因を調べる。
§7.25 の記録（`n1_results/lethal_speed/record_3012000_300.jsonl.gz`）の手を当て直して各点の盤面を再現し、
`Game.lethal_json` の計測用の欄（`trace`＝配り方ごとの手順・`first`＝根の候補を 1 手に限る・`guide`＝自分の手の
案内・`true_deal`＝本当の配り方 1 通り）で、

- 配り方ごとの**勝ちの手順**（根からの自分の手の列・相手の応手・自分の対話の答え・その判断点までに見えた
  自分の隠れた札〔根で山札・裏向きのライフに在った札〕）と、
- **本当の配り方で止まる応手**（配り方 0 の自分の手を案内に辿り、相手の各応手の後を読み直して「こちらが勝てない
  応手」が在ればそれを採る＝反証）

を並べ、配り方ごとに自分の手順が**どこで分かれるか**（分かれ目が自分の隠れた札を見た直後か）を表にする。

  python tests/scripts/lethal_trace.py targets --base n1_results/lethal_life_deal/judge_base.jsonl.gz \
                                               --true n1_results/lethal_life_deal/judge_true.jsonl.gz   # 的の点の一覧
  python tests/scripts/lethal_trace.py trace --points 3012080:p1:11:109 3012137:p2:12:146 --out n1_results/lethal_own_hidden/trace10.json
  python tests/scripts/lethal_trace.py trace --from-judge n1_results/lethal_own_hidden/judge_base.jsonl.gz --wins \
                                             --workers 4 --out n1_results/lethal_own_hidden/trace_wins.json         # 一般化（全証明）
  python tests/scripts/lethal_trace.py summarize --in n1_results/lethal_own_hidden/trace_wins.json

読み取り専用・テスト無し（理論の道具と同じ扱い）。旧い wheel で読むときは `PYTHONPATH=<その wheel の置き場>` を前に置く。
"""
import argparse
import gzip
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("OPCG_LOG_SILENT", "1")

from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402

RECORD = os.path.join("n1_results", "lethal_speed", "record_3012000_300.jsonl.gz")
OUT_DIR = os.path.join("n1_results", "lethal_own_hidden")


def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def _load_points(path: str) -> Dict[Tuple[int, str, int, int], Dict[str, Any]]:
    out = {}
    for line in _open(path):
        d = json.loads(line)
        if "rs" not in d:
            continue
        out[(d["seed"], d["seat"], d["turn"], d["step"])] = d
    return out


def _parse_point(s: str) -> Tuple[int, str, int, int]:
    a, b, c, d = s.split(":")
    return int(a), b, int(c), int(d)


# --- targets --------------------------------------------------------------------------------

def cmd_targets(a):
    B, T = _load_points(a.base), _load_points(a.true)
    rows = []
    for k in sorted(B):
        rb = B[k]["rs"]
        if rb["status"] != "win":
            continue
        rt = T.get(k, {}).get("rs", {})
        if rt.get("status") == "no":
            rows.append({"point": ":".join(map(str, k)), "opp_life": B[k].get("opp_life"), "move": rb.get("move"),
                         "nodes_base": rb.get("nodes"), "nodes_true": rt.get("nodes")})
    print(json.dumps({"wins": sum(1 for k in B if B[k]["rs"]["status"] == "win"), "targets": len(rows), "rows": rows},
                     ensure_ascii=False, indent=1))


# --- trace ----------------------------------------------------------------------------------

def _replay(db, g, step_to: int):
    seed = g["seed"]
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "synth_roles")
    game = DR.new_game(seed, p1, p2)
    for step, m in enumerate(g["moves"]):
        if step == step_to:
            return game
        name, move = m["name"], m["move"]
        pend = json.loads(game.pending_json())
        if not pend or pend["player_id"] != name or int(game.turn_count or 0) != m["turn"]:
            return None
        if move.get("kind") == "battle":
            game.apply_battle_action(name, move["action_type"], move.get("card_uuid"))
        else:
            game.apply_game_action(name, move["action_type"], json.dumps(move.get("payload") or {}, ensure_ascii=False))
    return None


def _my_moves(line: List[Dict[str, Any]]) -> List[str]:
    return [json.dumps(r["move"], sort_keys=True, ensure_ascii=False) for r in line if r.get("actor") == "me" and "move" in r]


def _analyze(base: Dict[str, Any]) -> Dict[str, Any]:
    """配り方ごとの自分の手の列を配り方 0 と比べ、最初に分かれる番目と、その直前までに見えた自分の隠れた札を出す。"""
    deals = base.get("lines") or []
    if not deals:
        return {"deals": 0}
    seqs = [_my_moves(d["line"]) for d in deals]
    ref = seqs[0]
    diverge = []
    for i, sq in enumerate(seqs[1:], start=1):
        k = next((j for j in range(min(len(ref), len(sq))) if ref[j] != sq[j]), None)
        if k is None and len(ref) != len(sq):
            k = min(len(ref), len(sq))
        if k is None:
            continue
        # 分かれた手（k 番目の自分の手）の時点で見えていた自分の隠れた札（配り方 i の側）
        mine = [r for r in deals[i]["line"] if r.get("actor") == "me"]
        seen_upto = [c for r in mine[: k + 1] for c in r.get("seen", [])]
        diverge.append({"deal": i, "at_my_move": k, "seen_before": seen_upto,
                        "ref": ref[k] if k < len(ref) else None, "other": sq[k] if k < len(sq) else None})
    uniq = len({tuple(s) for s in seqs})
    seen_any = any(c for d in deals for r in d["line"] for c in r.get("seen", []))
    return {"deals": len(deals), "distinct_my_sequences": uniq, "diverging_deals": len(diverge),
            "diverge_after_own_hidden": sum(1 for d in diverge if d["seen_before"]), "seen_any": seen_any,
            "diverge": diverge}


def trace_game(args) -> List[Dict[str, Any]]:
    g, points, a = args
    E.engine()
    db = D.load_db()
    out = []
    for (seed, seat, turn, step) in points:
        game = _replay(db, g, step)
        if game is None:
            out.append({"point": f"{seed}:{seat}:{turn}:{step}", "error": "desync"})
            continue
        opts = {"search_seed": seed * 1000 + step, "force": True, "trace": True}
        t0 = time.perf_counter()
        base = json.loads(game.lethal_json(seat, json.dumps(opts)))
        row: Dict[str, Any] = {"point": f"{seed}:{seat}:{turn}:{step}", "seat": seat, "opp_life": None,
                               "base": {k: base.get(k) for k in ("status", "move", "nodes", "deals", "line_len")},
                               "analysis": _analyze(base)}
        h = json.loads(game.hidden_json())
        opp = "p2" if seat == "p1" else "p1"
        row["opp_life"] = len(h["players"][opp]["life"])
        row["opp_hand"] = [c.get("card_id") for c in h["players"][opp]["hand"]]
        row["my_deck_top"] = [c.get("card_id") for c in h["players"][seat]["deck"][:5]]
        if a.full:
            row["base_lines"] = base.get("lines")
        if base.get("status") == "win" and (a.true or a.points):
            guide = [json.loads(m) for m in _my_moves(base["lines"][0]["line"])] if base.get("lines") else []
            topts = {**opts, "true_deal": True, "first": base["move"], "guide": guide}
            true = json.loads(game.lethal_json(seat, json.dumps(topts)))
            row["true"] = {k: true.get(k) for k in ("status", "nodes")}
            row["true_line"] = (true.get("lines") or [{}])[0].get("line")
        row["sec"] = round(time.perf_counter() - t0, 2)
        out.append(row)
    return out


def cmd_trace(a):
    games = {json.loads(l)["seed"]: json.loads(l) for l in _open(a.record)}
    if a.points:
        pts = [_parse_point(p) for p in a.points]
    else:
        J = _load_points(a.from_judge)
        pts = [k for k in sorted(J) if (not a.wins) or J[k]["rs"]["status"] == "win"]
        if a.limit:
            pts = pts[: a.limit]
    by_seed: Dict[int, List[Tuple[int, str, int, int]]] = {}
    for p in pts:
        by_seed.setdefault(p[0], []).append(p)
    jobs = [(games[s], sorted(v, key=lambda x: x[3]), a) for s, v in sorted(by_seed.items())]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    import multiprocessing as mp
    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(trace_game, jobs)):
            rows.extend(r)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f"[{i + 1}/{len(jobs)} games] points={len(rows)} sec={time.perf_counter() - t0:.0f}", flush=True)
    rows.sort(key=lambda r: _parse_point(r["point"]))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(json.dumps({"out": a.out, "points": len(rows), "wall_sec": round(time.perf_counter() - t0, 1)}))
    if a.points:
        _print_rows(rows)


def _fmt_move(r: Dict[str, Any]) -> str:
    mv = r.get("move") or {}
    at = mv.get("action_type", "?")
    cards = "/".join(r.get("cards") or [])
    extra = ""
    pl = mv.get("payload") or {}
    if at == "RESOLVE_EFFECT_SELECTION":
        extra = f" sel={pl.get('selected_uuids') and len(pl['selected_uuids'])} idx={pl.get('index')} acc={pl.get('accepted')}"
    return f"{at}[{cards}]{extra}"


def _print_rows(rows: List[Dict[str, Any]]):
    for r in rows:
        print("=" * 100)
        print(r["point"], "opp_life", r.get("opp_life"), "base", r.get("base"), "true", r.get("true"))
        print(" opp_hand", r.get("opp_hand"), "my_deck_top", r.get("my_deck_top"))
        print(" analysis", {k: v for k, v in (r.get("analysis") or {}).items() if k != "diverge"})
        for d in (r.get("analysis") or {}).get("diverge", []):
            print("   diverge", d["deal"], "at my move", d["at_my_move"], "seen_before", d["seen_before"])
        for d in r.get("base_lines") or []:
            print(f" deal {d['deal']} {d['result']} nodes={d['nodes']}")
            for x in d["line"]:
                if "end" in x or "stop" in x:
                    print("    ", {k: v for k, v in x.items() if k in ("end", "stop", "actor", "action", "message", "seen")})
                else:
                    msg = f" «{x['message']}»" if x.get("message") else ""
                    seen = f" seen={x['seen']}" if x.get("seen") else ""
                    print(f"     {x['actor']:3} {x['src']:6} {_fmt_move(x)}{msg}{seen}")
        if r.get("true_line") is not None:
            print(" true deal line:")
            for x in r["true_line"]:
                if "end" in x or "stop" in x:
                    print("    ", {k: v for k, v in x.items() if k in ("end", "stop", "actor", "action", "message", "seen")})
                else:
                    msg = f" «{x['message']}»" if x.get("message") else ""
                    seen = f" seen={x['seen']}" if x.get("seen") else ""
                    print(f"     {x['actor']:3} {x['src']:6} {_fmt_move(x)}{msg}{seen}")


def cmd_show(a):
    rows = json.load(open(a.inp, encoding="utf-8"))
    if a.points:
        want = set(a.points)
        rows = [r for r in rows if r["point"] in want]
    _print_rows(rows)


# --- summarize ------------------------------------------------------------------------------

def cmd_summarize(a):
    rows = json.load(open(a.inp, encoding="utf-8"))
    wins = [r for r in rows if r.get("base", {}).get("status") == "win"]
    n = len(wins)
    split = [r for r in wins if r["analysis"].get("distinct_my_sequences", 1) > 1]
    hidden = [r for r in split if r["analysis"].get("diverge_after_own_hidden", 0) > 0]
    hidden_all = [r for r in split if r["analysis"].get("diverge_after_own_hidden", 0) == r["analysis"].get("diverging_deals", -1)]
    seen_any = [r for r in wins if r["analysis"].get("seen_any")]
    by_life: Dict[str, Dict[str, int]] = {}
    for r in wins:
        d = by_life.setdefault(str(r.get("opp_life")), {"win": 0, "split": 0, "split_after_hidden": 0, "seen_any": 0})
        d["win"] += 1
        if r["analysis"].get("distinct_my_sequences", 1) > 1:
            d["split"] += 1
            if r["analysis"].get("diverge_after_own_hidden", 0) > 0:
                d["split_after_hidden"] += 1
        if r["analysis"].get("seen_any"):
            d["seen_any"] += 1
    rep = {"points": len(rows), "wins": n, "errors": sum(1 for r in rows if "error" in r),
           "split_my_sequence": len(split), "split_after_own_hidden_any_deal": len(hidden),
           "split_after_own_hidden_all_diverging_deals": len(hidden_all),
           "own_hidden_seen_in_any_line": len(seen_any), "by_opp_life": dict(sorted(by_life.items())),
           "true_status": {}}
    for r in wins:
        if "true" in r:
            k = r["true"]["status"]
            rep["true_status"][k] = rep["true_status"].get(k, 0) + 1
    if a.verbose:
        rep["split_points"] = [{"point": r["point"], "opp_life": r.get("opp_life"), "analysis": {k: v for k, v in r["analysis"].items() if k != "diverge"}}
                               for r in split]
    print(json.dumps(rep, ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("targets")
    p.add_argument("--base", required=True)
    p.add_argument("--true", required=True)
    p = sub.add_parser("trace")
    p.add_argument("--record", default=RECORD)
    p.add_argument("--points", nargs="*", help="seed:seat:turn:step …（指定すると本当の配り方の反証も読む・手順を全部出す）")
    p.add_argument("--from-judge", default=None, help="judge の出力（点の一覧の元）")
    p.add_argument("--wins", action="store_true", help="--from-judge のうち証明（win）の点だけ")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--true", action="store_true", help="証明した点で本当の配り方の反証も読む（--points では既定）")
    p.add_argument("--full", action="store_true", help="配り方ごとの手順を全部保存する（--points では既定）")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=os.path.join(OUT_DIR, "trace.json"))
    p = sub.add_parser("show")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--points", nargs="*")
    p = sub.add_parser("summarize")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "trace" and a.points:
        a.full = True
        a.true = True
    {"targets": cmd_targets, "trace": cmd_trace, "show": cmd_show, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
