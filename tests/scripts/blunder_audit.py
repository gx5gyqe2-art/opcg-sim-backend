"""大きく損をした手の洗い出し（WP `blunder-audit`・`docs/search_efficiency_plan.md` §7.41・台帳 2+371）。

読み取りだけの計測（エンジン・ネット・既定の挙動は変えない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。
固定の審判で最後まで打つ判定は `move_judge.py`（`rollout`／`paired`／`Game.fork_world`）をそのまま使う。

1. ``play``: 両席 r13d・**出荷の設定**（1,280 回・世界 2 本・枝予算なし・詰みの証明 root＝`commit_audit.ship_spec`）・
   random × real で局を打ち、**全ての判断点**で打った手・決め方（`main`＝木／`window`＝窓の根畳み／`commit`＝続き／
   `lethal`＝証明）・根の候補の訪問の上位 3 手を記録する（`games.jsonl`・局の手の列も）。局ごとに、候補が 2 つ以上の
   判断点から一様に無作為に上限 10 点を引く（`games.jsonl` の `pick`）。
2. ``judge``: 点ごとに局の手の列を当て直して局面を作り、比べる 2 つの手を決める:
   - ``main``: a＝打った束（箱は箱全体）・b＝根の訪問が次に多い束（打った束が最多でなければ最多の束）。
   - ``window``: a＝打った手・b＝根の訪問（同数は Q）で a 以外の最多。
   - ``commit``: 持ち越し無しで読み直す（同じ `search_seed`）。読み直しの手の先頭の原始手が a と違えば b＝読み直しの手、
     同じなら b＝読み直しの根の次点。
   - ``lethal``: 証明 off・持ち越し無しで読み直し、同じ規則で b を取る。
   固定の審判 r13d（160 回・世界 1 本・公平・証明 off）が a と b の先を K＝32 局ずつ最後まで打つ（同じ配り方と乱数で対の差）。
   **差（b − a）≥ 0.20 かつ 95% 区間の下限 > 0 の点＝損の候補**。損の候補だけ、審判 r20 でも打ち直し（同じ K）、
   深い読み（出荷の 4 倍の 5,120 回・世界 4 本・証明 off・seed はずらす）で a と b の根の値と、当てた後の値を書く。
   ``recheck``（候補だけ）: 独立の配り方 48 通りで審判 2 本とも打ち直す（K＝32 の 1 回の判定は偶然でも 3% ほど線を越えるため）。
   ``verify``（候補だけ）: 同じ打ち直しを、**手を打った側の今のターンの残りは出荷の設定が打つ**形で（配り方 24 通り × 審判 2 本）。
   a が箱の 1 段目・詰みの手順の 1 手目のとき、160 回の審判が残りを打てずに a だけが不利になる偏りを除く。
   ``winstab``（窓の点）: 探索の乱数（＝窓の根畳みが引く世界 1 本）だけを替えて 8 回選ばせ、選ぶ手が揺れるかを見る。
3. ``summary``: 損の率（全点・判断の種類別・決め方別・ターン帯・ライフ別・再現・確かめ）・窓の揺れ・損の点の一覧（`summary.json`）。

    python tests/scripts/blunder_audit.py play --games 100 --workers 4
    python tests/scripts/blunder_audit.py judge --workers 4
    python tests/scripts/blunder_audit.py recheck --workers 4
    python tests/scripts/blunder_audit.py verify --workers 4
    python tests/scripts/blunder_audit.py winstab --workers 4
    python tests/scripts/blunder_audit.py summary

再開できる（終えた局・点は飛ばす）。
"""
import argparse
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

from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402
from opcg_sim.loop import record_gen as RG  # noqa: E402
import move_judge as MJ  # noqa: E402  rollout／paired／world_seed

OUT_DIR = os.path.join("n1_results", "blunder_audit")
NET = "opcg_sim/data/learned/nrel_r13d.npz"
REFS = {"r13d": NET, "r20": "vm_nets/nrel_r20.npz"}
#: 深い読み（出荷の 4 倍・世界 4 本・証明 off・`commit_audit.DEEP` と同じ）。
DEEP = {"sims": 5120, "worlds": 4, "budget": 0}
JUDGE_XOR = 0x1D6E00007E570000
#: 損の候補の線（差 b − a）。
LOSS = 0.20
PER_GAME = 10
_G: Dict[str, Any] = {}


def ship_spec(net: Optional[str]) -> "E.SeatSpec":
    """出荷の設定（対戦 API と同じ: `PLAY_SIMS`・`PLAY_BUDGET`・`PLAY_WORLDS`・`PLAY_LETHAL_PROOF`）。"""
    return E.SeatSpec(net, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=E.PLAY_WORLDS,
                      lethal_proof=E.PLAY_LETHAL_PROOF)


def key(mv: Optional[Dict[str, Any]]) -> Optional[str]:
    """手の同一性（原始手の署名＋付与の枚数）。"""
    return None if mv is None else json.dumps([RG.move_sig(mv), RG._don_k(mv)], ensure_ascii=False)


def first_primitive(mv: Dict[str, Any]) -> Dict[str, Any]:
    """箱 → 実対局へ出す先頭の原始手（`commit_audit.first_primitive` と同じ）。"""
    at = mv.get("action_type")
    p = mv.get("payload") or {}
    if at == "SETUP_BOX":
        return p.get("base") or mv
    if at != "DON_BOX":
        return mv
    k = int(p.get("don_k") or 0)
    if k <= 0 and p.get("target_ids"):
        return {"kind": "game", "action_type": "ATTACK", "payload": {"uuid": p.get("uuid"), "target_ids": p["target_ids"]}}
    return {"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": p.get("uuid")}}


def pkey(mv):
    return None if mv is None else json.dumps(RG.move_sig(first_primitive(mv)), ensure_ascii=False)


def desc(game, mv):
    if mv is None:
        return None
    try:
        return json.dumps(json.loads(game.describe_move_json(json.dumps(mv, ensure_ascii=False))), ensure_ascii=False)[:300]
    except Exception:  # noqa: BLE001
        return json.dumps(mv, ensure_ascii=False)[:300]


def apply(game, actor, mv):
    if mv.get("kind") == "battle":
        game.apply_battle_action(actor, mv["action_type"], mv.get("card_uuid"))
    else:
        game.apply_game_action(actor, mv["action_type"], json.dumps(mv.get("payload") or {}, ensure_ascii=False))


def ranked(out) -> List[Dict[str, Any]]:
    """decide の出力 → 根の候補（訪問の多い順・同数は Q）。main は束（代表の合法手＝箱）、window は合法手。"""
    legal = (out.get("stats") or {}).get("legal") or []
    if out.get("kind") == "main" and out.get("groups"):
        rows = [{"mv": legal[g["rep"]], "n": float(g["n"]), "q": float(g["q"]), "rep": int(g["rep"])} for g in out["groups"]]
    else:
        st = out.get("stats") or {}
        ns, qs = st.get("N") or [], st.get("Q") or []
        rows = [{"mv": m, "n": float(ns[i]) if i < len(ns) else 0.0, "q": float(qs[i]) if i < len(qs) else 0.0, "rep": i}
                for i, m in enumerate(legal)]
    rows.sort(key=lambda r: (-r["n"], -r["q"]))
    return rows


def chosen(out, rows):
    """ranked の中で打った手の行（main は (sig, k) の束・それ以外は手そのもの）。"""
    if out.get("kind") == "main":
        sig, k = out.get("sig"), out.get("k")
        for r in rows:
            if RG.move_sig(r["mv"]) == sig and RG._don_k(r["mv"]) == k:
                return r
        return None
    mk = key(out.get("move"))
    return next((r for r in rows if key(r["mv"]) == mk), None)


def dtype(pend, own, mv) -> str:
    """判断の種類（集計用）。"""
    act = pend.get("action")
    at = (mv or {}).get("action_type")
    if act == "SELECT_COUNTER":
        return "counter"
    if act == "SELECT_BLOCKER":
        return "blocker"
    if act in ("MULLIGAN", "KEEP_HAND") or at in ("MULLIGAN", "KEEP_HAND"):
        return "mulligan"
    if act in ("MAIN", "MAIN_ACTION") or act is None:
        if at in ("DON_BOX", "ATTACH_DON"):
            return "main:attach"
        if at == "SETUP_BOX":
            return "main:setup_box"
        return "main:" + str(at).lower()
    return ("own_dialog:" if own else "opp_dialog:") + str(act)


# ---------------------------------------------------------------- 1. play

def play_one(args):
    seed, a = args
    E.engine()
    db = D.load_db()
    spec = ship_spec(a.net)
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "real")
    pts, moves = [], []
    src = {"p1": None, "p2": None}

    def obs(game, name, turn, step, out, move):
        pend = json.loads(game.pending_json())
        own = game.turn_player() == name
        kind = out.get("kind")
        rows = ranked(out)
        if kind in ("main", "window"):
            n_cand = len(rows)
        else:
            try:
                n_cand = len(json.loads(game.legal_json(name)))
            except Exception:  # noqa: BLE001
                n_cand = None
        h = json.loads(game.hidden_json())
        opp = "p2" if name == "p1" else "p1"
        ch = chosen(out, rows) if rows else None
        rec = {"step": step, "seat": name, "turn": turn, "own": own, "act": pend.get("action"), "kind": kind,
               "dtype": dtype(pend, own, move), "n_cand": n_cand, "move": move,
               "a_box": ch["mv"] if ch else None, "a_rank": rows.index(ch) if ch else None,
               "top": [{"mv": r["mv"], "n": r["n"], "q": round(r["q"], 4)} for r in rows[:3 if kind == "main" else 8]],
               "commit_left": len(out.get("commit") or []), "commit_src": src[name] if kind == "commit" else None,
               "my_life": len(h["players"][name]["life"]), "opp_life": len(h["players"][opp]["life"]),
               "my_hand": len(h["players"][name]["hand"]), "opp_hand": len(h["players"][opp]["hand"])}
        if kind != "commit":
            src[name] = (("window" if kind == "window" else (out.get("sig") or [None])[0]) if out.get("commit") else None)
        pts.append(rec)
        moves.append(move)

    t0 = time.perf_counter()
    try:
        r = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, observer=obs)
        winner, turns, void = r["winner"], r["turns"], r["winner"] is None
    except DR.GameAborted as exc:
        winner, turns, void = None, None, str(exc)
    elig = [p["step"] for p in pts if (p["n_cand"] or 0) >= 2]
    rng = random.Random(seed * 7 + 1)
    pick = sorted(rng.sample(elig, min(PER_GAME, len(elig))))
    return {"seed": seed, "leaders": [la, lb], "winner": winner, "turns": turns, "void": void,
            "sec": round(time.perf_counter() - t0, 1), "n_points": len(pts), "n_elig": len(elig), "pick": pick,
            "points": pts, "moves": moves}


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def cmd_play(a):
    os.makedirs(a.out, exist_ok=True)
    gpath = os.path.join(a.out, "games.jsonl")
    done = {g["seed"] for g in read_jsonl(gpath)}
    seeds = [a.seed_base + i for i in range(a.games) if a.seed_base + i not in done]
    print(f"PLAY todo {len(seeds)} (done {len(done)})", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers) as pool:
        for i, g in enumerate(pool.imap_unordered(play_one, [(s, a) for s in seeds], chunksize=1)):
            with open(gpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(seeds)}] seed={g['seed']} pts={g['n_points']} elig={g['n_elig']} pick={len(g['pick'])} "
                  f"sec={g['sec']} winner={g['winner']} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- 2. judge

def _init_judge(k):
    E.engine()
    _G["net"] = E.load_net(NET)
    _G["ship"] = ship_spec(NET)
    _G["refs"] = {n: E.SeatSpec(p, sims=MJ.REFEREE_SIMS) for n, p in REFS.items()}
    _G["db"] = D.load_db()
    _G["k"] = int(k)


def replay(g, step):
    db = _G["db"]
    p1, p2 = D.build_pair(db, g["leaders"][0], g["leaders"][1], g["seed"], "real")
    game = DR.new_game(g["seed"], p1, p2)
    for s, mv in enumerate(g["moves"]):
        if s == step:
            break
        name = json.loads(game.pending_json())["player_id"]
        apply(game, name, mv)
    return game


def pick_b(game, p):
    """比べる手 b（と a）を決める。戻り値 (a, b, info)。"""
    name, turn, seed = p["seat"], p["turn"], p["seed"]
    kind = p["kind"]
    info = {}
    if kind == "main":
        a = p["a_box"] or p["move"]
        others = [t for t in p["top"] if key(t["mv"]) != key(a)]
        return a, (others[0]["mv"] if others else None), {"b_src": "runner_up" if p["a_rank"] == 0 else "top"}
    if kind == "window":
        # 同じカードの別の 1 枚（手札の同名のカウンター等）は同じ手として飛ばす（記述＝カード番号で比べる）
        a = p["move"]
        da = desc(game, a)
        others = [t for t in p["top"] if key(t["mv"]) != key(a) and desc(game, t["mv"]) != da]
        return a, (others[0]["mv"] if others else None), {"b_src": "runner_up" if p["a_rank"] == 0 else "top"}
    # commit／lethal: 持ち越し無しで読み直す（lethal は証明 off で）
    a = p["move"]
    opts = json.loads(_G["ship"].decide_opts(seed, turn, name, None))
    if kind == "lethal":
        opts["lethal_proof"] = "off"
    fresh = json.loads(game.fork().decide(name, json.dumps(opts)))
    rows = ranked(fresh)
    fm = fresh.get("move")
    fch = chosen(fresh, rows) if rows else None
    fb = fch["mv"] if fch else fm
    info = {"fresh_kind": fresh.get("kind"), "fresh": fb, "fresh_same": pkey(fb) == pkey(a)}
    if fb is not None and pkey(fb) != pkey(a) and desc(game, fb) != desc(game, a):
        info["b_src"] = "fresh"
        return a, fb, info
    da = desc(game, a)
    others = [r for r in rows if pkey(r["mv"]) != pkey(a) and desc(game, r["mv"]) != da]
    info["b_src"] = "fresh_runner_up"
    return a, (others[0]["mv"] if others else None), info


def deep_read(game, name, ss, a, b):
    """深い読み: 根（main なら束の (n, q)、window なら手の (N, Q)）と、各手を当てた後の値。"""
    def dd(g, who, j=0):
        opts = {"net": _G["net"], "search_seed": (ss ^ JUDGE_XOR ^ j) & 0xFFFFFFFFFFFFFFFF, **DEEP}
        return json.loads(g.decide(who, json.dumps(opts)))
    o = dd(game.fork(), name)
    rows = ranked(o)
    tot = sum(r["n"] for r in rows) or 1.0
    res = {"kind": o.get("kind"), "best": desc(game, o.get("move")), "best_pkey_is_a": pkey(o.get("move")) == pkey(a),
           "best_pkey_is_b": pkey(o.get("move")) == pkey(b)}
    for tag, mv in (("a", a), ("b", b)):
        cand = [r for r in rows if key(r["mv"]) == key(mv)] or [r for r in rows if pkey(r["mv"]) == pkey(mv)]
        res[tag] = ({"n": round(max(cand, key=lambda r: r["n"])["n"] / tot, 4), "q": round(max(cand, key=lambda r: r["n"])["q"], 4),
                     "exact": key(cand[0]["mv"]) == key(mv)} if cand else None)
    # 当てた後の値（手を打った側から見た・深い読み）
    for tag, mv in (("a", a), ("b", b)):
        g = game.fork()
        try:
            g.apply_search_move(name, json.dumps(mv, ensure_ascii=False))
            w = g.winner()
            if w is not None:
                res["child_" + tag] = 1.0 if w == name else -1.0
                continue
            who = json.loads(g.pending_json())["player_id"]
            c = dd(g, who, 1)
            rs = ranked(c)
            tt = sum(r["n"] for r in rs)
            if tt > 0:
                v = sum(r["n"] * r["q"] for r in rs) / tt
            else:
                v = max((r["q"] for r in rs), default=None) if rs and any(r["n"] > 0 for r in rs) else None
            res["child_" + tag] = None if v is None else round(v if who == name else -v, 4)
        except Exception as exc:  # noqa: BLE001
            res["child_" + tag] = None
            res["child_err_" + tag] = f"{type(exc).__name__}: {exc}"[:120]
    return res


def board(game, me):
    h = json.loads(game.hidden_json())
    out = {}
    for seat in ("p1", "p2"):
        pl = h["players"][seat]
        don = pl.get("don") or {}

        def cs(c):
            return f"{c['card_id']}{'(R)' if c.get('is_rest') else ''}{'+D' + str(c['attached_don']) if c.get('attached_don') else ''}"
        out["me" if seat == me else "opp"] = {
            "leader": cs(pl["leader"]), "field": [cs(c) for c in pl["field"]], "hand": [c["card_id"] for c in pl["hand"]],
            "life": len(pl["life"]), "deck": len(pl["deck"]), "trash": len(pl.get("trash") or []),
            "don_active": len(don.get("active") or []) if isinstance(don.get("active"), list) else don.get("active"),
            "don_rested": len(don.get("rested") or []) if isinstance(don.get("rested"), list) else don.get("rested")}
    return out


def refjudge(game, who, seed, step, ma, mb, spec, K):
    wa, wb, errs = [], [], {}
    for i in range(K):
        ws = MJ.world_seed(seed, step, i)
        ra, ea = MJ.rollout(game, who, ma, ws, spec)
        rb, eb = MJ.rollout(game, who, mb, ws, spec)
        wa.append(ra)
        wb.append(rb)
        for e in (ea, eb):
            if e:
                errs[e] = errs.get(e, 0) + 1
    return dict(MJ.paired(wa, wb), a=wa, b=wb, void=sum(x is None for x in wa + wb), errs=errs or None)


def judge_point(job):
    g, p = job
    t0 = time.time()
    seed, step, name = g["seed"], p["step"], p["seat"]
    p = dict(p, seed=seed)
    row = {"seed": seed, "step": step, "seat": name, "turn": p["turn"], "kind": p["kind"], "dtype": p["dtype"],
           "own": p["own"], "act": p["act"], "n_cand": p["n_cand"], "commit_src": p.get("commit_src"),
           "my_life": p["my_life"], "opp_life": p["opp_life"], "my_hand": p["my_hand"], "opp_hand": p["opp_hand"],
           "ok": False, "why": None}
    try:
        game = replay(g, step)
    except Exception as exc:  # noqa: BLE001
        row["why"] = f"replay: {type(exc).__name__}"
        return row
    pend = json.loads(game.pending_json())
    if pend.get("player_id") != name or int(game.turn_count or 0) != p["turn"]:
        row["why"] = "seat/turn"
        return row
    try:
        a, b, info = pick_b(game, p)
    except Exception as exc:  # noqa: BLE001
        row["why"] = f"pick_b: {type(exc).__name__}: {exc}"[:160]
        return row
    row.update(info)
    if b is None:
        row["why"] = "no_b"
        return row
    row["a"], row["b"] = a, b
    row["a_desc"], row["b_desc"] = desc(game, a), desc(game, b)
    row["top"] = [{"d": desc(game, t["mv"]), "n": t["n"], "q": t["q"]} for t in p["top"]]
    row["sec_prep"] = round(time.time() - t0, 1)
    K = _G["k"]
    row["ref"] = {"r13d": refjudge(game, name, seed, step, a, b, _G["refs"]["r13d"], K)}
    r0 = row["ref"]["r13d"]
    row["cand"] = bool(r0["diff"] is not None and r0["diff"] >= LOSS and r0["ci"][0] is not None and r0["ci"][0] > 0)
    if row["cand"]:
        row["ref"]["r20"] = refjudge(game, name, seed, step, a, b, _G["refs"]["r20"], K)
        r1 = row["ref"]["r20"]
        row["confirmed"] = bool(r1["diff"] is not None and r1["diff"] > 0)
        row["confirmed_strict"] = bool(r1["diff"] is not None and r1["ci"][0] is not None and r1["ci"][0] > 0)
        try:
            row["deep"] = deep_read(game, name, E.search_seed(seed, p["turn"], name), a, b)
        except Exception as exc:  # noqa: BLE001
            row["deep"] = {"err": f"{type(exc).__name__}: {exc}"[:160]}
        row["board"] = board(game, name)
        row["recent"] = g["moves"][max(0, step - 6):step]
    row["ok"] = True
    row["sec"] = round(time.time() - t0, 1)
    return row


def cmd_judge(a):
    games = {g["seed"]: g for g in read_jsonl(os.path.join(a.out, "games.jsonl"))}
    rpath = os.path.join(a.out, "points.jsonl")
    done = {(x["seed"], x["step"]) for x in read_jsonl(rpath)}
    jobs = []
    for s in sorted(games):
        g = games[s]
        if g["void"] and g["winner"] is None and not g["points"]:
            continue
        light = {k: v for k, v in g.items() if k != "points"}
        by = {p["step"]: p for p in g["points"]}
        for st in g["pick"]:
            if (s, st) not in done:
                jobs.append((light, by[st]))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"JUDGE todo {len(jobs)} (done {len(done)}) k={a.k}", flush=True)
    t0 = time.time()
    n = 0
    with get_context("fork").Pool(a.workers, initializer=_init_judge, initargs=(a.k,)) as pool:
        for row in pool.imap_unordered(judge_point, jobs, chunksize=1):
            row["k"] = a.k
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            r0 = (row.get("ref") or {}).get("r13d") or {}
            r1 = (row.get("ref") or {}).get("r20") or {}
            print(f"JUDGE seed={row['seed']} step={row['step']} {row['kind']} {row['dtype']} ok={row['ok']} why={row['why']} "
                  f"d13={r0.get('diff')} ci={r0.get('ci')} d20={r1.get('diff')} sec={row.get('sec')} "
                  f"({n}/{len(jobs)} {time.time() - t0:.0f}s)", flush=True)
    print("JUDGE_DONE", flush=True)


# ---------------------------------------------------------------- 2b. recheck

#: 打ち直しの配り方の番号のずらし（judge の 0..K-1 と重ねない＝独立の配り方）。
RECHECK_OFFSET = 1000


def recheck_point(job):
    g, row = job
    t0 = time.time()
    seed, step, name = row["seed"], row["step"], row["seat"]
    out = {"seed": seed, "step": step}
    game = replay(g, step)
    K = _G["k"]
    for rn in ("r13d", "r20"):
        wa, wb = [], []
        for i in range(K):
            ws = MJ.world_seed(seed, step, RECHECK_OFFSET + i)
            wa.append(MJ.rollout(game, name, row["a"], ws, _G["refs"][rn])[0])
            wb.append(MJ.rollout(game, name, row["b"], ws, _G["refs"][rn])[0])
        out[rn] = dict(MJ.paired(wa, wb), a=wa, b=wb)
    out["sec"] = round(time.time() - t0, 1)
    return out


def cmd_recheck(a):
    """損の候補を、独立の配り方 K 通りで審判 2 本とも打ち直す（K＝32 の 1 回の判定は偶然でも 3% ほど線を越えるため）。"""
    games = {g["seed"]: g for g in read_jsonl(os.path.join(a.out, "games.jsonl"))}
    rows = [x for x in read_jsonl(os.path.join(a.out, "points.jsonl")) if x.get("cand")]
    rpath = os.path.join(a.out, "recheck.jsonl")
    done = {(x["seed"], x["step"]) for x in read_jsonl(rpath)}
    jobs = [({k: v for k, v in games[x["seed"]].items() if k != "points"}, x) for x in rows if (x["seed"], x["step"]) not in done]
    print(f"RECHECK todo {len(jobs)} (done {len(done)}) k={a.k}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init_judge, initargs=(a.k,)) as pool:
        for i, r in enumerate(pool.imap_unordered(recheck_point, jobs, chunksize=1)):
            r["k"] = a.k
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"RECHECK seed={r['seed']} step={r['step']} r13d={r['r13d']['diff']} {r['r13d']['ci']} "
                  f"r20={r['r20']['diff']} {r['r20']['ci']} sec={r['sec']} ({i + 1}/{len(jobs)} {time.time() - t0:.0f}s)", flush=True)
    print("RECHECK_DONE", flush=True)


# ---------------------------------------------------------------- 2c. verify

#: 確かめの配り方の番号のずらし（judge・recheck と重ねない）。
VERIFY_OFFSET = 2000


def rollout_ship(game, who, move, wseed, ref, ship, max_steps=MJ.MAX_STEPS):
    """`MJ.rollout` と同じだが、**手を打った側の今のターンの残り**（そのターン中の自分の判断）は出荷の設定が打つ。
    審判の弱い打ち継ぎ（箱の 1 段目だけを当てた後に続きを打てない等）で a／b のどちらかが不利になるのを除く。"""
    g = game.fork_world(who, "fair", int(wseed))
    t0 = int(game.turn_count or 0)
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
            use_ship = name == who and turn == t0
            spec = ship if use_ship else ref
            c = carries.setdefault((name, use_ship), DR.Carry())
            out = json.loads(g.decide(name, spec.decide_opts(int(wseed), turn, name, c.get(turn, name))))
            c.put(out)
            mv = out.get("move")
            if mv is None:
                break
            apply(g, name, mv)
            steps += 1
    except Exception as exc:  # noqa: BLE001
        return None, "%s: %s" % (type(exc).__name__, str(exc)[:120])
    w = g.winner()
    if w is None:
        return None, "no_winner"
    return (1 if w == who else 0), None


def verify_point(job):
    g, row = job
    t0 = time.time()
    seed, step, name = row["seed"], row["step"], row["seat"]
    out = {"seed": seed, "step": step}
    game = replay(g, step)
    K = _G["k"]
    for rn in ("r13d", "r20"):
        wa, wb = [], []
        for i in range(K):
            ws = MJ.world_seed(seed, step, VERIFY_OFFSET + i)
            wa.append(rollout_ship(game, name, row["a"], ws, _G["refs"][rn], _G["ship"])[0])
            wb.append(rollout_ship(game, name, row["b"], ws, _G["refs"][rn], _G["ship"])[0])
        out[rn] = dict(MJ.paired(wa, wb), a=wa, b=wb)
    out["sec"] = round(time.time() - t0, 1)
    return out


def cmd_verify(a):
    """損の候補を、今のターンの残りを出荷の設定が打ち継ぐ形で、独立の配り方 K 通り・審判 2 本で打ち直す。"""
    games = {g["seed"]: g for g in read_jsonl(os.path.join(a.out, "games.jsonl"))}
    rows = [x for x in read_jsonl(os.path.join(a.out, "points.jsonl")) if x.get("cand")]
    rpath = os.path.join(a.out, "verify.jsonl")
    done = {(x["seed"], x["step"]) for x in read_jsonl(rpath)}
    jobs = [({k: v for k, v in games[x["seed"]].items() if k != "points"}, x) for x in rows if (x["seed"], x["step"]) not in done]
    print(f"VERIFY todo {len(jobs)} (done {len(done)}) k={a.k}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init_judge, initargs=(a.k,)) as pool:
        for i, r in enumerate(pool.imap_unordered(verify_point, jobs, chunksize=1)):
            r["k"] = a.k
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"VERIFY seed={r['seed']} step={r['step']} r13d={r['r13d']['diff']} {r['r13d']['ci']} "
                  f"r20={r['r20']['diff']} {r['r20']['ci']} sec={r['sec']} ({i + 1}/{len(jobs)} {time.time() - t0:.0f}s)", flush=True)
    print("VERIFY_DONE", flush=True)


# ---------------------------------------------------------------- 2d. window stability

def winstab_point(job):
    """窓の点で、探索の乱数（＝窓の根畳みが引く世界 1 本）だけを替えて N 回選ばせ、選ぶ手が揺れるかを見る。"""
    g, r, n = job
    game = replay(g, r["step"])
    who = r["seat"]
    opts = json.loads(_G["ship"].decide_opts(r["seed"], r["turn"], who, None))
    picks, qs = [], set()
    for j in range(n):
        o = dict(opts, search_seed=(opts["search_seed"] + 7919 * j) & 0xFFFFFFFFFFFFFFFF)
        d = json.loads(game.fork().decide(who, json.dumps(o)))
        picks.append(desc(game, d["move"]))
        qs.add(json.dumps([round(x, 4) for x in d["stats"]["Q"]]))
    c = {}
    for x in picks:
        c[x] = c.get(x, 0) + 1
    return {"seed": r["seed"], "step": r["step"], "dtype": r["dtype"], "my_life": r["my_life"], "a": r["a_desc"],
            "b": r["b_desc"], "picks": c, "a_share": c.get(r["a_desc"], 0) / n, "b_share": c.get(r["b_desc"], 0) / n,
            "q_sets": len(qs), "d13": r["ref"]["r13d"]["diff"], "ci13": r["ref"]["r13d"]["ci"], "cand": r["cand"]}


def cmd_winstab(a):
    games = {g["seed"]: g for g in read_jsonl(os.path.join(a.out, "games.jsonl"))}
    rows = [x for x in read_jsonl(os.path.join(a.out, "points.jsonl")) if x["ok"] and x["kind"] == "window"]
    jobs = [({k: v for k, v in games[x["seed"]].items() if k != "points"}, x, a.n) for x in rows]
    with get_context("fork").Pool(a.workers, initializer=_init_judge, initargs=(1,)) as pool, \
            open(os.path.join(a.out, "window_stability.jsonl"), "w", encoding="utf-8") as f:
        for r in pool.imap_unordered(winstab_point, jobs, chunksize=1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("WINSTAB_DONE", len(jobs), flush=True)


# ---------------------------------------------------------------- 3. summary

def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def rate(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci": wilson(k, n)}


def bucket_turn(t):
    return "t1-2" if t <= 2 else "t3-4" if t <= 4 else "t5-6" if t <= 6 else "t7-8" if t <= 8 else "t9+"


def group_rates(rows, keyf):
    out = {}
    for x in rows:
        k = keyf(x)
        out.setdefault(k, []).append(x)
    return {k: {"cand": rate(sum(x["cand"] for x in v), len(v)),
                "confirmed": rate(sum(bool(x.get("confirmed")) for x in v), len(v)),
                "replicated": rate(sum(bool(x.get("replicated")) for x in v), len(v)),
                "verified": rate(sum(bool(x.get("verified")) for x in v), len(v)),
                "mean_diff": round(sum(x["ref"]["r13d"]["diff"] or 0 for x in v) / len(v), 4)}
            for k, v in sorted(out.items(), key=lambda kv: -len(kv[1]))}


def cmd_summary(a):
    games = read_jsonl(os.path.join(a.out, "games.jsonl"))
    rows = read_jsonl(os.path.join(a.out, "points.jsonl"))
    ok = [x for x in rows if x["ok"] and x["ref"]["r13d"]["diff"] is not None]
    # 打ち直し（独立の配り方・審判 2 本）: 両審判の対をまとめた差の区間が 0 を外れ、審判 2 本とも差が正＝再現した損
    rc = {(x["seed"], x["step"]): x for x in read_jsonl(os.path.join(a.out, "recheck.jsonl"))}
    for x in ok:
        r = rc.get((x["seed"], x["step"]))
        if not x["cand"] or r is None:
            continue
        pool = MJ.paired(r["r13d"]["a"] + r["r20"]["a"], r["r13d"]["b"] + r["r20"]["b"])
        x["recheck"] = {"r13d": r["r13d"]["diff"], "r20": r["r20"]["diff"], "pool": pool["diff"], "pool_ci": pool["ci"]}
        x["replicated"] = bool(pool["ci"][0] is not None and pool["ci"][0] > 0
                               and r["r13d"]["diff"] > 0 and r["r20"]["diff"] > 0)
    # 確かめ（今のターンの残りを出荷の設定が打ち継ぐ・独立の配り方・審判 2 本）: 同じ規則で通った点＝確かめ済みの損
    vf = {(x["seed"], x["step"]): x for x in read_jsonl(os.path.join(a.out, "verify.jsonl"))}
    for x in ok:
        r = vf.get((x["seed"], x["step"]))
        if not x["cand"] or r is None:
            continue
        pool = MJ.paired(r["r13d"]["a"] + r["r20"]["a"], r["r13d"]["b"] + r["r20"]["b"])
        x["verify"] = {"r13d": r["r13d"]["diff"], "r20": r["r20"]["diff"], "pool": pool["diff"], "pool_ci": pool["ci"]}
        x["verified"] = bool(pool["ci"][0] is not None and pool["ci"][0] > 0
                             and r["r13d"]["diff"] > 0 and r["r20"]["diff"] > 0)
    s = {"games": len(games), "void": sum(1 for g in games if g["winner"] is None),
         "decisions": sum(g["n_points"] for g in games), "eligible": sum(g["n_elig"] for g in games),
         "picked": sum(len(g["pick"]) for g in games), "rows": len(rows), "ok": len(ok),
         "not_ok": {}, "k": rows[0]["k"] if rows else None, "loss_line": LOSS}
    for x in rows:
        if not x["ok"]:
            s["not_ok"][x["why"]] = s["not_ok"].get(x["why"], 0) + 1
    # 全判断点の決め方・種類（母集団）
    pop_kind, pop_dtype = {}, {}
    for g in games:
        for p in g["points"]:
            if (p["n_cand"] or 0) >= 2:
                pop_kind[p["kind"]] = pop_kind.get(p["kind"], 0) + 1
                pop_dtype[p["dtype"]] = pop_dtype.get(p["dtype"], 0) + 1
    s["population_eligible"] = {"kind": pop_kind, "dtype": dict(sorted(pop_dtype.items(), key=lambda kv: -kv[1]))}
    s["sec_per_point"] = round(sum(x["sec"] for x in ok) / len(ok), 1) if ok else None
    s["sec_per_game"] = round(sum(g["sec"] for g in games) / len(games), 1) if games else None
    s["all"] = {"cand": rate(sum(x["cand"] for x in ok), len(ok)),
                "confirmed": rate(sum(bool(x.get("confirmed")) for x in ok), len(ok)),
                "confirmed_strict": rate(sum(bool(x.get("confirmed_strict")) for x in ok), len(ok)),
                "rechecked": sum(1 for x in ok if "recheck" in x),
                "replicated": rate(sum(bool(x.get("replicated")) for x in ok), len(ok)),
                "verified": rate(sum(bool(x.get("verified")) for x in ok), len(ok)),
                "a_better_mirror": rate(sum(1 for x in ok if x["ref"]["r13d"]["diff"] <= -LOSS and x["ref"]["r13d"]["ci"][1] < 0), len(ok)),
                "mean_diff": round(sum(x["ref"]["r13d"]["diff"] for x in ok) / len(ok), 4) if ok else None}
    diffs = [x["ref"]["r13d"]["diff"] for x in ok]
    s["hist"] = MJ._hist(diffs, [-0.5, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.5])
    s["by_kind"] = group_rates(ok, lambda x: x["kind"])
    s["by_dtype"] = group_rates(ok, lambda x: x["dtype"])
    s["by_kind_dtype"] = group_rates(ok, lambda x: x["kind"] + "|" + x["dtype"].split(":")[0])
    s["by_b_src"] = group_rates(ok, lambda x: x["kind"] + "|" + str(x.get("b_src")))
    s["by_turn"] = group_rates(ok, lambda x: bucket_turn(x["turn"]))
    s["by_my_life"] = group_rates(ok, lambda x: "life%d" % min(x["my_life"], 4))
    s["by_opp_life"] = group_rates(ok, lambda x: "life%d" % min(x["opp_life"], 4))
    s["by_own"] = group_rates(ok, lambda x: "own" if x["own"] else "opp")
    s["by_rank"] = group_rates(ok, lambda x: str(x.get("b_src")))
    s["by_kind_mylife"] = group_rates(ok, lambda x: x["kind"] + "|life%d" % min(x["my_life"], 2))
    s["losses"] = [{k: x.get(k) for k in ("seed", "step", "seat", "turn", "kind", "dtype", "act", "b_src", "a_desc", "b_desc",
                                          "my_life", "opp_life", "confirmed", "confirmed_strict", "replicated", "recheck",
                                          "verified", "verify")}
                   | {"d13": x["ref"]["r13d"]["diff"], "ci13": x["ref"]["r13d"]["ci"],
                      "d20": (x["ref"].get("r20") or {}).get("diff"), "ci20": (x["ref"].get("r20") or {}).get("ci"),
                      "deep_best_is_a": (x.get("deep") or {}).get("best_pkey_is_a"),
                      "deep_best_is_b": (x.get("deep") or {}).get("best_pkey_is_b")}
                   for x in ok if x["cand"]]
    ws = read_jsonl(os.path.join(a.out, "window_stability.jsonl"))
    if ws:
        def part(xs):
            ds = [x["d13"] for x in xs]
            return {"n": len(xs), "mean_diff": round(sum(ds) / len(ds), 4) if ds else None,
                    "mean_ci": MJ._mean_ci(ds)["ci"] if len(ds) > 1 else None,
                    "d_ge_0.2": sum(d >= LOSS for d in ds), "d_le_-0.2": sum(d <= -LOSS for d in ds), "cand": sum(x["cand"] for x in xs)}
        st = [x for x in ws if x["a_share"] == 1]
        un = [x for x in ws if x["a_share"] < 1]
        s["window_stability"] = {"points": len(ws), "draws": 8, "unstable": rate(len(un), len(ws)),
                                 "q_varies": rate(sum(x["q_sets"] > 1 for x in ws), len(ws)),
                                 "stable": part(st), "unstable_part": part(un),
                                 "by_dtype": {k: rate(sum(1 for x in ws if x["dtype"] == k and x["a_share"] < 1),
                                                      sum(1 for x in ws if x["dtype"] == k))
                                              for k in sorted({x["dtype"] for x in ws})}}
    txt = json.dumps(s, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(json.dumps({k: s[k] for k in ("games", "decisions", "eligible", "ok", "sec_per_point", "sec_per_game", "all")},
                     ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play")
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--seed-base", type=int, default=3301000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--net", default=NET)
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("judge")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("recheck")
    p.add_argument("--k", type=int, default=48)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("verify")
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("winstab")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("summary")
    p.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)
    {"play": cmd_play, "judge": cmd_judge, "recheck": cmd_recheck, "verify": cmd_verify, "winstab": cmd_winstab, "summary": cmd_summary}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
