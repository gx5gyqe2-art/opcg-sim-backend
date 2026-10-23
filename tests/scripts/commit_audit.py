"""自分の手番の続き（箱の commit）を盤面を見ずに打って損をしていないかの監査（`docs/search_efficiency_plan.md` §7.40・
WP `commit-audit`・台帳 2+369）。読み取りだけ（エンジン・既定の挙動は変えない）。1 回きりの計測＝テストは足さない。

§7.26（`2026-10-22_defense_audit.md`・守りの続きの照合の欠陥）と同じ型が、自分の手番の続き（付与 → 攻撃の箱・準備箱・
登場時の対話・窓の根畳みの続き）にも無いかを数える。

``run``: 両席 r13d・出荷の設定（1,280 回・世界 2 本・枝予算なし・詰みの証明 root）・random × real で打つ。
  **全ての判断点**で実局の決め方（`main`＝木／`window`＝窓の根畳み／`commit`＝続きを打った／`lethal`＝証明）を数え、
  `commit` の点では:
  - 続きの出どころ（その続きを積んだ決定: 付与→攻撃の箱 `box`・登場時の対話 `play`・準備箱 `setup`・窓の根畳み `window`）と
    続きの段の種類（付与・攻撃・対話〔要求の種類〕・登場・戦闘）、
  - **持ち越し無しで読み直した手**（同じ `search_seed`・持ち越しだけを外した `Game.decide`）、
  - 較正: 探索の乱数だけを替えた読み直し（`search_seed ^ RESEED_XOR`・持ち越し無し）＝同じネットの乱数の揺らぎ、
  を書く。続きの手と読み直しの手が違う点（と、較正で 2 つの読み直しが違う点）では、深い読み（出荷の 4 倍の 5,120 回・
  世界 4 本・枝予算なし・詰みの証明 off・判定役の seed はずらす）で 2 つの手を比べる:
  - ``root``: その点の根の木（メインの判断点だけ木になる）で、各手を先頭の原始手に持つ束のうち訪問が最多の束の (n, q)。
  - ``child``: 各手を当てた後の盤面で深い読みを 1 回ずつ回し、手を打った側から見た値（木＝訪問重みの q の平均・
    窓＝最良の枝の値・決着＝±1）。対話・戦闘の点（木にならない）も比べられる。
``summary``: 数える（`summary.json`）。

    python tests/scripts/commit_audit.py run --games 300 --workers 4
    python tests/scripts/commit_audit.py summary

固定の審判で最後まで打つ判定（WP `move-judge`）の器は本線に無い（2026-10-23）＝深い読みの値だけで比べる。
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("OPCG_LOG_SILENT", "1")

from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402
from opcg_sim.loop import record_gen as RG  # noqa: E402

OUT_DIR = os.path.join("n1_results", "commit_audit")
#: 深い読み（`value_to_move.py` の DEEP と同じ・詰みの証明 off）。
DEEP = {"sims": 5120, "worlds": 4, "budget": 0}
#: 判定役の seed のずらし（比べる読みと同じ世界にしない・`value_to_move.JUDGE_XOR` と同じ値）。
JUDGE_XOR = 0x1D6E00007E570000
#: 較正の読み直しの seed のずらし（`value_to_move.RESEED_XOR` と同じ値）。
RESEED_XOR = 0x5EED_0F_F5E7_0000
_G: Dict[str, Any] = {}


def ship_spec(net: Optional[str]) -> "E.SeatSpec":
    """出荷の設定（対戦 API と同じ: `PLAY_SIMS`・`PLAY_BUDGET`・`PLAY_WORLDS`・`PLAY_LETHAL_PROOF`）。"""
    return E.SeatSpec(net, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=E.PLAY_WORLDS,
                      lethal_proof=E.PLAY_LETHAL_PROOF)


def key(mv: Optional[Dict[str, Any]]) -> Optional[str]:
    """手の同一性（原始手の署名）。"""
    return None if mv is None else json.dumps(RG.move_sig(mv), ensure_ascii=False)


def desc(game, mv: Optional[Dict[str, Any]]) -> Optional[str]:
    if mv is None:
        return None
    try:
        return json.dumps(json.loads(game.describe_move_json(json.dumps(mv, ensure_ascii=False))), ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return json.dumps(mv, ensure_ascii=False)


def first_primitive(mv: Dict[str, Any]) -> Dict[str, Any]:
    """箱 → 実対局へ出す先頭の原始手（Rust `don_box_first_primitive` の写し）。"""
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


def _n_legal(game, name: str) -> Optional[int]:
    try:
        return len(json.loads(game.legal_json(name)))
    except Exception:  # noqa: BLE001
        return None


def step_type(mv: Dict[str, Any], pend: Dict[str, Any]) -> str:
    at = mv.get("action_type")
    if at == "ATTACH_DON":
        return "attach"
    if at == "ATTACK":
        return "attack"
    if at in ("PLAY", "ACTIVATE_MAIN"):
        return "play"
    if at in ("SELECT_COUNTER", "SELECT_BLOCKER", "PASS"):
        return "battle"
    if at == "TURN_END":
        return "turn_end"
    return f"dialog:{pend.get('action')}"


def apply(game, actor: str, mv: Dict[str, Any]) -> None:
    if mv.get("kind") == "battle":
        game.apply_battle_action(actor, mv["action_type"], mv.get("card_uuid"))
    else:
        game.apply_game_action(actor, mv["action_type"], json.dumps(mv.get("payload") or {}, ensure_ascii=False))


#: 判定役の seed を複数にするときの 2 本目以降のずらし（`--judge-seeds`）。
JUDGE_STEP = 0x9E3779B97F4A7C15


def deep_decide(game, name: str, ss: int, j: int = 0) -> Dict[str, Any]:
    opts = {"net": _G["net"], "search_seed": (ss ^ JUDGE_XOR ^ (j * JUDGE_STEP)) & 0xFFFFFFFFFFFFFFFF, **DEEP}
    return json.loads(game.decide(name, json.dumps(opts)))


def out_value(o: Dict[str, Any]) -> Optional[float]:
    """decide の出力 → 決める側から見た局面の値（木＝訪問重みの q の平均・窓＝最良の枝の値）。"""
    if o.get("kind") == "main":
        gs = o.get("groups") or []
        tot = sum(float(g["n"]) for g in gs)
        if tot > 0:
            return sum(float(g["n"]) * float(g["q"]) for g in gs) / tot
    st = o.get("stats") or {}
    qs, ns = st.get("Q") or [], st.get("N") or []
    vals = [float(q) for q, n in zip(qs, ns) if float(n) > 0]
    if vals:
        return max(vals)
    return None


def child_value(game, mover: str, mv: Dict[str, Any], ss: int, j: int = 0, forward: bool = False) -> Dict[str, Any]:
    """手を当てた後の盤面の値（手を打った側から見た・深い読み）。`forward`＝次の木の判断点（メイン）まで判定役の決め方で進めてから測る（窓の値と木の値を混ぜない）。"""
    g = game.fork()
    try:
        apply(g, mover, mv)
    except Exception as exc:  # noqa: BLE001
        return {"v": None, "err": f"{type(exc).__name__}: {exc}"[:120]}
    for _ in range(40 if forward else 1):
        w = g.winner()
        if w is not None:
            return {"v": 1.0 if w == mover else -1.0, "end": True}
        pend = json.loads(g.pending_json())
        if not pend:
            return {"v": None, "err": "no pending"}
        who = pend["player_id"]
        o = deep_decide(g, who, ss, j)
        v = out_value(o)
        if not forward or o.get("kind") == "main" or not o.get("move"):
            break
        # `forward`: 木の判断点（メイン）まで、判定役の決め方（窓・候補 1 つ）で進める＝両方の手の値を同じ読み（木の根の値）で測る
        apply(g, who, o["move"])
    if forward and o.get("kind") != "main":
        v = None
    if v is None:
        return {"v": None, "kind": o.get("kind"), "who": who}
    return {"v": round(v if who == mover else -v, 4), "kind": o.get("kind"), "who": "me" if who == mover else "opp"}


def root_values(game, name: str, ss: int, moves: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """その点の深い読みの根（木のときだけ）で、各手を先頭の原始手に持つ束のうち訪問最多の束の (n, q)。"""
    o = deep_decide(game.fork(), name, ss)
    if o.get("kind") != "main":
        return {"kind": o.get("kind")}
    legal = (o.get("stats") or {}).get("legal") or []
    gs = o.get("groups") or []
    tot = sum(float(g["n"]) for g in gs) or 1.0
    res = {"kind": "main", "best": key(first_primitive(o["move"])) if o.get("move") else None, "per": []}
    for mv in moves:
        k = key(mv)
        cand = [g for g in gs if key(first_primitive(legal[g["rep"]])) == k]
        if not cand:
            res["per"].append(None)
            continue
        g = max(cand, key=lambda g: float(g["n"]))
        res["per"].append({"n": round(float(g["n"]) / tot, 4), "q": round(float(g["q"]), 4),
                           "q_max": round(max(float(c["q"]) for c in cand), 4), "groups": len(cand)})
    return res


def judge_pair(game, name: str, ss: int, a: Dict[str, Any], b: Dict[str, Any], tag: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    r = {"tag": tag, "root": root_values(game, name, ss, [a, b]),
         "child": [child_value(game, name, a, ss), child_value(game, name, b, ss)]}
    r["sec"] = round(time.perf_counter() - t0, 2)
    return r


def judge_multi(game, name: str, ss: int, a: Dict[str, Any], b: Dict[str, Any], k: int) -> Dict[str, Any]:
    """各手を当てた後の値を判定役の seed k 本で読み、平均する（窓の値は世界 1 本の読みなので揺れる）。"""
    t0 = time.perf_counter()
    va = [child_value(game, name, a, ss, j, True).get("v") for j in range(k)]
    vb = [child_value(game, name, b, ss, j, True).get("v") for j in range(k)]
    ok = [(x, y) for x, y in zip(va, vb) if x is not None and y is not None]
    return {"va": va, "vb": vb, "dv": round(sum(y - x for x, y in ok) / len(ok), 4) if ok else None,
            "sec": round(time.perf_counter() - t0, 2)}


def play_one(args) -> List[Dict[str, Any]]:
    seed, a = args
    E.engine()
    _G["net"] = E.load_net(a.net)
    db = D.load_db()
    spec = ship_spec(a.net)
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "real")
    kinds: Dict[str, int] = {}
    src = {"p1": None, "p2": None}           # 席ごとの今の続きの出どころ
    pos = {"p1": 0, "p2": 0}                 # 続きの何段目か
    origin = {"p1": None, "p2": None}        # 続きを積んだ決定の step
    hist: List[tuple] = []                   # (step, 席, 手の種類)
    points: List[Dict[str, Any]] = []

    def obs(game, name, turn, step, out, move):
        kind = out.get("kind")
        own = game.turn_player() == name
        kk = f"{'own' if own else 'opp'}:{kind}"
        kinds[kk] = kinds.get(kk, 0) + 1
        hist.append((step, name, move.get("action_type")))
        if kind == "commit":
            pos[name] += 1
            pend = json.loads(game.pending_json())
            ss = E.search_seed(seed, turn, name)
            base = json.loads(spec.decide_opts(seed, turn, name, None))
            t0 = time.perf_counter()
            fresh = json.loads(game.fork().decide(name, json.dumps(base)))
            base2 = dict(base, search_seed=ss ^ RESEED_XOR)
            fresh2 = json.loads(game.fork().decide(name, json.dumps(base2)))
            dt = time.perf_counter() - t0
            fm, gm = fresh.get("move"), fresh2.get("move")
            rec = {"seed": seed, "step": step, "seat": name, "turn": turn, "own": own,
                   "src": src[name], "pos": pos[name], "left": len(out.get("commit") or []),
                   "act": pend.get("action"), "type": step_type(move, pend),
                   "live": desc(game, move), "fresh": desc(game, fm), "fresh_kind": fresh.get("kind"),
                   "reseed": desc(game, gm), "reseed_kind": fresh2.get("kind"),
                   "n_legal": _n_legal(game, name),
                   "diff": key(move) != key(fm), "diff_desc": desc(game, move) != desc(game, fm),
                   "cal_diff": key(fm) != key(gm), "cal_diff_desc": desc(game, fm) != desc(game, gm),
                   "sec_fresh": round(dt, 2), "origin": origin[name]}
            if origin[name] is not None:
                rec["between_opp"] = [h[2] for h in hist if origin[name] < h[0] < step and h[1] != name]
            jt = a.judge_types.split(",") if a.judge_types else None
            sel = jt is None or any(rec["type"].startswith(t) for t in jt)
            if a.judge_seeds and sel and rec["diff"] and fm is not None:
                rec["judge_multi"] = judge_multi(game, name, ss, move, fm, a.judge_seeds)
            if a.judge_seeds and sel and rec["cal_diff"] and fm is not None and gm is not None:
                rec["judge_multi_cal"] = judge_multi(game, name, ss, fm, gm, a.judge_seeds)
            if a.deep and rec["diff"] and fm is not None:
                rec["judge"] = judge_pair(game, name, ss, move, fm, "live_vs_fresh")
            if a.deep and rec["cal_diff"] and fm is not None and gm is not None:
                rec["judge_cal"] = judge_pair(game, name, ss, fm, gm, "fresh_vs_reseed")
            points.append(rec)
        else:
            if out.get("commit"):
                at = (out.get("sig") or [None])[0]
                src[name] = ("window" if kind == "window" else
                             {"DON_BOX": "box", "PLAY": "play", "ACTIVATE_MAIN": "play",
                              "SETUP_BOX": "setup"}.get(at, f"other:{at}"))
                src[name] = f"{src[name]}@{'own' if own else 'opp'}"
                pos[name], origin[name] = 0, step
            else:
                src[name], pos[name], origin[name] = None, 0, None

    t0 = time.perf_counter()
    try:
        r = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, observer=obs)
        winner, turns, void = r["winner"], r["turns"], r["winner"] is None
    except DR.GameAborted as exc:
        winner, turns, void = None, None, str(exc)
    g = {"game": {"seed": seed, "winner": winner, "turns": turns, "void": void, "kinds": kinds,
                  "points": len(points), "sec": round(time.perf_counter() - t0, 1)}}
    return [g] + points


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"points_{a.seed_base}_{a.games}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            if "game" in d:
                done.add(d["game"]["seed"])
    seeds = [a.seed_base + i for i in range(a.games) if a.seed_base + i not in done]
    print(f"todo {len(seeds)} (done {len(done)}) -> {path}", flush=True)
    import multiprocessing as mp
    with open(path, "a", encoding="utf-8") as f, mp.get_context("fork").Pool(a.workers) as pool:
        for i, recs in enumerate(pool.imap_unordered(play_one, [(s, a) for s in seeds], chunksize=1)):
            for r in recs[1:] + recs[:1]:          # 局の行を最後に＝局の行があれば点は全部書けている
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            g = recs[0]["game"]
            print(f"[{i + 1}/{len(seeds)}] seed={g['seed']} pts={g['points']} sec={g['sec']} winner={g['winner']} "
                  f"turns={g['turns']}", flush=True)


# --- 集計 -------------------------------------------------------------------------------------

def _rate(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None}


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def _load(paths):
    games, pts = [], []
    import gzip
    for p in paths:
        for line in (gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")):
            d = json.loads(line)
            (games if "game" in d else pts).append(d["game"] if "game" in d else d)
    seeds = {g["seed"] for g in games}
    return games, [p for p in pts if p["seed"] in seeds]


def _judge_side(j: Optional[Dict[str, Any]]):
    """judge → (root の dq〔b − a〕, child の dv〔b − a〕)。b＝読み直しの手（較正では乱数を替えた手）。"""
    if not j:
        return None, None
    dq = None
    r = j.get("root") or {}
    if r.get("kind") == "main" and r["per"][0] and r["per"][1]:
        dq = r["per"][1]["q"] - r["per"][0]["q"]
    c = j.get("child") or [{}, {}]
    dv = None
    if c[0].get("v") is not None and c[1].get("v") is not None:
        dv = c[1]["v"] - c[0]["v"]
    return dq, dv


def _better(vals, eps=1e-9):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    b = sum(1 for v in vals if v > eps)
    w = sum(1 for v in vals if v < -eps)
    srt = sorted(vals)
    return {"n": n, "fresh_better": b, "live_better": w, "tie": n - b - w,
            "fresh_better_rate_excl_tie": round(b / (b + w), 4) if b + w else None,
            "ci": wilson(b, b + w), "mean": round(sum(vals) / n, 4) if n else None,
            "median": round(srt[n // 2], 4) if n else None}


def _card_key(d: Optional[str]) -> Optional[str]:
    """記述から同名複製の区別（`selected_slots`）を外した鍵＝札の名前の水準の同一性。"""
    if d is None:
        return None
    try:
        x = json.loads(d)
    except Exception:  # noqa: BLE001
        return d
    x.pop("selected_slots", None)
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def diff_shape(p: Dict[str, Any]) -> str:
    """続きの手と読み直しの手の違い方（選ぶ対話の型を分ける）。"""
    if not p["diff"]:
        return "same"
    if _card_key(p["live"]) == _card_key(p["fresh"]):
        return "same_card_other_copy"
    try:
        lv, fr = json.loads(p["live"]), json.loads(p["fresh"])
    except Exception:  # noqa: BLE001
        return "other"
    if lv.get("action_type") != fr.get("action_type"):
        return f"{lv.get('action_type')}->{fr.get('action_type')}"
    if "selected" in lv or "selected" in fr:
        if not lv.get("selected"):
            return "live_none_fresh_some"
        if not fr.get("selected"):
            return "live_some_fresh_none"
        return "other_card"
    if lv.get("accepted") is False or fr.get("accepted") is False:
        return "accept_vs_decline"
    if lv.get("card") != fr.get("card"):
        return "other_card"
    return "other_target"


def summarize(games, pts) -> Dict[str, Any]:
    s: Dict[str, Any] = {}
    s["games"] = len(games)
    s["void"] = sum(1 for g in games if g["void"])
    kinds: Dict[str, int] = {}
    for g in games:
        for k, v in g["kinds"].items():
            kinds[k] = kinds.get(k, 0) + v
    s["decisions"] = kinds
    own_all = sum(v for k, v in kinds.items() if k.startswith("own:"))
    s["commit_share"] = {"own": _rate(kinds.get("own:commit", 0), own_all),
                         "opp": _rate(kinds.get("opp:commit", 0), sum(v for k, v in kinds.items() if k.startswith("opp:"))),
                         "all": _rate(kinds.get("own:commit", 0) + kinds.get("opp:commit", 0), sum(kinds.values()))}

    def table(field):
        out = {}
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for p in pts:
            groups.setdefault(field(p), []).append(p)
        for k in sorted(groups, key=lambda k: -len(groups[k])):
            g = groups[k]
            row = {"points": len(g),
                   "diff": _rate(sum(p["diff"] for p in g), len(g)),
                   "diff_desc": _rate(sum(p["diff_desc"] for p in g), len(g)),
                   "diff_card": _rate(sum(diff_shape(p) not in ("same", "same_card_other_copy") for p in g), len(g)),
                   "cal_diff": _rate(sum(p["cal_diff"] for p in g), len(g)),
                   "cal_diff_desc": _rate(sum(p["cal_diff_desc"] for p in g), len(g))}
            jr = [_judge_side(p.get("judge")) for p in g if p.get("judge")]
            jc = [_judge_side(p.get("judge_cal")) for p in g if p.get("judge_cal")]
            row["deep_root"] = _better([x[0] for x in jr])
            row["deep_child"] = _better([x[1] for x in jr])
            row["cal_deep_root"] = _better([x[0] for x in jc])
            row["cal_deep_child"] = _better([x[1] for x in jc])
            if any("judge_multi" in p or "judge_multi_cal" in p for p in g):
                # 補助の判定（`--judge-seeds`・次の木の判断点まで進めてから seed k 本の平均）
                row["multi"] = _better([(p.get("judge_multi") or {}).get("dv") for p in g if p.get("judge_multi")])
                row["cal_multi"] = _better([(p.get("judge_multi_cal") or {}).get("dv") for p in g if p.get("judge_multi_cal")])
            if any("between_opp" in p for p in g):
                wb = [p for p in g if p.get("between_opp")]
                nb = [p for p in g if "between_opp" in p and not p["between_opp"]]
                row["opp_moved_between"] = {"points": len(wb), "diff": _rate(sum(p["diff"] for p in wb), len(wb)),
                                            "no_opp_points": len(nb), "no_opp_diff": _rate(sum(p["diff"] for p in nb), len(nb))}
            out[k] = row
        return out

    s["by_all"] = table(lambda p: "all")
    s["by_side"] = table(lambda p: "own" if p["own"] else "opp")
    s["by_type"] = table(lambda p: f"{'own' if p['own'] else 'opp'}:{p['type']}")
    s["by_src"] = table(lambda p: str(p["src"]))
    s["by_src_type"] = table(lambda p: f"{p['src']}|{p['type']}")
    s["by_type_shape"] = table(lambda p: f"{'own' if p['own'] else 'opp'}:{p['type']}|{diff_shape(p)}")
    s["fresh_kind"] = {}
    for p in pts:
        s["fresh_kind"][str(p["fresh_kind"])] = s["fresh_kind"].get(str(p["fresh_kind"]), 0) + 1
    s["fresh_kind_when_diff"] = {}
    for p in pts:
        if p["diff"]:
            s["fresh_kind_when_diff"][str(p["fresh_kind"])] = s["fresh_kind_when_diff"].get(str(p["fresh_kind"]), 0) + 1
    return s


def cmd_summary(a):
    import glob
    paths = sorted(glob.glob(os.path.join(a.out, "points_*.jsonl"))) or sorted(glob.glob(os.path.join(a.out, "points_*.jsonl.gz")))
    games, pts = _load(paths)
    s = summarize(games, pts)
    path = os.path.join(a.out, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: s[k] for k in ("games", "void", "commit_share")}, ensure_ascii=False))
    for k in ("by_side", "by_type", "by_src"):
        print(f"== {k}")
        for t, r in s[k].items():
            print(f"  {t:42s} pts={r['points']:5d} diff={r['diff']['rate']} desc={r['diff_desc']['rate']} "
                  f"cal={r['cal_diff']['rate']} | root fb={r['deep_root']['fresh_better']}/{r['deep_root']['live_better']} "
                  f"child fb={r['deep_child']['fresh_better']}/{r['deep_child']['live_better']} "
                  f"| cal root {r['cal_deep_root']['fresh_better']}/{r['cal_deep_root']['live_better']} "
                  f"child {r['cal_deep_child']['fresh_better']}/{r['cal_deep_child']['live_better']}")
    print(f"-> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--seed-base", type=int, default=3291000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--net", default=None, help="既定＝出荷のネット（r13d）")
    p.add_argument("--no-deep", dest="deep", action="store_false", help="深い読みの比べを省く（試走用）")
    p.add_argument("--judge-seeds", type=int, default=0,
                   help="補助の判定: 各手を当てた後の値を判定役の seed k 本の平均で読む（0＝しない）")
    p.add_argument("--judge-types", default="", help="補助の判定を当てる段の種類（前方一致・カンマ区切り・空＝全部）")
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("summary")
    p.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)
    {"run": cmd_run, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
