"""手の良し悪しの判定役——固定の審判が最後まで打つ（WP `move-judge`・`docs/search_efficiency_plan.md` §7.32）。

読み取りだけの計測（エンジン・ネット・既定の挙動は変えない）。§7.22（`value_to_move.py`・`logs/value_to_move/`）の続き:
深い読みを判定役にすると候補の手を出したネットが先の局面も自分の好みで高く評価する（r20 の手は r13d 判定で 27%・r20 判定で 71%）。
ここでは **比べる 2 つの手のそれぞれを局面に当て、その先を同じ審判のネットが両席とも最後まで打つ**（160 回・世界 1 本・公平・
詰みの証明 off＝アリーナの席と同じ組み方）。審判の好みは両方の手に同じだけかかる。

1. ``sample``: §7.22 の記録から標本を選ぶ——**出荷の設定で r13d と r20 の手が違った点**（`changed`）と**較正の点**
   （r13d が探索の乱数だけで替えた手・`calib`）。無作為に各 N 点。局面を作り直すための手の列（`moves.jsonl.gz`）も写す。
2. ``judge``: 点ごとに、手の列を探索なしで当て直して局面を作り直し（合法手の並びと記録の符号化で照合）、2 つの手
   （`a`＝r13d の出荷の手・`b`＝r20 の出荷の手〔較正では r13d が乱数だけで替えた手〕）を比べる。
   1 つの手につき K 局: 局ごとに手を打つ側の知らない札（相手の手札・両者の山札・裏向きのライフ）を公平に配り直し
   （`Game.fork_world(..., "fair", seed)`＝`decide` の世界の引き直しと同じ作り）、手を当て（箱の手は `apply_search_move` で
   箱全体）、その先を審判（`SeatSpec(net, sims=160)`）が両席とも打ち切りまで打つ。同じ配り方・同じ審判の乱数で a と b の両方を
   打つ＝対の差。審判は r13d と r20 の 2 本。参考に理論の式の値段（`live_theory.price_candidates`＝その 1 手だけの静的な値段と、
   `shadow_forbid.seq_prices`＝純付与をそれが準備する攻撃の値段で読む）も 2 つの手について書く。
3. ``summary``: 較正（勝率の差が 0 のまわりか）・r20 の手 vs r13d の手の勝率の差の分布・判定できた点（審判 2 本とも対の
   95% 区間が 0 を含まず向きが同じ）・§7.22 の深い読みの判定との比較・理論の式の符号の一致率。

    python tests/scripts/move_judge.py sample --band real --vm logs/value_to_move/real --out mj_out/real --n-changed 400 --n-calib 200
    python tests/scripts/move_judge.py judge --out mj_out/real --k 32 --workers 4
    python tests/scripts/move_judge.py summary --out mj_out/real mj_out/synth --write logs/move_judge/summary.json

再開できる（終えた点は飛ばす）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。
"""
import argparse
import gzip
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.loop import decks as D            # noqa: E402
from opcg_sim.loop import driver as DR          # noqa: E402
from opcg_sim.loop import engine as E           # noqa: E402
from opcg_sim.loop import record_gen as RG      # noqa: E402
import value_to_move as VM                      # noqa: E402

#: 審判の席（アリーナの席と同じ: 160 回・世界 1 本・公平・詰みの証明 off＝`decide` の既定）。
REFEREE_SIMS = 160
DEFAULT_REFS = "r13d=opcg_sim/data/learned/nrel_r13d.npz,r20=vm_nets/nrel_r20.npz"
#: 比べる手を取り出す読み（§7.22 と同じ基準の設定・同じ `search_seed`＝合法手の並びと束を再現するためだけ）。
BASE_NET = "opcg_sim/data/learned/nrel_r13d.npz"
#: 世界の引き直しの seed のずらし（§7.22 の読み・深い読み・乱数の対照のどれとも重ねない）。
WORLD_XOR = 0x4D4A_7B1E_0000_0000
MAX_STEPS = 400
_G = {}


def read_jsonl_gz(path):
    if not os.path.exists(path):
        return []
    opener = gzip.open if path.endswith(".gz") else open
    out = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def world_seed(seed, step, i):
    """点 (局 seed, step) の i 本目の配り方の seed（pure・u64）。"""
    return E.search_seed(int(seed) ^ WORLD_XOR, int(step) * 101 + int(i), "p1")


# ---------------------------------------------------------------- 1. sample

def replay_reach(mv, decks):
    """記録の手の列を探索なしで当て直し、外れた step（全部当たれば None）。"""
    E.engine()
    db = _G.setdefault("db", D.load_db())
    seed = int(mv["game"])
    p1, p2 = D.build_pair(db, mv["leaders"][0], mv["leaders"][1], seed, decks)
    game = DR.new_game(seed, p1, p2)
    for s, move in enumerate(mv["moves"]):
        name = json.loads(game.pending_json())["player_id"]
        try:
            if move.get("kind") == "battle":
                game.apply_battle_action(name, move["action_type"], move.get("card_uuid"))
            else:
                game.apply_game_action(name, move["action_type"], json.dumps(move.get("payload") or {}, ensure_ascii=False))
        except Exception:                           # noqa: BLE001
            return s
    return None


def cmd_sample(a):
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.vm, "meta.json"), encoding="utf-8"))
    base = VM.parse_nets(meta["nets"])[0][0]
    games = read_jsonl_gz(os.path.join(a.vm, "games.jsonl.gz"))
    bad = {int(g["game"]) for g in games if g["bad_step"] is not None}
    pts = [p for p in read_jsonl_gz(os.path.join(a.vm, "points.jsonl.gz")) if int(p["seed"]) not in bad]
    reseed = {(int(x["seed"]), int(x["step"])): x for x in read_jsonl_gz(os.path.join(a.vm, "reseed.jsonl.gz"))}
    deep = {}
    for jn in ("deep_r13d", "deep_r20"):
        deep[jn] = {(int(x["seed"]), int(x["step"])): x for x in read_jsonl_gz(os.path.join(a.vm, jn + ".jsonl.gz"))}

    def deep_of(key, ra, rb):
        out = {}
        for jn, jd in deep.items():
            d = jd.get(key)
            if not d or not d["ok"]:
                out[jn] = None
                continue
            ga, gb = d["groups"].get(str(ra)), d["groups"].get(str(rb))
            out[jn] = None if ga is None or gb is None else {"na": ga[0], "qa": ga[1], "nb": gb[0], "qb": gb[1]}
        return out

    changed, calib = [], []
    for p in pts:
        key = (int(p["seed"]), int(p["step"]))
        ra = p["res"][base]["prod"]["rep"]
        rb = p["res"][a.net]["prod"]["rep"]
        row = {"seed": key[0], "step": key[1], "turn": p["turn"], "who": p["who"], "src": p["src"],
               "leader": p["leader"], "n_groups": p["n_groups"], "legal": p["legal"]}
        if ra is not None and rb is not None and ra != rb:
            changed.append(dict(row, kind="changed", rep_a=ra, rep_b=rb, deep=deep_of(key, ra, rb)))
        r = reseed.get(key)
        if r and r["ok"] and base in r["res"]:
            rc = r["res"][base]["prod"]["rep"]
            if ra is not None and rc is not None and ra != rc:
                calib.append(dict(row, kind="calib", rep_a=ra, rep_b=rc, deep=deep_of(key, ra, rc)))
    # 記録の手の列が**今のエンジン**で局面まで当て直せる点だけから引く（§7.22 の後に誘発の順の修正〔rs-trigger-order-2・
    # rs-trigger-simul〕が入り、記録の 1 割弱の局は途中で記録から外れる＝その点は局面を作れない）。無作為の順に見て N 点そろうまで。
    mvs = {int(m["game"]): m for m in read_jsonl_gz(os.path.join(a.vm, "moves.jsonl.gz"))}
    decks = json.load(open(os.path.join(meta["dirs"][0], "meta_n_record.json"), encoding="utf-8"))["decks"]
    reach = {}

    def reproducible(x):
        g = int(x["seed"])
        if g not in reach:
            reach[g] = replay_reach(mvs[g], decks)
        return reach[g] is None or reach[g] > int(x["step"])
    rng = random.Random(a.seed)
    picks, dropped = {}, {}
    for kind, pool, n in (("changed", changed, a.n_changed), ("calib", calib, a.n_calib)):
        order = list(pool)
        rng.shuffle(order)
        got, drop = [], 0
        for x in order:
            if len(got) >= n:
                break
            if reproducible(x):
                got.append(x)
            else:
                drop += 1
        picks[kind] = sorted(got, key=lambda x: (x["seed"], x["step"]))
        dropped[kind] = drop
    pick_c, pick_k = picks["changed"], picks["calib"]
    sample = pick_c + pick_k
    need = {int(x["seed"]) for x in sample}
    with open(os.path.join(a.out, "sample.jsonl"), "w", encoding="utf-8") as f:
        for x in sample:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "moves.jsonl"), "w", encoding="utf-8") as f:
        for g in sorted(need):
            f.write(json.dumps(mvs[g]) + "\n")
    info = {"band": a.band, "vm": a.vm, "base": base, "net": a.net, "pool_changed": len(changed), "pool_calib": len(calib),
            "n_changed": len(pick_c), "n_calib": len(pick_k), "dropped_not_reproducible": dropped,
            "games_checked": len(reach), "games_diverged": sum(1 for v in reach.values() if v is not None),
            "sample_seed": a.seed, "decks": decks, "dirs": meta["dirs"]}
    json.dump(info, open(os.path.join(a.out, "meta.json"), "w"), ensure_ascii=False, indent=1)
    print("SAMPLE", json.dumps(info), flush=True)


# ---------------------------------------------------------------- 2. judge

def _init_judge(refs, k):
    import live_theory as LT
    import shadow_forbid as SF
    import theory_rs as TR
    from opcg_sim.learned.train import plan_labels as PL
    E.engine()
    _G["base"] = E.load_net(BASE_NET)
    _G["refs"] = [(name, E.SeatSpec(path, sims=REFEREE_SIMS)) for name, path in VM.parse_nets(refs)]
    _G["db"] = D.load_db()
    _G["k"] = int(k)
    _G["LT"], _G["SF"], _G["TR"] = LT, SF, TR
    _G["cards"] = PL.Cards()
    _G["idx2cid"] = TR.idx2cid()
    VM._G["db"] = _G["db"]


def play_out(game, spec, gseed, max_steps=MAX_STEPS):
    """`driver.run_game` の後半と同じ（既にある盤面を、両席 `spec` で打ち切りまで進める）。戻り値 (winner, steps)。"""
    carries = {"p1": DR.Carry(), "p2": DR.Carry()}
    steps = 0
    while game.winner() is None and steps < max_steps:
        pending = json.loads(game.pending_json())
        if not pending:
            break
        name = pending["player_id"]
        turn = int(game.turn_count or 0)
        carry = carries[name].get(turn, name)
        out = json.loads(game.decide(name, spec.decide_opts(gseed, turn, name, carry)))
        move = out.get("move")
        if move is None:
            carries[name].put(out)
            break
        carries[name].put(out)
        if move.get("kind") == "battle":
            game.apply_battle_action(name, move["action_type"], move.get("card_uuid"))
        else:
            game.apply_game_action(name, move["action_type"], json.dumps(move.get("payload") or {}, ensure_ascii=False))
        steps += 1
    return game.winner(), steps


def rollout(game, who, move, wseed, spec):
    """配り方 `wseed` の世界で `move` を当て、審判 `spec` が最後まで打つ → 1（who の勝ち）／0／None（決着せず・失敗）。"""
    g = game.fork_world(who, "fair", int(wseed))
    try:
        g.apply_search_move(who, json.dumps(move, ensure_ascii=False))
        w, _steps = play_out(g, spec, int(wseed))
    except Exception as exc:                        # noqa: BLE001  当てられない・局の失敗＝決着せず
        return None, "%s: %s" % (type(exc).__name__, str(exc)[:120])
    if w is None:
        return None, "no_winner"
    return (1 if w == who else 0), None


def paired(a, b):
    """対の差（b − a・両方決着した局だけ）の平均と 95% 区間・各手の勝率。"""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "wa": None, "wb": None, "diff": None, "ci": [None, None], "sd": None}
    d = [y - x for x, y in pairs]
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    h = 1.96 * math.sqrt(var / n) if n > 1 else float("inf")
    return {"n": n, "wa": round(sum(x for x, _ in pairs) / n, 4), "wb": round(sum(y for _, y in pairs) / n, 4),
            "diff": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)], "sd": round(math.sqrt(var), 4)}


def theory_prices(game, who, out):
    """2 つの手の理論の値段（束の並び＝`out["groups"]`）。値段が付かない手は None。"""
    LT, SF = _G["LT"], _G["SF"]
    sc, tok, ci = LT.raw_row(game, who)
    cands = LT.raw_candidates(game, who, out)
    if not cands:
        return None, None
    priced = LT.price_candidates(sc, tok, ci, _G["cards"], _G["idx2cid"], cands)
    seq, _n_re, _n_at = SF.seq_prices(cands, priced)
    reps = [int(g["rep"]) for g in out.get("groups") or []]
    return reps, {"static": [p["price"] for p in priced], "seq": [p["price"] for p in seq]}


def judge_point(job):
    p, mv, decks = job
    t0 = time.time()
    db = _G["db"]
    seed, step, who = int(p["seed"]), int(p["step"]), p["who"]
    row = {"kind": p["kind"], "seed": seed, "step": step, "turn": p["turn"], "who": who, "src": p["src"],
           "rep_a": p["rep_a"], "rep_b": p["rep_b"], "ok": False, "why": None}
    leaders = mv["leaders"]
    p1, p2 = D.build_pair(db, leaders[0], leaders[1], seed, decks)
    game = DR.new_game(seed, p1, p2)
    try:
        for s, move in enumerate(mv["moves"]):
            if s == step:
                break
            name = json.loads(game.pending_json())["player_id"]
            if move.get("kind") == "battle":
                game.apply_battle_action(name, move["action_type"], move.get("card_uuid"))
            else:
                game.apply_game_action(name, move["action_type"], json.dumps(move.get("payload") or {}, ensure_ascii=False))
    except Exception as exc:                        # noqa: BLE001
        row["why"] = "replay: %s" % type(exc).__name__
        return row
    name = json.loads(game.pending_json())["player_id"]
    turn = int(game.turn_count or 0)
    if name != who or turn != int(p["turn"]):
        row["why"] = "seat/turn"
        return row
    opts = {"net": _G["base"], "search_seed": E.search_seed(seed, turn, who), "sims": REFEREE_SIMS}
    out = json.loads(game.fork().decide(who, json.dumps(opts)))
    if VM.legal_key(out) != p["legal"]:
        row["why"] = "legal"
        return row
    if not VM.same_as_record(game, who, p):
        row["why"] = "record"
        return row
    legal = out["stats"]["legal"]
    ma, mb = legal[int(p["rep_a"])], legal[int(p["rep_b"])]
    row["sig_a"], row["sig_b"] = RG.move_sig(ma), RG.move_sig(mb)
    row["k_a"], row["k_b"] = RG._don_k(ma), RG._don_k(mb)
    try:
        reps, prices = theory_prices(game, who, out)
        if reps is not None:
            ia, ib = reps.index(int(p["rep_a"])), reps.index(int(p["rep_b"]))
            row["price"] = {k: {"a": v[ia], "b": v[ib]} for k, v in prices.items()}
        else:
            row["price"] = None
    except Exception as exc:                        # noqa: BLE001  値段の失敗は判定を止めない
        row["price"] = None
        row["price_err"] = "%s: %s" % (type(exc).__name__, str(exc)[:120])
    row["sec_prep"] = round(time.time() - t0, 2)
    K = _G["k"]
    refs = {}
    errs = {}
    for rname, spec in _G["refs"]:
        t1 = time.time()
        wa, wb = [], []
        for i in range(K):
            ws = world_seed(seed, step, i)
            ra, ea = rollout(game, who, ma, ws, spec)
            rb, eb = rollout(game, who, mb, ws, spec)
            wa.append(ra)
            wb.append(rb)
            for e in (ea, eb):
                if e:
                    errs[e] = errs.get(e, 0) + 1
        refs[rname] = dict(paired(wa, wb), a=wa, b=wb, void=sum(x is None for x in wa + wb),
                           sec=round(time.time() - t1, 1))
    row["ref"] = refs
    row["errs"] = errs or None
    row["ok"] = True
    row["sec"] = round(time.time() - t0, 1)
    return row


def cmd_judge(a):
    meta = json.load(open(os.path.join(a.out, "meta.json"), encoding="utf-8"))
    sample = VM.read_jsonl(os.path.join(a.out, "sample.jsonl"))
    mvs = {int(m["game"]): m for m in VM.read_jsonl(os.path.join(a.out, "moves.jsonl"))}
    decks_by_src = {}
    rpath = os.path.join(a.out, "rows.jsonl")
    done = {(x["kind"], int(x["seed"]), int(x["step"])) for x in VM.read_jsonl(rpath)}
    kinds = set(a.kinds.split(","))
    jobs = []
    for p in sample:
        if p["kind"] not in kinds or (p["kind"], int(p["seed"]), int(p["step"])) in done:
            continue
        src = p["src"]
        if src not in decks_by_src:
            decks_by_src[src] = json.load(open(os.path.join(src, "meta_n_record.json"), encoding="utf-8"))["decks"]
        jobs.append((p, mvs[int(p["seed"])], decks_by_src[src]))
    # 記録 dir（npz）の読み込みはワーカーごとに 1 dir を手元に置くので、src 順に並べて切り替えを減らす
    jobs.sort(key=lambda j: (j[0]["src"], j[0]["seed"], j[0]["step"]))
    if a.limit:
        jobs = jobs[:a.limit]
    print("JUDGE_START", json.dumps({"band": meta["band"], "k": a.k, "refs": a.refs, "todo": len(jobs), "done": len(done)}),
          flush=True)
    t0 = time.time()
    n = 0
    with Pool(a.workers, initializer=_init_judge, initargs=(a.refs, a.k)) as pool:
        for row in pool.imap_unordered(judge_point, jobs, chunksize=1):
            row["k"] = int(a.k)
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            brief = {r: (v["diff"], v["void"]) for r, v in (row.get("ref") or {}).items()}
            print("JUDGE %s seed=%d step=%d ok=%s why=%s %s sec=%s (%d/%d %.0fs)"
                  % (row["kind"], row["seed"], row["step"], row["ok"], row["why"], json.dumps(brief), row.get("sec"),
                     n, len(jobs), time.time() - t0), flush=True)
    print("JUDGE_DONE", flush=True)


# ---------------------------------------------------------------- 3. summary

MU = 0.0551   #: 手札 1 枚ぶんの価値（`theory_rs.MU`・値段の単位）


def _mean_ci(xs):
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "ci": [None, None]}
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    h = 1.96 * math.sqrt(var / n) if n > 1 else None
    return {"n": n, "mean": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)] if h is not None else [None, None]}


def _hist(xs, edges):
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        out["[%s,%s)" % (lo, hi)] = sum(1 for x in xs if lo <= x < hi)
    out["[%s,+]" % edges[-1]] = sum(1 for x in xs if x >= edges[-1])
    out["[-,%s)" % edges[0]] = sum(1 for x in xs if x < edges[0])
    return out


def sign(x, tol=0.0):
    return 0 if x is None or abs(x) <= tol else (1 if x > 0 else -1)


def decided(row, refs):
    """判定できた点＝審判 2 本とも対の 95% 区間が 0 を含まず、向きが同じ。戻り値 +1（b が良い）／−1／0。"""
    s = []
    for r in refs:
        v = row["ref"].get(r)
        if not v or v["n"] < 2 or v["ci"][0] is None:
            return 0
        if v["ci"][0] > 0:
            s.append(1)
        elif v["ci"][1] < 0:
            s.append(-1)
        else:
            return 0
    return s[0] if len(set(s)) == 1 else 0


def summarize_band(out_dir):
    meta = json.load(open(os.path.join(out_dir, "meta.json"), encoding="utf-8"))
    rows = VM.read_jsonl(os.path.join(out_dir, "rows.jsonl"))
    sample = VM.read_jsonl(os.path.join(out_dir, "sample.jsonl"))
    deep_by = {(x["kind"], int(x["seed"]), int(x["step"])): x.get("deep") for x in sample}
    refs = sorted({r for x in rows if x.get("ref") for r in x["ref"]})
    s = {"band": meta["band"], "sample": {"changed": meta["n_changed"], "calib": meta["n_calib"],
                                          "pool_changed": meta["pool_changed"], "pool_calib": meta["pool_calib"]},
         "rows": len(rows), "k": (rows[0].get("k") if rows else None), "refs": refs}
    s["not_ok"] = {}
    for x in rows:
        if not x["ok"]:
            s["not_ok"][x["why"]] = s["not_ok"].get(x["why"], 0) + 1
    ok = [x for x in rows if x["ok"]]
    s["ok_rows"] = len(ok)
    s["sec_per_point"] = round(sum(x["sec"] for x in ok) / len(ok), 1) if ok else None
    for kind in ("calib", "changed"):
        xs = [x for x in ok if x["kind"] == kind]
        ks = {"points": len(xs)}
        for r in refs:
            vs = [x["ref"][r] for x in xs if x["ref"].get(r) and x["ref"][r]["diff"] is not None]
            diffs = [v["diff"] for v in vs]
            ks[r] = {"diff": _mean_ci(diffs), "wa": _mean_ci([v["wa"] for v in vs]), "wb": _mean_ci([v["wb"] for v in vs]),
                     "b_better": VM.rate(sum(d > 0 for d in diffs), sum(d != 0 for d in diffs)),
                     "ties": sum(d == 0 for d in diffs),
                     "ci_excludes_0": {"b": sum(v["ci"][0] > 0 for v in vs), "a": sum(v["ci"][1] < 0 for v in vs)},
                     "hist": _hist(diffs, [-0.5, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.5]),
                     "void_games": sum(v["void"] for v in vs),
                     "games": sum(2 * len(v["a"]) for v in vs)}
        if len(refs) >= 2:
            r0, r1 = refs[0], refs[1]
            both = [x for x in xs if all(x["ref"].get(r) and x["ref"][r]["diff"] is not None for r in (r0, r1))]
            nz = [x for x in both if x["ref"][r0]["diff"] != 0 and x["ref"][r1]["diff"] != 0]
            ks["sign_agree_refs"] = VM.rate(sum(sign(x["ref"][r0]["diff"]) == sign(x["ref"][r1]["diff"]) for x in nz), len(nz))
            ds = [decided(x, (r0, r1)) for x in both]
            ks["decided"] = {"n": sum(d != 0 for d in ds), "b_better": sum(d > 0 for d in ds), "a_better": sum(d < 0 for d in ds),
                             "rate": VM.rate(sum(d != 0 for d in ds), len(both)),
                             "split": sum(1 for x in both if all(
                                 (x["ref"][r]["ci"][0] > 0 or x["ref"][r]["ci"][1] < 0) for r in (r0, r1))
                                 and decided(x, (r0, r1)) == 0)}
            per = {}
            for r in (r0, r1):
                per[r] = sum(1 for x in both if x["ref"][r]["ci"][0] > 0 or x["ref"][r]["ci"][1] < 0)
            ks["ci_excludes_0_per_ref"] = per
            xs2 = [(x["ref"][r0]["diff"] + x["ref"][r1]["diff"]) / 2 for x in both]
            ks["diff_mean_of_refs"] = _mean_ci(xs2)
            # §7.22 の深い読みの判定（訪問の多い方）との比較
            dp = {}
            for jn in ("deep_r13d", "deep_r20"):
                agree_n = agree_d = n_all = n_dec = 0
                for x in both:
                    d = (deep_by.get((x["kind"], x["seed"], x["step"])) or {}).get(jn)
                    if not d or d["na"] == d["nb"]:
                        continue
                    dv = 1 if d["nb"] > d["na"] else -1
                    m = sign((x["ref"][r0]["diff"] + x["ref"][r1]["diff"]) / 2)
                    if m != 0:
                        n_all += 1
                        agree_n += (m == dv)
                    dd = decided(x, (r0, r1))
                    if dd != 0:
                        n_dec += 1
                        agree_d += (dd == dv)
                dp[jn] = {"deep_b_better": VM.rate(sum(1 for x in both if ((deep_by.get((x["kind"], x["seed"], x["step"])) or {}).get(jn) or {"na": 0, "nb": 0})["nb"] > ((deep_by.get((x["kind"], x["seed"], x["step"])) or {}).get(jn) or {"na": 0, "nb": 0})["na"]),
                                                   sum(1 for x in both if (deep_by.get((x["kind"], x["seed"], x["step"])) or {}).get(jn))),
                          "agree_with_refs_mean": VM.rate(agree_n, n_all), "agree_on_decided": VM.rate(agree_d, n_dec)}
            ks["vs_deep"] = dp
            # 理論の式（参考）: 値段の差 b − a の符号と、審判の勝率の差の符号の一致
            th = {}
            for pk in ("static", "seq"):
                pr = [x for x in both if x.get("price") and x["price"][pk]["a"] is not None and x["price"][pk]["b"] is not None]
                dpr = [(x, (x["price"][pk]["b"] - x["price"][pk]["a"]) / MU) for x in pr]
                nz2 = [(x, d) for x, d in dpr if abs(d) > 1e-9]
                agree = {}
                for r in (r0, r1):
                    z = [(x, d) for x, d in nz2 if x["ref"][r]["diff"] != 0]
                    agree[r] = VM.rate(sum(sign(d) == sign(x["ref"][r]["diff"]) for x, d in z), len(z))
                z = [(x, d) for x, d in nz2 if sign((x["ref"][r0]["diff"] + x["ref"][r1]["diff"]) / 2) != 0]
                agree["refs_mean"] = VM.rate(sum(sign(d) == sign((x["ref"][r0]["diff"] + x["ref"][r1]["diff"]) / 2) for x, d in z), len(z))
                z = [(x, d) for x, d in nz2 if decided(x, (r0, r1)) != 0]
                agree["decided"] = VM.rate(sum(sign(d) == decided(x, (r0, r1)) for x, d in z), len(z))
                th[pk] = {"priced": len(pr), "unpriced": len(both) - len(pr), "ties": len(dpr) - len(nz2),
                          "b_higher": VM.rate(sum(d > 0 for _, d in nz2), len(nz2)),
                          "dprice_mu": _mean_ci([d for _, d in dpr]),
                          "abs_dprice_mu_median": (round(sorted(abs(d) for _, d in dpr)[len(dpr) // 2], 3) if dpr else None),
                          "sign_agree": agree}
            ks["theory"] = th
        s[kind] = ks
    return s


def cmd_summary(a):
    out = {"bands": [summarize_band(d) for d in a.out]}
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    if a.write:
        os.makedirs(os.path.dirname(a.write) or ".", exist_ok=True)
        with open(a.write, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    print(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--band", required=True)
    p.add_argument("--vm", required=True, help="§7.22 の記録の dir（logs/value_to_move/<band>）")
    p.add_argument("--out", required=True)
    p.add_argument("--net", default="r20", help="比べるネット（§7.22 の記録の名前）")
    p.add_argument("--n-changed", type=int, default=400)
    p.add_argument("--n-calib", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261023)
    p = sub.add_parser("judge")
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=32, help="1 つの手につき打つ局の数（配り方の数）")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--refs", default=DEFAULT_REFS, help="審判のネット name=path,...")
    p.add_argument("--kinds", default="calib,changed")
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("summary")
    p.add_argument("--out", nargs="+", required=True)
    p.add_argument("--write", default=None)
    a = ap.parse_args(argv)
    {"sample": cmd_sample, "judge": cmd_judge, "summary": cmd_summary}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
