"""WP weak-points §3: 自分の効果の対話で「手札から選ぶ」点（捨てる・戻す・公開する等）を数え、標本を判定する（1 回きり）。

局: reaudit_wr の大きな損の 100 局（両席 r13d・出荷の設定・seed 3341000〜3341099・`reaudit_wr/blunder/games.jsonl` の手の列）。
  scan : 全局を当て直し、own_dialog の SEARCH_AND_SELECT（候補 2 つ以上）のうち、候補の札が全部自分の手札にある点を集める
         （メッセージ・自分の手番か・候補の札・記録の窓の値 top）。→ hand_points.jsonl
  judge: 点を無作為に N 点選び、今の本線で出荷の設定（r13d・blunder_audit.ship_spec）の窓の値を読み直し、
         a＝今の CPU の手・b＝窓の値が次に高い別の札（同じ札番号は飛ばす）を、審判 r13d（160 回）で K 局ずつ比べる。
         今のターンの残りは出荷の設定が打つ（blunder_audit.rollout_ship＝§7.41 の verify と同じ打ち継ぎ）。
         出荷のネット r22（世界 2 本・証明 root）の窓の手も書く。→ hand_judge.jsonl
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, "tests/scripts")
from multiprocessing import get_context

OUT = "n1_results/weak_points"
_G = {}


def _init():
    import blunder_audit as BA
    BA._init_judge(1)
    from opcg_sim.loop import engine as E
    _G["BA"], _G["E"] = BA, E
    _G["r22"] = E.SeatSpec(None, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=E.PLAY_WORLDS, lethal_proof=E.PLAY_LETHAL_PROOF)


def scan_game(g):
    BA = _G["BA"]
    rows = []
    want = {p["step"]: p for p in g["points"] if p["dtype"] == "own_dialog:SEARCH_AND_SELECT" and p["n_cand"] >= 2}
    if not want:
        return rows
    game = BA.replay(g, 0)
    for s, mv in enumerate(g["moves"]):
        if s in want:
            p = want[s]
            pend = json.loads(game.pending_json())
            h = json.loads(game.hidden_json())
            me = p["seat"]
            hand = {c["uuid"]: c["card_id"] for c in h["players"][me]["hand"]}
            sel = pend.get("selectable_uuids") or []
            if sel and all(u in hand for u in sel):
                rows.append({"seed": g["seed"], "step": s, "seat": me, "turn": p["turn"], "msg": pend.get("message"),
                             "my_turn": h["manager"]["turn_player"] == me,
                             "cands": sorted({hand[u] for u in sel}), "n_sel": len(sel),
                             "constraints": pend.get("constraints"),
                             "picked": [hand.get(u) for u in (mv.get("payload") or {}).get("selected_uuids") or []],
                             "top": [{"cards": [hand.get(u) for u in (t["mv"].get("payload") or {}).get("selected_uuids") or []],
                                      "q": t["q"]} for t in p["top"]],
                             "my_life": len(h["players"][me]["life"]),
                             "opp_life": len(h["players"]["p2" if me == "p1" else "p1"]["life"])})
        name = json.loads(game.pending_json())["player_id"]
        BA.apply(game, name, mv)
        if s >= max(want):
            break
    return rows


def judge_point(job):
    p, g, K = job
    BA, E = _G["BA"], _G["E"]
    t0 = time.time()
    game = BA.replay(g, p["step"])
    me, turn, seed = p["seat"], p["turn"], p["seed"]
    h = json.loads(game.hidden_json())
    hand = {c["uuid"]: c["card_id"] for c in h["players"][me]["hand"]}
    cards = lambda mv: [hand.get(u) for u in (mv.get("payload") or {}).get("selected_uuids") or []]
    out = json.loads(game.fork().decide(me, BA._G["ship"].decide_opts(seed, turn, me, None)))
    st = out.get("stats") or {}
    legal, Q = st.get("legal") or [], st.get("Q") or []
    order = sorted(range(len(legal)), key=lambda i: -Q[i])
    a = out["move"]
    ia = next((i for i in order if BA.key(legal[i]) == BA.key(a)), order[0])
    ib = next((i for i in order if sorted(cards(legal[i])) != sorted(cards(a))), None)
    r = dict(p, now_pick=cards(a), now_same=sorted(cards(a)) == sorted(p["picked"]), window=out.get("window"),
             q=[[cards(legal[i]), round(Q[i], 4)] for i in order])
    o22 = json.loads(game.fork().decide(me, _G["r22"].decide_opts(seed, turn, me, None)))
    r["r22_pick"] = cards(o22["move"])
    if ib is None:
        r["judge"] = None
        return r
    b = legal[ib]
    r["b"] = cards(b)
    r["gap"] = round(Q[ia] - Q[ib], 4)
    wa, wb = [], []
    ref = BA._G["refs"]["r13d"]
    for i in range(K):
        ws = BA.MJ.world_seed(seed, p["step"], i) ^ 0x3A3A
        ra, _ = BA.rollout_ship(game, me, a, ws, ref, BA._G["ship"])
        rb, _ = BA.rollout_ship(game, me, b, ws, ref, BA._G["ship"])
        wa.append(ra); wb.append(rb)
    r["judge"] = dict(BA.MJ.paired(wa, wb), a=wa, b=wb)
    r["sec"] = round(time.time() - t0, 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("scan", "judge"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--k", type=int, default=16)
    a = ap.parse_args()
    games = [json.loads(l) for l in open("n1_results/reaudit_wr/blunder/games.jsonl")]
    games = [g for g in games if not g.get("void")]
    byseed = {g["seed"]: g for g in games}
    ctx = get_context("spawn")
    if a.cmd == "scan":
        with ctx.Pool(a.workers, initializer=_init) as pool, open(f"{OUT}/hand_points.jsonl", "w") as f:
            for rows in pool.imap_unordered(scan_game, games):
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("SCAN_DONE", flush=True)
        return
    pts = [json.loads(l) for l in open(f"{OUT}/hand_points.jsonl")]
    rng = random.Random(20261024)
    must = [p for p in pts if p["seed"] == 3341039 and p["step"] == 84]
    rest = [p for p in pts if p not in must]
    rng.shuffle(rest)
    pick = must + rest[: a.n - len(must)]
    outp = f"{OUT}/hand_judge.jsonl"
    done = set()
    if os.path.exists(outp):
        done = {(json.loads(l)["seed"], json.loads(l)["step"]) for l in open(outp)}
    todo = [(p, byseed[p["seed"]], a.k) for p in pick if (p["seed"], p["step"]) not in done]
    print(f"judge {len(todo)}", flush=True)
    with ctx.Pool(a.workers, initializer=_init) as pool, open(outp, "a") as f:
        for n, r in enumerate(pool.imap_unordered(judge_point, todo), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            j = r.get("judge") or {}
            print(f"{n}/{len(todo)} {r['seed']}:{r['step']} diff {j.get('diff')} {j.get('ci')} {r.get('sec')}", flush=True)
    print("JUDGE_DONE", flush=True)


if __name__ == "__main__":
    main()
