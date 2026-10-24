"""WP weak-points §5: ライフの【トリガー】（と任意の誘発）の発動確認の点を、生成と同じ設定の局で集め、
発動する／しないを固定の審判（`move_judge.rollout`・`paired`）で比べる（1 回きり・読み取りだけ）。

局: 生成役 r22・gen_n51.json の引数（1,280 回・世界 1 本・枝予算なし・q_min_n 1/8・根の事前分布の温度 2・
両方向 ε 3%・Dirichlet 0.25・温度 4 ターン・decks real）・seed 3671000〜3671199。
点: 要求が CONFIRM_TRIGGER（can_skip）で探索の候補が 2 手（accepted true／false）の点。点ごとに盤面を fork して持ち、
局の後で K 局ずつ審判（r22・160 回・世界 1 本・公平・証明 off）が両席最後まで打つ（同じ配り方・同じ乱数で両手＝対の差）。
出力: points.jsonl（1 行 1 点）・games.jsonl（1 行 1 局）。終えた seed は飛ばす（再開できる）。

    python n1_results/weak_points/trig_collect.py --seed-base 3671000 --games 200 --workers 4 --k 24
"""
import argparse, json, os, sys, time
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "tests", "scripts")); sys.path.insert(0, os.path.join(REPO, "tests"))
os.environ.setdefault("OPCG_LOG_SILENT", "1")
from multiprocessing import get_context

OUT = os.environ.get("TRIG_OUT") or os.path.join(REPO, "n1_results", "weak_points", "trig")
NET = "opcg_sim/data/learned/nrel_r22.npz"
SEARCH = {"worlds": 1, "budget": 0, "select_rule": "q_min_n", "q_min_frac": 0.125, "root_prior_temp": 2.0}
_G = {}


def _init(k):
    os.chdir(REPO)
    import _bootstrap  # noqa: F401
    from opcg_sim.loop import record_gen as RG, engine as E
    RG._init_worker(1280, NET, 0.25, 4, "real", True, 0.03, 0.03, dict(SEARCH), 0.0, None)
    _G["RG"], _G["E"] = RG, E
    _G["ref"] = E.SeatSpec(NET, sims=160)
    _G["k"] = k


def _card_of(h, uuid):
    for pid, P in h["players"].items():
        for z, cs in P.items():
            for c in (cs if isinstance(cs, list) else [cs] if isinstance(cs, dict) else []):
                if isinstance(c, dict) and c.get("uuid") == uuid:
                    return pid, c.get("card_id"), z
    return None, None, None


def play_and_judge(seed):
    import move_judge as MJ
    from opcg_sim.loop import decks as D, driver as DR
    RG, E = _G["RG"], _G["E"]
    spec, db = RG._G["spec"], RG._G["db"]
    rec = RG._Recorder(seed=seed, db=db, eps_play=spec.eps_play, eps_hold=spec.eps_hold, eps_target=0.0)
    pts = []
    t0 = time.time()

    def obs(game, name, turn, step, out, move):
        rec(game, name, turn, step, out, move)
        pend = json.loads(game.pending_json())
        if pend.get("action") != "CONFIRM_TRIGGER":
            return
        st = out.get("stats") or {}
        legal = st.get("legal") or []
        h = json.loads(game.hidden_json())
        src = pend.get("source_card_uuid")
        owner, cid, zone = _card_of(h, src)
        me = h["players"][name]; op = h["players"]["p2" if name == "p1" else "p1"]
        info = {"seed": seed, "step": step, "turn": turn, "seat": name, "turn_player": h["manager"]["turn_player"],
                "msg": pend.get("message"), "life_trigger": (pend.get("message") or "").startswith("【トリガー】"),
                "card": cid, "zone": zone, "can_skip": pend.get("can_skip"),
                "my_life": len(me["life"]), "opp_life": len(op["life"]), "my_hand": len(me["hand"]), "opp_hand": len(op["hand"]),
                "my_field": len(me.get("field") or []), "opp_field": len(op.get("field") or []),
                "battle": h["manager"]["active_battle"] is not None, "kind": out.get("kind"),
                "window": out.get("window"),
                "cpu": (move.get("payload") or {}).get("accepted"), "n_legal": len(legal),
                "q": {str((m.get("payload") or {}).get("accepted")): (round(q, 4) if q is not None else None)
                      for m, q in zip(legal, st.get("Q") or [None] * len(legal))}}
        if len(legal) == 2:
            pts.append((info, game.fork(), legal))
        else:
            pts.append((info, None, None))

    try:
        la, lb = D.leader_pair(db, seed, "random")
        p1, p2 = D.build_pair(db, la, lb, seed, "real")
        res = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, max_steps=RG.MAX_STEPS, observer=obs, swap=rec.swap)
        g = {"seed": seed, "winner": res["winner"], "turns": res["turns"], "steps": res["steps"], "leaders": [la, lb]}
    except Exception as exc:  # noqa: BLE001
        g = {"seed": seed, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    g["sec_play"] = round(time.time() - t0, 1)
    rows = []
    for info, game, legal in pts:
        info["winner"] = g.get("winner")
        if game is not None:
            ia = next(i for i, m in enumerate(legal) if (m.get("payload") or {}).get("accepted") is True)
            ib = 1 - ia
            wa, wb = [], []
            t1 = time.time()
            for i in range(_G["k"]):
                ws = MJ.world_seed(seed, info["step"], i)
                ra, _ = MJ.rollout(game, info["seat"], legal[ia], ws, _G["ref"])
                rb, _ = MJ.rollout(game, info["seat"], legal[ib], ws, _G["ref"])
                wa.append(ra); wb.append(rb)
            # a＝発動する・b＝発動しない（差＝断る − 発動する）
            info["judge"] = dict(MJ.paired(wa, wb), a=wa, b=wb, sec=round(time.time() - t1, 1))
        rows.append(info)
    g["n_points"] = len(rows)
    g["sec"] = round(time.time() - t0, 1)
    return g, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-base", type=int, default=3671000)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--k", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    gp, pp = os.path.join(OUT, "games.jsonl"), os.path.join(OUT, "points.jsonl")
    done = set()
    if os.path.exists(gp):
        done = {json.loads(l)["seed"] for l in open(gp)}
    todo = [a.seed_base + i for i in range(a.games) if a.seed_base + i not in done]
    print(f"todo {len(todo)}", flush=True)
    with get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.k,)) as pool, \
            open(gp, "a") as fg, open(pp, "a") as fp:
        for n, (g, rows) in enumerate(pool.imap_unordered(play_and_judge, todo), 1):
            for r in rows:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
            fg.write(json.dumps(g, ensure_ascii=False) + "\n")
            fp.flush(); fg.flush()
            print(f"{n}/{len(todo)} seed {g['seed']} pts {g.get('n_points')} {g.get('sec')}s {g.get('error','')}", flush=True)
    print("TRIG_DONE", flush=True)


if __name__ == "__main__":
    main()
