"""生成で捨てられた局を、生成と同じ設定で打ち直して捨てた理由を見る（1 回きり・WP rs-overflow-trigger）。
    python drop_probe.py SEED [SEED ...]"""
import json, sys, time, traceback
sys.path.insert(1, ".")
from opcg_sim.loop import record_gen as RG, driver as DR, decks as D, engine as E
import os
extra = json.loads(os.environ.get("PROBE_EXTRA", "{}"))
search = {**extra, "worlds": 1, "select_rule": "q_min_n", "q_min_frac": 0.125, "root_prior_temp": 2.0, "budget": 0}
RG._init_worker(1280, "opcg_sim/data/learned/nrel_r22.npz", 0.25, 4, "real", True, 0.03, 0.03, search, 0.0, None)
for seed in map(int, sys.argv[1:]):
    spec, db = RG._G["spec"], RG._G["db"]
    rec = RG._Recorder(seed=seed, db=db, eps_play=0.03, eps_hold=0.03, eps_target=0.0)
    t = time.time()
    tail = []
    def obs(game, name, turn, step, out, move):
        rec(game, name, turn, step, out, move)
        p = json.loads(game.pending_json()); h = json.loads(game.hidden_json())["manager"]
        u = (move.get("payload") or {}).get("uuid") or move.get("card_uuid")
        cid = None
        if u:
            for P in json.loads(game.hidden_json())["players"].values():
                for z, cs in P.items():
                    for c in (cs if isinstance(cs, list) else [cs] if isinstance(cs, dict) else []):
                        if isinstance(c, dict) and c.get("uuid") == u:
                            cid = (c.get("card_id"), z)
        tail.append((step, name, turn, p.get("action"), (p.get("message") or "")[:50], out.get("kind"), move.get("action_type"), cid, h["phase"], h["active_battle"] is not None))
        del tail[:-8]
    try:
        la, lb = D.leader_pair(db, seed, "random")
        p1, p2, kinds = D.build_pair(db, la, lb, seed, "real", with_kinds=True)
        res = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, max_steps=RG.MAX_STEPS, observer=obs, post=rec.post, swap=rec.swap)
        h = json.loads(res["game"].hidden_json())
        print("  final", {k: {z: (len(v) if isinstance(v, list) else (v or {}).get("card_id") if isinstance(v, dict) else v) for z, v in P.items() if z in ("life", "deck", "hand", "field", "trash", "leader")} for k, P in h["players"].items()})
        print(seed, "winner", res["winner"], "turns", res["turns"], "steps", res["steps"], "acts", res["acts"], "MAX_STEPS", RG.MAX_STEPS, "sec", round(time.time() - t, 1))
    except Exception as exc:
        print(seed, "EXC", type(exc).__name__, exc, "sec", round(time.time() - t, 1))
        traceback.print_exc()
    for x in tail: print("  ", x)
