"""WP weak-points §1: 守りの見逃し（overflow_trigger の再監査・確かめ後 甲 6）を今の本線で打ち直して読む（1 回きり）。
器は defense_audit.replay_to（両席 r13d・世界 1 本・1,280 回・予算なし・証明 off・random × synth_roles）。
点ごとに: 再現（同じ手か）・要求・盤面の要約・実局の decide（窓の世界・木か・候補の値）・守れた手（verify の行）・
読みの形を替えた decide（短い木をライフ 2〜3 へ・読み 4 倍・出荷のネット r22〔世界 2 本〕）。"""
import argparse, json, os, sys
sys.path.insert(0, "tests/scripts")
import defense_audit as DA
import lethal_audit as LA
from opcg_sim.loop import engine as E

PTS = [(3351006, "p1", 159), (3351014, "p2", 208), (3351102, "p2", 99), (3351122, "p1", 182),
       (3351127, "p2", 110), (3351191, "p1", 114)]
VER = {}
for l in open("n1_results/overflow_trigger/defense/verify.jsonl"):
    r = json.loads(l)
    VER[(r["seed"], r["step"])] = r
a = argparse.Namespace(net="opcg_sim/data/learned/nrel_r13d.npz", sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=1,
                       lethal_proof=None, window_worlds=None)

rows = []
only = set(int(x) for x in sys.argv[1:])
for seed, seat, step in PTS:
    if only and seed not in only:
        continue
    game, move, got = DA.replay_to(seed, a, seat, step)
    out = got["out"]
    opts = json.loads(got["spec"].decide_opts(seed, got["turn"], seat, None))
    pend = json.loads(game.pending_json())
    st = out.get("stats") or {}
    legal = st.get("legal") or []
    v = VER[(seed, step)]
    r = {"seed": seed, "seat": seat, "step": step, "turn": got["turn"], "pending": pend.get("action"), "msg": pend.get("message"),
         "board": LA.summarize_board(game, seat),
         "live": LA.describe(game, move), "same_as_audit": LA.describe(game, move) == v["fresh_move"] or None,
         "kind": out.get("kind"), "window": out.get("window"),
         "cands": [{"move": LA.describe(game, m), "Q": round(q, 3)} for m, q in zip(legal, st.get("Q") or [])],
         "safe": [x["move"] for x in v["rows"] if (x.get("ko_full") or {}).get("res") is False]}
    for tag, extra in (("tree_life2", {"window_tree_life": 2}), ("tree_life3", {"window_tree_life": 3}),
                       ("sims_x4", {"sims": 4 * E.PLAY_SIMS}), ("worlds64", {"window_worlds": 64}),
                       ("r22_ship", {"net": None, "worlds": 2})):
        o = dict(opts, **extra)
        if tag == "r22_ship":
            o.pop("net", None)
            o2 = json.loads(E.SeatSpec(None, sims=E.PLAY_SIMS, worlds=2, budget=E.PLAY_BUDGET, lethal_proof="root").decide_opts(seed, got["turn"], seat, None))
            o = o2
        d = json.loads(game.fork().decide(seat, json.dumps(o)))
        dst = d.get("stats") or {}
        r[tag] = {"move": LA.describe(game, d["move"]), "window": d.get("window"), "kind": d.get("kind"),
                  "cands": [[LA.describe(game, m), round(q, 3)] for m, q in zip(dst.get("legal") or [], dst.get("Q") or [])]}
    print(json.dumps(r, ensure_ascii=False), flush=True)
    rows.append(r)
json.dump(rows, open("n1_results/weak_points/def_points%s.json" % ("_" + "_".join(sys.argv[1:]) if only else ""), "w"),
          ensure_ascii=False, indent=1)
