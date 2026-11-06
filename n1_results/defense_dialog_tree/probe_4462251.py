"""4462251・172: 受けた後（と断った後）の盤面で、今のターンの残りを出荷の設定が打つ手の列と、各窓の決め方・候補の値を並べる（1 回きり）。"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "scripts")); sys.path.insert(0, ROOT); os.chdir(ROOT)
import blunder_audit as BA
import defense_audit as DA
from opcg_sim.loop import engine as E, decks as D
E.engine()
NET = "opcg_sim/data/learned/nrel_r29f4.npz"
a = argparse.Namespace(net=None, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=1, lethal_proof=None, window_worlds=None)
game, move, got = DA.replay_to(4462251, a, "p1", 172)
h = json.loads(game.hidden_json())
for s in ("p1", "p2"):
    P = h["players"][s]
    print(s, "leader", P["leader"]["card_id"], "rest", P["leader"].get("is_rest"), "life", len(P["life"]),
          "field", [(c["card_id"], c.get("is_rest"), c.get("attached_don")) for c in P["field"]],
          "hand", [c["card_id"] for c in P["hand"]], "don_active", len(P["don"]["active"]))
print("battle", h["manager"].get("active_battle"))
print("pending", json.dumps(json.loads(game.pending_json()), ensure_ascii=False)[:600])
legal = json.loads(game.legal_json("p1"))
ship = E.SeatSpec(NET, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=E.PLAY_WORLDS, lethal_proof=E.PLAY_LETHAL_PROOF)
for li, mv in enumerate(legal):
    print("=== candidate", li, BA.desc(game, mv))
    g = game.fork()
    BA.apply(g, "p1", mv)
    t0 = int(g.turn_count)
    for k in range(40):
        if g.winner() is not None or int(g.turn_count) != t0:
            break
        pend = json.loads(g.pending_json())
        name = pend["player_id"]
        out = json.loads(g.decide(name, ship.decide_opts(4462251, t0, name, None)))
        st = out.get("stats") or {}
        lg = json.loads(g.legal_json(name))
        print(f"  [{name}] {pend.get('action')} kind={out.get('kind')} win={out.get('window')} "
              f"move={BA.desc(g, out['move'])[:90]}  Q={[round(x,3) for x in st.get('Q', [])][:6]}  cands={[BA.desc(g, m)[:60] for m in lg][:6]}")
        BA.apply(g, name, out["move"])
    print("  winner", g.winner(), "turn", g.turn_count)
