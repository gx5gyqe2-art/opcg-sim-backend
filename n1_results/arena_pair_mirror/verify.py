"""WP arena-pair-mirror の検算: 旧規約／新規約の 1 ペアを同じ seed で打ち比べる。
usage: verify.py <label> <rule old|new> <cand> <best> <leaders> <decks> <seed0> <n> <out.jsonl>"""
import json, sys, multiprocessing as mp
sys.path.insert(0, "/home/user/opcg-sim-backend")
from opcg_sim.loop import arena as A

def old_detail(seed):
    db, cand, best = A._G["db"], A._G["cand"], A._G["best"]
    la, lb = A.D.leader_pair(db, seed, A._G["leaders_mode"])
    decks = A._G["decks"]
    try:
        ab = A.D.build_pair(db, la, lb, seed, decks)
        ba = A.D.build_pair(db, lb, la, seed, decks)
        a = A.DR.run_game(seed, {"p1": cand, "p2": best}, *ab)
        b = A.DR.run_game(seed, {"p1": best, "p2": cand}, *ba)
        if a["winner"] is None or b["winner"] is None:
            raise A.DR.GameAborted("void")
    except Exception as e:
        return {"seed": seed, "score": None, "void": str(e)[:80]}
    wa = 1.0 if a["winner"] == "p1" else 0.0
    wb = 1.0 if b["winner"] == "p2" else 0.0
    return {"seed": seed, "score": wa + wb, "games": [wa, wb], "leaders": [la, lb],
            "cand_leaders": [ab[0][0], ba[1][0]], "turns": [a["turns"], b["turns"]]}

def main():
    label, rule, cand, best, leaders, decks, s0, n, out = sys.argv[1:]
    fn = old_detail if rule == "old" else A.play_pair_detail
    seeds = list(range(int(s0), int(s0) + int(n)))
    with mp.Pool(4, initializer=A.init_pool,
                 initargs=(cand, best, None, leaders, decks, 1800, 160)) as pool:
        rows = pool.map(fn, seeds)
    with open(out, "w") as f:
        for r in rows:
            r["label"] = label; r["rule"] = rule
            f.write(json.dumps(r) + "\n")
    sc = [r["score"] for r in rows if r["score"] is not None]
    print(label, rule, "pairs", len(sc), "void", len(rows) - len(sc),
          "wr", round(sum(sc) / (2 * len(sc)), 4),
          "dist", {k: sc.count(k) for k in (0.0, 1.0, 2.0)}, flush=True)

if __name__ == "__main__":
    main()
