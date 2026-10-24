"""守りの再監査の修正前（rs-commit-hidden の new300）の確かめ済みの見逃し 6 点で、旧い旗・各修正だけの形が何を選ぶかを見る（WP rs-window-read・1 回きり）。

局を修正後の既定（§7.24 の設定）で打ち直し、見逃しの点の盤面で、探索の乱数を 8 通りに替えて各設定に選ばせる。
"""
import json
import os
import sys
from multiprocessing import get_context

REPO = "/home/user/opcg-sim-backend"
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("OPCG_LOG_SILENT", "1")
from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402

PTS = [(3013026, 144), (3013119, 169), (3013147, 122), (3013194, 89), (3013217, 157), (3013280, 148)]
SETS = {"fixed": {}, "legacy": {"window_carry_legacy": True, "window_worlds_legacy": True, "window_tree_legacy": True},
        "no_tree": {"window_tree_legacy": True}, "tree_only": {"window_carry_legacy": True, "window_worlds_legacy": True},
        "tree4x": {"sims": 5120}}


class Stop(Exception):
    pass


def one(pt):
    seed, target = pt
    E.engine()
    db = D.load_db()
    spec = E.SeatSpec(None, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, **SETS["legacy"])
    base_spec = E.SeatSpec(None, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET)
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "synth_roles")
    res = {"seed": seed, "step": target}

    def obs(game, name, turn, step, out, move):
        if step != target:
            return
        res["live"] = json.dumps(move, ensure_ascii=False)[:90]
        res["window"] = out.get("window")
        res["life"] = len(json.loads(game.state_json() if hasattr(game, "state_json") else "{}").get("players", {}).get(name, {}).get("life", [])) if False else None
        base = json.loads(base_spec.decide_opts(seed, turn, name, None))
        for tag, extra in SETS.items():
            picks = {}
            for j in range(8):
                o = dict(base, **extra, search_seed=(base["search_seed"] + 7919 * j) & 0xFFFFFFFFFFFFFFFF)
                d = json.loads(game.fork().decide(name, json.dumps(o)))
                m = d["move"]
                k = m["action_type"] + (":" + str(m.get("card_uuid"))[:8] if m.get("card_uuid") else "")
                picks[k] = picks.get(k, 0) + 1
            res[tag] = picks
        raise Stop()

    try:
        DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, observer=obs)
    except Stop:
        pass
    return res


if __name__ == "__main__":
    out = []
    with get_context("fork").Pool(4) as pool:
        for r in pool.imap_unordered(one, PTS):
            print(json.dumps(r, ensure_ascii=False), flush=True)
            out.append(r)
    json.dump(sorted(out, key=lambda r: r["seed"]), open("n1_results/window_read/defense_points_before.json", "w"), ensure_ascii=False, indent=1)
