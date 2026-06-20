"""decide のホットパス内訳を cProfile で測る（最適化対象の特定用・手動）。"""
import os, sys, cProfile, pstats, io, random
os.environ.setdefault("OPCG_LOG_SILENT", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_ai
from cpu_selfplay import _load_db, build_deck

KEY = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {}).get('PLAYER_ID', 'player_id')


def midgames(seeds=(10, 11, 12), steps=14, diff="hard"):
    """各シードで mid-game に到達した (manager, actor) を集める（CPU 手番）。"""
    db = _load_db()
    out = []
    for sd in seeds:
        random.seed(sd)
        l1, c1 = build_deck(db, "p1"); l2, c2 = build_deck(db, "p2")
        mgr = GameManager(Player("p1", c1, l1), Player("p2", c2, l2)); mgr.start_game()
        mem = {}
        for _ in range(steps):
            if mgr.winner: break
            pend = mgr.get_pending_request()
            if not pend: break
            pid = pend[KEY]; actor = mgr.p1 if mgr.p1.name == pid else mgr.p2
            mv = cpu_ai.decide_guarded(mgr, actor, diff, mem=mem.setdefault(pid, {}))
            if mv is None: break
            mgr.action_events = []
            if mv["kind"] == "battle":
                action_api.apply_battle_action(mgr, actor, mv["action_type"], mv.get("card_uuid"))
            else:
                action_api.apply_game_action(mgr, actor, mv["action_type"], mv.get("payload", {}))
        pend = mgr.get_pending_request()
        if pend:
            pid = pend[KEY]; out.append((mgr, mgr.p1 if pid == "p1" else mgr.p2))
    return out


def main():
    positions = midgames()
    print(f"profiling {len(positions)} midgame decides (hard)...")
    pr = cProfile.Profile()
    pr.enable()
    for mgr, actor in positions:
        cpu_ai.decide_guarded(mgr, actor, "hard", rng=random.Random(0), mem={})
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    print("\n===== TOP 25 by cumulative ====="); ps.sort_stats("cumulative").print_stats(25)
    print(s.getvalue()[:4000])
    s2 = io.StringIO(); ps2 = pstats.Stats(pr, stream=s2)
    print("\n===== TOP 25 by tottime (self) ====="); ps2.sort_stats("tottime").print_stats(25)
    print(s2.getvalue()[:4000])


if __name__ == "__main__":
    main()
