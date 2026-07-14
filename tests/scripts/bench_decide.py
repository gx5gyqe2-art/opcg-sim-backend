"""CPython vs PyPy 用の decide ベンチ（一時・spike）。

hard vs hard の決定論セルフプレイを回し、各 decider 呼び出し（=探索）の所要を計測する。
同一スクリプトを CPython と PyPy で実行し、per-decide の中央値/平均を比較する。
JIT ウォームアップ用に最初の1ゲームは計測から除外する。
"""
import os, sys, time, statistics, random
os.environ.setdefault("OPCG_LOG_SILENT", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401  (tests/harness を path に載せる＝cpu_selfplay 等を解決)

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_ai
from cpu_selfplay import _load_db, build_deck, check_invariants


def timed_game(seed, db, diff="hard", max_steps=400, collect=None):
    random.seed(seed)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    mgr = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    mgr.start_game()
    mem = {}
    props = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {})
    KEY_PID = props.get('PLAYER_ID', 'player_id')
    step = 0
    while mgr.winner is None and step < max_steps:
        pend = mgr.get_pending_request()
        if not pend:
            break
        pid = pend[KEY_PID]
        actor = mgr.p1 if mgr.p1.name == pid else mgr.p2
        t0 = time.perf_counter()
        move = cpu_ai.decide_guarded(mgr, actor, diff, random, mem.setdefault(actor.name, {}))
        dt = (time.perf_counter() - t0) * 1000.0
        if collect is not None and dt >= 1.0:   # 1ms 未満の自明手は除外（探索のある手のみ）
            collect.append(dt)
        if move is None:
            break
        mgr.action_events = []
        if move["kind"] == "battle":
            action_api.apply_battle_action(mgr, actor, move["action_type"], move.get("card_uuid"))
        else:
            action_api.apply_game_action(mgr, actor, move["action_type"], move.get("payload", {}))
        step += 1
    return step


def main():
    db = _load_db()
    impl = sys.implementation.name
    ver = ".".join(map(str, sys.version_info[:3]))
    # ウォームアップ（PyPy の JIT を温める。CPython では無害）。
    for s in (1, 2):
        timed_game(s, db, max_steps=200)
    # 計測本番
    samples = []
    wall0 = time.perf_counter()
    total_steps = 0
    for s in (10, 11, 12):
        total_steps += timed_game(s, db, max_steps=400, collect=samples)
    wall = time.perf_counter() - wall0
    samples.sort()
    n = len(samples)
    print(f"\n=== {impl} {ver} ===")
    print(f"decides(>=1ms)={n}  steps={total_steps}  wall={wall:.2f}s")
    if n:
        print(f"per-decide ms: mean={statistics.mean(samples):.1f} "
              f"median={statistics.median(samples):.1f} "
              f"p90={samples[int(n*0.9)-1]:.1f} max={samples[-1]:.1f}")


if __name__ == "__main__":
    main()
