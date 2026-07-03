"""実 PyPy ワーカー e2e スモーク（手動・CI 非対象）。CPython→PyPy の実 IPC で同一手を確認。"""
import os, sys, time, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import _bootstrap  # noqa: E402,F401  (tests/harness を path に載せる＝cpu_selfplay 等を解決)
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_ai
from cpu_selfplay import _load_db, build_deck
from opcg_sim.api import decide_client

KEY = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {}).get('PLAYER_ID', 'player_id')


def key(m):
    return None if m is None else (m.get("kind"), m.get("action_type"), m.get("card_uuid"), repr(m.get("payload")))


def main():
    random.seed(7); db = _load_db()
    l1, c1 = build_deck(db, "p1"); l2, c2 = build_deck(db, "p2")
    mgr = GameManager(Player("p1", c1, l1), Player("p2", c2, l2)); mgr.start_game()
    mem = {}
    for _ in range(8):
        if mgr.winner: break
        pend = mgr.get_pending_request()
        if not pend: break
        pid = pend[KEY]; actor = mgr.p1 if mgr.p1.name == pid else mgr.p2
        mv = cpu_ai.decide_guarded(mgr, actor, "hard", mem=mem.setdefault(pid, {}))
        if mv is None: break
        mgr.action_events = []
        if mv["kind"] == "battle":
            action_api.apply_battle_action(mgr, actor, mv["action_type"], mv.get("card_uuid"))
        else:
            action_api.apply_game_action(mgr, actor, mv["action_type"], mv.get("payload", {}))
    pid = mgr.get_pending_request()[KEY]; actor = mgr.p1 if pid == "p1" else mgr.p2

    st = random.getstate()
    random.setstate(st)
    ref = cpu_ai.decide_guarded(mgr, actor, "hard", mem={})
    print("USE_WORKER =", decide_client.USE_WORKER, flush=True)

    decide_client.spawn_worker()
    for _ in range(150):
        if os.path.exists(decide_client.SOCK_PATH): break
        time.sleep(0.1)
    time.sleep(1.5)  # import + cache ロード待ち

    random.setstate(st)
    t0 = time.perf_counter()
    viaworker = decide_client.decide(mgr, actor, "hard", mem={})
    dt = (time.perf_counter() - t0) * 1000

    print("inprocess :", key(ref), flush=True)
    print("via PyPy  :", key(viaworker), f"({dt:.0f}ms incl IPC)", flush=True)
    print("E2E SAME MOVE:", key(ref) == key(viaworker), flush=True)


if __name__ == "__main__":
    main()
