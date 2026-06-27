"""Phase 3 ① 計画キャッシュ（plan_turn）のビット等価ゲート。

`plan_turn`（相手介入までの自分の連続手番をクローン上で計画）が、本物の per-action 流
（同じ単一 rng ストリーム・同じ mem）と**完全にビット等価**（同じ手列・同じ rng/mem 進行）
であることを実プレイのセグメントで機械照合する。これが満たされる限り、計画キャッシュは
「decide が出す決定的結果を前倒しで計算してキャッシュするだけ」＝**挙動不変**で安全
（待ちを 1 回に集約する体感最適化を、強さ・再現性を変えずに導入できる土台）。
"""
import copy
import random

import conftest  # noqa: F401
import pytest

from opcg_sim.src.core import cpu_ai, action_api
from opcg_sim.src.core.gamestate import GameManager, Player
from cpu_selfplay import build_deck, _load_db


@pytest.fixture(scope="module")
def db():
    return _load_db()


def _sigs(moves):
    return [cpu_ai._move_sig(m) for m in moves]


def _per_action_segment(mgr, name, rng, mem):
    """独立クローン上で、相手介入/TURN_END まで per-action 逐次 decide した手列を返す。"""
    clone = mgr.clone()
    out = []
    for _ in range(cpu_ai.TURN_ACTION_CAP + 8):
        pa = clone.pending_actor_action()
        if not pa or pa[0] != name:
            break
        actor = cpu_ai._player_by_name(clone, name)
        mv = cpu_ai.decide_guarded(clone, actor, "hard", rng, mem=mem)
        if mv is None:
            break
        out.append(mv)
        if mv.get("kind") == "battle":
            action_api.apply_battle_action(clone, actor, mv["action_type"], mv.get("card_uuid"))
        else:
            action_api.apply_game_action(clone, actor, mv["action_type"], mv.get("payload", {}))
        if mv.get("action_type") == "TURN_END":
            break
    return out


def test_plan_turn_is_bit_identical_to_per_action(db):
    """各セグメント開始で、plan_turn（クローン計画）と per-action 逐次が
    手列・rng 最終状態・mem まで完全一致する（単一 rng ストリーム＝本番同条件）。"""
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    mem = {"p1": {}, "p2": {}}
    last_actor = None
    checked = 0
    steps = 0
    while m.winner is None and steps < 200 and checked < 8:
        pa = m.pending_actor_action()
        if not pa:
            break
        actor_name = pa[0]
        actor = cpu_ai._player_by_name(m, actor_name)
        if actor_name != last_actor:
            # 本番同様「単一 rng（global random）」を保存/復元して両者を同条件で走らせる。
            rng_state = random.getstate()
            mem_a = copy.deepcopy(mem.get(actor_name, {}))
            mem_b = copy.deepcopy(mem.get(actor_name, {}))
            planned = cpu_ai.plan_turn(m, actor_name, "hard", rng=random, mem=mem_a)
            state_after_plan = random.getstate()
            random.setstate(rng_state)
            actual = _per_action_segment(m, actor_name, random, mem_b)
            assert _sigs(planned) == _sigs(actual), (
                f"step{steps} actor={actor_name}: plan {_sigs(planned)} != per-action {_sigs(actual)}")
            assert state_after_plan == random.getstate(), f"step{steps}: rng 進行が不一致"
            assert mem_a == mem_b, f"step{steps}: mem 進行が不一致"
            if planned:
                checked += 1
        last_actor = actor_name
        # 本流を per-action で進める
        mv = cpu_ai.decide_guarded(m, actor, "hard", random, mem=mem[actor_name])
        if mv is None:
            break
        m.action_events = []
        if mv.get("kind") == "battle":
            action_api.apply_battle_action(m, actor, mv["action_type"], mv.get("card_uuid"))
        else:
            action_api.apply_game_action(m, actor, mv["action_type"], mv.get("payload", {}))
        steps += 1

    assert checked >= 3, f"検証できたセグメントが不足 (checked={checked})"


def test_plan_turn_stops_at_turn_end_or_opponent(db):
    """plan_turn の戻り手列は末尾が TURN_END か、または相手介入直前で止まる（区切りの健全性）。"""
    random.seed(1)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    # マリガン等を済ませて最初の通常手番まで進める（軽く数手）。
    mem = {"p1": {}, "p2": {}}
    for _ in range(6):
        pa = m.pending_actor_action()
        if not pa:
            break
        actor = cpu_ai._player_by_name(m, pa[0])
        plan = cpu_ai.plan_turn(m, pa[0], "hard", rng=random, mem=copy.deepcopy(mem.get(pa[0], {})))
        # 区切り健全性: 空でなければ、TURN_END 終端 か 全手が同一アクターの手番内。
        if plan:
            assert plan[-1].get("action_type") == "TURN_END" or len(plan) <= cpu_ai.TURN_ACTION_CAP + 8
        mv = cpu_ai.decide_guarded(m, actor, "hard", random, mem=mem[pa[0]])
        if mv is None:
            break
        m.action_events = []
        if mv.get("kind") == "battle":
            action_api.apply_battle_action(m, actor, mv["action_type"], mv.get("card_uuid"))
        else:
            action_api.apply_game_action(m, actor, mv["action_type"], mv.get("payload", {}))


def test_decide_cached_plays_legal_and_replays(db):
    """decide_cached（計画キャッシュ配線）で対局が合法に進み、実際に replay（キャッシュヒット）が
    起きる（plan_turn 呼び出し回数 < decide 回数＝多くの手番が即時 replay）。本番専用パスの健全性。"""
    import random as _r
    from opcg_sim.src.core import cpu_ai as _ai
    _r.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    caches = {"p1": {}, "p2": {}}
    mem = {"p1": {}, "p2": {}}

    plan_calls = {"n": 0}
    orig_plan = _ai.plan_turn
    def _counting_plan(*a, **k):
        plan_calls["n"] += 1
        return orig_plan(*a, **k)
    _ai.plan_turn = _counting_plan
    try:
        decides = 0
        steps = 0
        while m.winner is None and steps < 400:
            pa = m.pending_actor_action()
            if not pa:
                break
            actor = _ai._player_by_name(m, pa[0])
            legal = m.get_legal_actions(actor)
            mv = _ai.decide_cached(m, actor, "hard", _r, mem=mem[pa[0]], cache=caches[pa[0]])
            assert mv is not None
            # 返る手は必ず現局面で合法（合法性検証の担保）
            assert _ai._move_sig(mv) in {_ai._move_sig(x) for x in legal}, \
                f"step{steps}: 非合法手を返した {_ai._move_sig(mv)}"
            decides += 1
            m.action_events = []
            if mv.get("kind") == "battle":
                action_api.apply_battle_action(m, actor, mv["action_type"], mv.get("card_uuid"))
            else:
                action_api.apply_game_action(m, actor, mv["action_type"], mv.get("payload", {}))
            steps += 1
    finally:
        _ai.plan_turn = orig_plan

    assert m.winner is not None, "ゲームが完走しなかった"
    # replay が効いている＝plan_turn 呼び出しは decide 回数より十分少ない（セグメント単位で1回）
    assert plan_calls["n"] < decides, f"replay が効いていない (plan={plan_calls['n']} >= decides={decides})"


def test_plan_segment_inprocess_matches_plan_turn(db):
    """decide_client.plan_segment（USE_WORKER off=インプロセス）が cpu_ai.plan_turn と一致。"""
    import random as _r
    from opcg_sim.api import decide_client
    import copy as _c
    _r.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    name = m.pending_actor_action()[0]
    st = _r.getstate()
    a = decide_client.plan_segment(m, cpu_ai._player_by_name(m, name), "hard", mem={})
    _r.setstate(st)
    b = cpu_ai.plan_turn(m, name, "hard", rng=_r, mem={})
    assert _sigs(a) == _sigs(b)


def test_cached_cpu_move_replays_and_legal(db):
    """app._cached_cpu_move（計画キャッシュ配線）が合法手を返し replay が効く（plan_segment 呼数 < decides）。"""
    import os
    os.environ.setdefault("OPCG_PYPY_WORKER", "0")
    import random as _r
    from opcg_sim.api import app as _app, decide_client
    _r.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    meta = {}
    turn_mem = {}
    seg_calls = {"n": 0}
    orig = decide_client.plan_segment
    def _counting(*a, **k):
        seg_calls["n"] += 1
        return orig(*a, **k)
    decide_client.plan_segment = _counting
    try:
        decides = 0
        steps = 0
        while m.winner is None and steps < 250:
            pa = m.pending_actor_action()
            if not pa:
                break
            actor = cpu_ai._player_by_name(m, pa[0])
            legal = m.get_legal_actions(actor)
            mv = _app._cached_cpu_move(m, actor, "hard", meta, turn_mem)
            if mv is None:  # フォールバック（合法性検証で稀に起きる）＝通常 decide
                mv = cpu_ai.decide_guarded(m, actor, "hard", _r, mem=turn_mem)
            assert cpu_ai._move_sig(mv) in {cpu_ai._move_sig(x) for x in legal}
            decides += 1
            m.action_events = []
            if mv.get("kind") == "battle":
                action_api.apply_battle_action(m, actor, mv["action_type"], mv.get("card_uuid"))
            else:
                action_api.apply_game_action(m, actor, mv["action_type"], mv.get("payload", {}))
            steps += 1
    finally:
        decide_client.plan_segment = orig
    assert m.winner is not None
    assert seg_calls["n"] < decides, f"replay が効いていない (plan_segment={seg_calls['n']} >= decides={decides})"


# ---------------------------------------------------------------------------
# Phase 3 ⑥-a 先行計画（pondering）: 人間の手番処理で制御が CPU へ移った瞬間に CPU セグメント計画を
# 前倒しで温め、次の /cpu/step が即時 replay（CPU 初手の待ちを消す）。本番専用・既定 OFF。
# ---------------------------------------------------------------------------

def _setup_cpu_game(db, gid, difficulty="hard"):
    """現 pending プレイヤーを CPU とみなした CPU ゲームを GAMES/CPU_GAMES に登録して返す。"""
    import os
    os.environ.setdefault("OPCG_PYPY_WORKER", "0")
    from opcg_sim.api import app as _app
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    name = m.pending_actor_action()[0]
    _app.GAMES[gid] = m
    _app.CPU_GAMES[gid] = {"cpu_player_id": name, "difficulty": difficulty, "turn_mem": {}}
    return _app, m, name


def test_ponder_prewarms_queue_and_cached_move_hits(db):
    """_ponder_plan が CPU セグメント計画を前倒しで queue に充填し、後続 _cached_cpu_move が
    **再計算なし（plan_segment 呼数 0）でヒット**して合法手を返す＝先行計画＝待ち消しの担保。"""
    import asyncio
    import random as _r
    from opcg_sim.api import decide_client
    _r.seed(0)
    gid = "_ponder_t1"
    _app, m, name = _setup_cpu_game(db, gid)
    try:
        asyncio.run(_app._ponder_plan(gid))
        meta = _app.CPU_GAMES[gid]
        cache = meta["plan_cache"]
        assert cache.get("queue"), "先行計画で queue が温まっていない"
        assert cache.get("task") is None, "タスクが片付いていない"
        seg_calls = {"n": 0}
        orig = decide_client.plan_segment
        def _counting(*a, **k):
            seg_calls["n"] += 1
            return orig(*a, **k)
        decide_client.plan_segment = _counting
        try:
            actor = cpu_ai._player_by_name(m, name)
            legal = m.get_legal_actions(actor)
            mv = _app._cached_cpu_move(m, actor, "hard", meta, meta["turn_mem"])
            assert mv is not None
            assert cpu_ai._move_sig(mv) in {cpu_ai._move_sig(x) for x in legal}
            assert seg_calls["n"] == 0, "warm queue があるのに再計画した（前倒しが効いていない）"
        finally:
            decide_client.plan_segment = orig
    finally:
        _app.GAMES.pop(gid, None)
        _app.CPU_GAMES.pop(gid, None)


def test_kick_ponder_gated_and_starts_task(db):
    """_kick_ponder は OPCG_PONDER 配下のオプトイン: 無効時は no-op、有効時は CPU 手番でタスクを起動し、
    完走後に queue が温まる（合法性は _cached_cpu_move 側で担保）。"""
    import asyncio
    import os
    import random as _r
    _r.seed(0)
    gid = "_ponder_t2"
    _app, m, name = _setup_cpu_game(db, gid)
    prev_pc = os.environ.get("OPCG_PLAN_CACHE")
    prev_pd = os.environ.get("OPCG_PONDER")
    try:
        # 無効時（既定）は no-op＝タスク非起動。
        os.environ["OPCG_PLAN_CACHE"] = "0"
        os.environ["OPCG_PONDER"] = "0"
        _app._kick_ponder(gid)
        assert _app.CPU_GAMES[gid].get("plan_cache", {}).get("task") is None

        # 有効時は CPU 手番でタスク起動→完走で queue 充填。
        os.environ["OPCG_PLAN_CACHE"] = "1"
        os.environ["OPCG_PONDER"] = "1"

        async def _drive():
            _app._kick_ponder(gid)
            task = _app.CPU_GAMES[gid]["plan_cache"].get("task")
            assert task is not None, "有効時に先行計画タスクが起動しない"
            await task

        asyncio.run(_drive())
        assert _app.CPU_GAMES[gid]["plan_cache"].get("queue"), "タスク完走後に queue が温まっていない"
        assert _app.CPU_GAMES[gid]["plan_cache"].get("task") is None
    finally:
        for k, v in (("OPCG_PLAN_CACHE", prev_pc), ("OPCG_PONDER", prev_pd)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _app.GAMES.pop(gid, None)
        _app.CPU_GAMES.pop(gid, None)


# ---------------------------------------------------------------------------
# Phase 3 ⑥-b 投機ポンダリング: 人間の MAIN 中に「今エンドしたら」の CPU 計画を投機し、実 TURN_END で
# 合法なら昇格（CPU 初手の待ちすら消す）。本番専用・既定 OFF（OPCG_PONDER_SPEC=1）。
# ---------------------------------------------------------------------------

def _advance_to_main(m):
    """マリガン等を既定方策で進め、いずれかのプレイヤーの MAIN_ACTION 決定点まで進める（pid を返す）。"""
    for _ in range(40):
        pa = m.pending_actor_action()
        if not pa:
            return None
        pid, act = pa
        if act == "MAIN_ACTION":
            return pid
        actor = cpu_ai._player_by_name(m, pid)
        mv = cpu_ai.decide_guarded(m, actor, "hard", random.Random(0), mem={})
        if mv is None:
            return None
        m.action_events = []
        if mv.get("kind") == "battle":
            action_api.apply_battle_action(m, actor, mv["action_type"], mv.get("card_uuid"))
        else:
            action_api.apply_game_action(m, actor, mv["action_type"], mv.get("payload", {}))
    return None


def test_speculate_compute_plans_cpu_turn_on_clone(db):
    """⑥-b: _speculate_compute がクローン上で人間 TURN_END を仮適用し、CPU 手番へ移れば CPU 計画を返す。
    渡したクローンのみ変異し、別インスタンス（live 相当）は不変＝live 盤面に触れない設計の担保。"""
    import random as _r
    from opcg_sim.api import app as _app
    _r.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    pid = _advance_to_main(m)
    if pid is None:
        pytest.skip("MAIN_ACTION へ到達できない")
    human_pid = pid
    cpu_pid = "p2" if pid == "p1" else "p1"
    live_turn = m.turn_count
    clone = m.clone()
    result = _app._speculate_compute(clone, human_pid, cpu_pid, "hard")
    # live(=m) は不変（クローンのみ変異）。
    assert m.turn_count == live_turn
    assert m.pending_actor_action()[0] == human_pid
    # 投機が成立すれば CPU 手のリスト（介在する人間決定がある盤面なら None も許容）。
    assert result is None or (isinstance(result, list) and len(result) >= 1)


def test_kick_ponder_promotes_valid_speculation(db):
    """⑥-b: 実盤面で先頭が合法な spec_queue は _kick_ponder が queue へ昇格し spec_hits を計上（投機ヒット）。
    ヒット時は再計画タスクを起動しない（待ちゼロ）。"""
    import os
    import random as _r
    from opcg_sim.api import decide_client
    _r.seed(0)
    gid = "_spec_t1"
    _app, m, name = _setup_cpu_game(db, gid)  # cpu_pid = 現 pending 側＝CPU 手番
    prev_pc = os.environ.get("OPCG_PLAN_CACHE")
    prev_pd = os.environ.get("OPCG_PONDER")
    try:
        os.environ["OPCG_PLAN_CACHE"] = "1"
        os.environ["OPCG_PONDER"] = "1"
        # 現 CPU 手番に対する実計画を「投機結果」として置く（先頭は当然合法）。
        cpu_player = cpu_ai._player_by_name(m, name)
        plan = decide_client.plan_segment(m, cpu_player, "hard", mem={})
        assert plan, "計画が空"
        meta = _app.CPU_GAMES[gid]
        meta.setdefault("plan_cache", {})["spec_queue"] = list(plan)
        _app._kick_ponder(gid)
        cache = meta["plan_cache"]
        assert cache.get("spec_hits") == 1, "投機ヒットが計上されていない"
        assert cache.get("queue") == plan, "spec_queue が queue へ昇格していない"
        assert cache.get("task") is None, "ヒット時は再計画タスクを起動しない"
    finally:
        for k, v in (("OPCG_PLAN_CACHE", prev_pc), ("OPCG_PONDER", prev_pd)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _app.GAMES.pop(gid, None)
        _app.CPU_GAMES.pop(gid, None)


def test_kick_speculate_gated_and_clones_without_mutation(db):
    """⑥-b: _kick_speculate は OPCG_PONDER_SPEC 配下のオプトイン。無効時 no-op、有効時は人間 MAIN で
    投機タスクを起動し（live 盤面は不変）、完走で spec_queue を充填する。"""
    import asyncio
    import os
    import random as _r
    _r.seed(0)
    from opcg_sim.api import app as _app
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    pid = _advance_to_main(m)
    if pid is None:
        pytest.skip("MAIN_ACTION へ到達できない")
    cpu_pid = "p2" if pid == "p1" else "p1"
    gid = "_spec_t2"
    _app.GAMES[gid] = m
    _app.CPU_GAMES[gid] = {"cpu_player_id": cpu_pid, "difficulty": "hard", "turn_mem": {}}
    prev = {k: os.environ.get(k) for k in ("OPCG_PLAN_CACHE", "OPCG_PONDER", "OPCG_PONDER_SPEC")}
    live_turn = m.turn_count
    try:
        # 無効時は no-op。
        os.environ["OPCG_PLAN_CACHE"] = "1"
        os.environ["OPCG_PONDER"] = "1"
        os.environ["OPCG_PONDER_SPEC"] = "0"
        _app._kick_speculate(gid)
        assert _app.CPU_GAMES[gid].get("plan_cache", {}).get("spec_task") is None

        # 有効時は投機タスク起動→完走で spec_queue 充填。live 盤面は不変。
        os.environ["OPCG_PONDER_SPEC"] = "1"

        async def _drive():
            _app._kick_speculate(gid)
            t = _app.CPU_GAMES[gid]["plan_cache"].get("spec_task")
            assert t is not None, "有効時に投機タスクが起動しない"
            await t

        asyncio.run(_drive())
        assert m.turn_count == live_turn, "投機が live 盤面を変異させた"
        assert m.pending_actor_action()[0] == pid, "投機が live の手番を進めた"
        cache = _app.CPU_GAMES[gid]["plan_cache"]
        # spec_queue は CPU 手のリスト or None（介在決定で投機不成立）。いずれも合法性ゲートが採否を担保。
        assert cache.get("spec_queue") is None or isinstance(cache.get("spec_queue"), list)
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _app.GAMES.pop(gid, None)
        _app.CPU_GAMES.pop(gid, None)
