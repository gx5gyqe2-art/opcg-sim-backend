"""CPU 思考の駆動（計画キャッシュ／pondering ⑥-a／投機 ⑥-b）。

**並行処理の不変条件**（挙動不変で維持）:
  - live 盤面を OS スレッドへ渡さない。clone は**メインスレッドで原子的**に取り、重い計算だけを
    `asyncio.to_thread` へ逃がす（スレッド側 deepcopy とメインスレッドの盤面変更の競合を防ぐ）。
  - 投機は世代 `spec_gen` を進めて旧結果を supersede。採否は最終的に `_kick_ponder` の合法性ゲートが担保。
  - すべて既定 OFF（`OPCG_PLAN_CACHE`/`OPCG_PONDER`/`OPCG_PONDER_SPEC`）＝未設定なら従来挙動と完全同値。
  - journal はスレッドローカル（`tests/test_journal_concurrency.py` がガード）。

旧 app.py から逐語移設。`GAMES`/`CPU_GAMES` は state から取得（app.py と同一 dict を共有）。
"""
import os
import asyncio

from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core import action_api
from opcg_sim.api import decide_client
from ..state import GAMES, CPU_GAMES


def _ponder_enabled() -> bool:
    """Phase 3 ⑥-a 先行計画（pondering）の作動条件。①計画キャッシュ配下のオプトイン（既定 OFF・
    本番体感最適化のみ）。OPCG_PLAN_CACHE=1（①の replay 経路）かつ OPCG_PONDER=1 のとき作動。"""
    return (os.environ.get("OPCG_PLAN_CACHE", "0") == "1"
            and os.environ.get("OPCG_PONDER", "0") == "1")


def _plan_segment(manager, cpu_player, difficulty, mem=None):
    """「このターンの計画手列」を返す（ポンダリング/計画キャッシュの単一の真実源）。

    hard（α-β）を `decide_client`（PyPy ワーカー）でセグメント計画する。「手 dict のリスト（=このターンの
    連続手番）」を返す＝後段のキャッシュ/replay/ポンダリングが共通に乗る。
    """
    return decide_client.plan_segment(manager, cpu_player, difficulty, mem=mem)


async def _ponder_plan(game_id: str) -> None:
    """Phase 3 ⑥-a: 人間の手番処理で制御が CPU へ移った瞬間、CPU セグメントの計画を**前倒し**で計算して
    `meta["plan_cache"]["queue"]` を温める（次の /cpu/step で即時 replay＝CPU 初手の待ちを消す）。

    計算は `plan_segment`（PyPy ワーカー＝別プロセス）へ `asyncio.to_thread` でオフロードし、イベント
    ループを塞がない。①と同じ「decide の決定的結果の前倒し」に留め、合法性ゲート（`_cached_cpu_move`）が
    stale を安全に弾く＝挙動不変。例外時は queue を空にして通常 decide へフォールバックさせる（安全側）。
    """
    manager = GAMES.get(game_id); meta = CPU_GAMES.get(game_id)
    if not manager or not meta:
        return
    cache = meta.setdefault("plan_cache", {})
    try:
        cpu_pid = meta["cpu_player_id"]; difficulty = meta.get("difficulty", "hard")
        turn_mem = meta.setdefault("turn_mem", {})
        # live 盤面をそのまま OS スレッドへ渡すと、スレッド側の deepcopy（plan_turn 内 clone / ワーカーへの
        # pickle）がメインスレッドの盤面変更と競合する（読み取り中の書き換え）。**メインスレッドで原子的に
        # clone**してから渡し、スレッドは隔離されたスナップショットだけに触れる（_kick_speculate と同方針）。
        snap = manager.clone()
        cpu_player = snap.p1 if snap.p1.name == cpu_pid else snap.p2
        actions = await asyncio.to_thread(
            _plan_segment, snap, cpu_player, difficulty, mem=turn_mem)
        cache["queue"] = actions or None
    except Exception:
        cache["queue"] = None
    finally:
        cache["task"] = None


def _kick_ponder(game_id: str) -> None:
    """人間アクション適用後、pending が CPU 手番なら先行計画タスクを起動する（二重起動防止・既定 OFF）。

    旧 queue は前提が変わったので破棄してから焼き直す。イベントループ外（同期テスト等）では `create_task`
    が起動できないため no-op（pondering は本番のみ＝決定性・既存テストへ影響なし）。"""
    if not _ponder_enabled():
        return
    manager = GAMES.get(game_id); meta = CPU_GAMES.get(game_id)
    if not manager or not meta or manager.winner is not None:
        return
    pending = manager.get_pending_request()
    cpu_pid = meta.get("cpu_player_id")
    if not pending or pending.get("player_id") != cpu_pid:
        return
    cache = meta.setdefault("plan_cache", {})
    # ⑥-b: 「人間が今エンドしたら」を投機済み（spec_queue）で、実盤面でも先頭が合法なら昇格＝投機ヒット
    # （CPU 初手の待ちすら消える）。外れ/未完なら下の ⑥-a（実盤面の先行計画）へ。合法性ゲートが採否を担保。
    spec = cache.pop("spec_queue", None)
    if spec:
        cpu_player = manager.p1 if manager.p1.name == cpu_pid else manager.p2
        legal_sigs = {cpu_ai._move_sig(m) for m in manager.get_legal_actions(cpu_player)}
        if cpu_ai._move_sig(spec[0]) in legal_sigs:
            cache["queue"] = spec
            cache["spec_hits"] = cache.get("spec_hits", 0) + 1
            return  # 投機が当たった＝再計画不要
        cache["spec_misses"] = cache.get("spec_misses", 0) + 1
    if cache.get("task") is not None:
        return  # 既に先行計画が走行中
    cache["queue"] = None
    try:
        cache["task"] = asyncio.create_task(_ponder_plan(game_id))
    except RuntimeError:
        cache["task"] = None  # 実行中のイベントループが無い（テスト等）＝起動しない


def _speculate_enabled() -> bool:
    """Phase 3 ⑥-b 投機ポンダリングの作動条件。⑥-a（OPCG_PONDER）配下のさらなるオプトイン
    （OPCG_PONDER_SPEC=1）。既定 OFF＝従来挙動完全同値。当たり率を計測してから本採用を判断する。"""
    return _ponder_enabled() and os.environ.get("OPCG_PONDER_SPEC", "0") == "1"


def _speculate_compute(clone, human_pid, cpu_pid, difficulty):
    """⑥-b 投機の本体（`to_thread` で別スレッド実行）。**クローン上で**人間の TURN_END を仮適用し、
    pending が素直に CPU 手番へ移ったら CPU セグメントを計画して返す（介在する人間決定があれば None）。
    live 盤面には一切触れない（呼び出し側がメインスレッドで原子的に clone 済み）。"""
    human = clone.p1 if clone.p1.name == human_pid else clone.p2
    clone.action_events = []
    action_api.apply_game_action(clone, human, "TURN_END", {})
    pa = clone.pending_actor_action()
    if not pa or pa[0] != cpu_pid:
        return None
    cpu_player = clone.p1 if clone.p1.name == cpu_pid else clone.p2
    return _plan_segment(clone, cpu_player, difficulty, mem={})


async def _speculate_plan(game_id: str, clone, human_pid: str, gen: int) -> None:
    """⑥-b: 「人間が今エンドしたら」の CPU 計画を投機して `spec_queue` に保持（次の TURN_END で昇格判定）。

    計算は `_speculate_compute` を `to_thread` でオフロード（別プロセスのワーカー）。世代 `gen` が
    最新でなければ（人間がさらに動いて盤面が変わった＝supersede）結果は捨てる。使い捨て clone・使い捨て mem
    ＝live 盤面/turn_mem 不変。採否は最終的に `_kick_ponder` の合法性ゲートが担保する。"""
    meta = CPU_GAMES.get(game_id)
    if not meta:
        return
    cache = meta.setdefault("plan_cache", {})
    try:
        cpu_pid = meta["cpu_player_id"]; difficulty = meta.get("difficulty", "hard")
        result = await asyncio.to_thread(
            _speculate_compute, clone, human_pid, cpu_pid, difficulty)
        if cache.get("spec_gen") == gen:        # まだ最新の投機なら採用
            cache["spec_queue"] = result or None
    except Exception:
        if cache.get("spec_gen") == gen:
            cache["spec_queue"] = None
    finally:
        if cache.get("spec_gen") == gen:
            cache["spec_task"] = None


def _kick_speculate(game_id: str) -> None:
    """人間の MAIN 手番（TURN_END が合法）の最中に「今エンドしたら」を投機する（⑥-b・既定 OFF）。

    新しい人間アクションのたびに世代 `spec_gen` を進めて旧投機を supersede（1 ゲーム 1 タスク＝本数ゲート）。
    clone は**メインスレッドで原子的**に取り（読み書き競合なし）、重い計算だけを task へ逃がす。"""
    if not _speculate_enabled():
        return
    manager = GAMES.get(game_id); meta = CPU_GAMES.get(game_id)
    if not manager or not meta or manager.winner is not None:
        return
    pending = manager.get_pending_request()
    cpu_pid = meta.get("cpu_player_id")
    # 人間（=CPU でない側）の MAIN_ACTION 決定点のときだけ投機（TURN_END が合法な静止点）。
    if not pending or pending.get("player_id") == cpu_pid or pending.get("action") != "MAIN_ACTION":
        return
    cache = meta.setdefault("plan_cache", {})
    try:
        clone = manager.clone()  # メインスレッドで原子的に隔離＝以降 task が触れても競合しない
    except Exception:
        return
    gen = cache.get("spec_gen", 0) + 1
    cache["spec_gen"] = gen
    human_pid = manager.p1.name if manager.p1.name != cpu_pid else manager.p2.name
    try:
        cache["spec_task"] = asyncio.create_task(_speculate_plan(game_id, clone, human_pid, gen))
    except RuntimeError:
        cache["spec_task"] = None  # 実行中のイベントループが無い（テスト等）＝起動しない


def _cached_cpu_move(manager, cpu_player, difficulty, meta, turn_mem):
    """Phase 3 ① 計画キャッシュ（本番体感最適化）: 対局ごとの `meta["plan_cache"]` を用い、
    次の計画手が現局面で**合法なら即返す**（探索なし＝即時 replay・ワーカー往復なし）。ミス/前提崩れ
    （相手の介入で前提が変わった等）なら `plan_segment` でセグメントを再計画してキャッシュ。先頭手が
    現局面で不正なら None を返し、呼び出し側が通常 `decide` にフォールバック（**合法性検証で常に安全**）。
    """
    cache = meta.setdefault("plan_cache", {})
    legal = manager.get_legal_actions(cpu_player)
    legal_by_sig = {cpu_ai._move_sig(m): m for m in legal}
    q = cache.get("queue")
    if q:
        sig = cpu_ai._move_sig(q[0])
        if sig in legal_by_sig:
            cache["queue"] = q[1:]
            return legal_by_sig[sig]
        cache["queue"] = None  # 前提崩れ＝破棄して再計画
    actions = _plan_segment(manager, cpu_player, difficulty, mem=turn_mem)
    if actions:
        sig = cpu_ai._move_sig(actions[0])
        if sig in legal_by_sig:
            cache["queue"] = actions[1:]
            return legal_by_sig[sig]
    cache["queue"] = None
    return None
