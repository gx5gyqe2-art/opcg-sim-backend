"""CPython(FastAPI) → PyPy ワーカーへ decide を委譲する薄いブリッジ。

方式B（プロセス分離）の CPython 側。`OPCG_PYPY_WORKER=1` のときだけ PyPy ワーカーへ盤面を pickle 送信し、
それ以外・送受信失敗時は **従来どおりインプロセスで `cpu_ai.decide_guarded` を実行**（＝現行挙動・ロールバック経路）。

- 方策・評価・カード挙動は不変（実行場所を移すだけ）。`tr`/`mem` はワーカーから返る値で更新する。
- ワーカー未起動・IPC 失敗は握り潰してインプロセス実行へフォールバック（可用性優先）。
"""
import os
import socket
import struct
import pickle
import random
import subprocess
import sys
import threading
from typing import Any, Dict, Optional

from opcg_sim.src.core import cpu_ai

USE_WORKER = os.environ.get("OPCG_PYPY_WORKER", "0") == "1"
SOCK_PATH = os.environ.get("OPCG_WORKER_SOCK", "/tmp/opcg_decide.sock")
PYPY_BIN = os.environ.get("OPCG_PYPY_BIN", "pypy3")
CONNECT_TIMEOUT = float(os.environ.get("OPCG_WORKER_TIMEOUT", "30"))

_spawn_lock = threading.Lock()
_proc: Optional[subprocess.Popen] = None


def spawn_worker() -> None:
    """PyPy ワーカーを subprocess で常駐起動（既に生きていれば何もしない）。冪等。"""
    global _proc
    if not USE_WORKER:
        return
    with _spawn_lock:
        if _proc is not None and _proc.poll() is None:
            return
        env = dict(os.environ)
        _proc = subprocess.Popen(
            [PYPY_BIN, "-m", "opcg_sim.tools.decide_worker"],
            env=env, stdout=sys.stderr, stderr=sys.stderr,
        )


def _readn(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed mid-frame")
        buf.extend(chunk)
    return bytes(buf)


def _roundtrip(req):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(CONNECT_TIMEOUT)
    s.connect(SOCK_PATH)
    try:
        body = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
        s.sendall(struct.pack(">I", len(body)) + body)
        (n,) = struct.unpack(">I", _readn(s, 4))
        tag, payload = pickle.loads(_readn(s, n))
        if tag != "ok":
            raise RuntimeError("worker error: %s" % payload)
        return payload
    finally:
        s.close()


def decide(manager, player, difficulty: str = "normal", *, mem: Optional[Dict[str, Any]] = None,
           trace: Optional[Dict[str, Any]] = None,
           trace_read_ahead: bool = False):
    """本番の decide。ワーカー有効時は PyPy へ委譲、失敗時はインプロセスへフォールバック。

    difficulty=="learned"（本番既定）は学習型CPU（Gen2 value+policy+MCTS）へインプロセスで分岐する
    （docs/reports/cpu_rl_pilot_p3_results_20260630.md）。モデル未同梱環境での hard フォールバックは
    **ゲーム生成時**（routers `_learned_available()` ゲート）で確定済みなので、ここへ到達する learned は
    **learned-only（per-move の L1 フォールバック無し）**＝観測する手は必ず学習型。
    """
    if mem is None:
        mem = {}
    if difficulty == "learned":
        # learned-only（L1 フォールバック無し）＝観測する手は必ず学習型。本番は numpy 必須。
        # cpu_trace 時は trace に手の分析（regret/候補/L1第二意見）を書き込む。
        from opcg_sim.src.core import cpu_learned
        return cpu_learned.decide_learned(manager, player, trace=trace)
    if USE_WORKER:
        try:
            req = (manager, player.name, difficulty, mem,
                   random.getstate(), trace is not None, trace_read_ahead)
            move, tr, mem2 = _roundtrip(req)
            # mem（turn_mem）はワーカー側で変異するので CPython 側へ完全反映。
            mem.clear()
            mem.update(mem2)
            if trace is not None and tr:
                trace.update(tr)
            return move
        except Exception:
            # ワーカー未起動／IPC 失敗 → 再起動を試みつつ今回はインプロセスで応答。
            try:
                spawn_worker()
            except Exception:
                pass
    return cpu_ai.decide_guarded(
        manager, player, difficulty, mem=mem,
        trace=trace, trace_read_ahead=trace_read_ahead,
    )


def plan_segment(manager, player, difficulty: str = "normal", *, mem: Optional[Dict[str, Any]] = None):
    """Phase 3 ① 計画キャッシュ: セグメント（相手介入/TURN_END まで）の自分の連続手番を計画して
    action list を返す（ワーカー優先・失敗時インプロセス）。`mem` はワーカー側の進行を反映する。

    difficulty=="learned" は先読み計画をせず、学習型CPUの1手だけを返す（毎手 MCTS で決める）。
    learned-only（L1 フォールバック無し）。"""
    if mem is None:
        mem = {}
    if difficulty == "learned":
        from opcg_sim.src.core import cpu_learned
        mv = cpu_learned.decide_learned(manager, player)
        return [mv] if mv else []
    if USE_WORKER:
        try:
            req = (manager, player.name, difficulty, mem,
                   random.getstate(), False, False, "plan")
            actions, _tr, mem2 = _roundtrip(req)
            mem.clear()
            mem.update(mem2)
            return actions
        except Exception:
            try:
                spawn_worker()
            except Exception:
                pass
    return cpu_ai.plan_turn(manager, player.name, difficulty, mem=mem)
