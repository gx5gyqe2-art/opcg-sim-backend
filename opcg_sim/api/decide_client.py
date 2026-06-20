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
           profile=None, plan=None, trace: Optional[Dict[str, Any]] = None,
           trace_read_ahead: bool = False):
    """本番の decide。ワーカー有効時は PyPy へ委譲、失敗時はインプロセスへフォールバック。"""
    if mem is None:
        mem = {}
    if USE_WORKER:
        try:
            req = (manager, player.name, difficulty, mem, profile, plan,
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
        manager, player, difficulty, mem=mem, profile=profile, plan=plan,
        trace=trace, trace_read_ahead=trace_read_ahead,
    )
