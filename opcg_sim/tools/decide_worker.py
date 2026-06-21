"""PyPy 常駐ワーカー: CPU の探索（decide）だけを別ランタイム（PyPy）で実行する。

方式B（プロセス分離）の PyPy 側。CPython の FastAPI プロセスから Unix ドメインソケット経由で
**pickle 化した盤面＋難易度＋mem/profile/plan＋RNG 状態**を受け取り、`cpu_ai.decide_guarded` を
実行して **(move, trace, mem)** を返す。

- **依存は stdlib のみ**（`opcg_sim/src/**` も stdlib-only）＝PyPy で確実に動く。配信スタック
  （fastapi/pydantic/grpcio）は一切読まない（Phase0 で PyPy 非互換と判明したものを回避）。
- 常駐（長寿命）＝JIT がフルウォーム＝中盤 decide が CPython 比 ~2.1x（docs/reports の実測）。
- **方策・評価・カード挙動は一切変えない**（cpu_ai をそのまま呼ぶだけの実行オフロード）。

プロトコル（1 リクエスト = 1 接続）:
  client → worker : pickle((manager, cpu_pid, difficulty, mem, profile, plan, rng_state, want_trace, read_ahead))
  worker → client : pickle(("ok", (move, trace, mem)))  /  pickle(("err", repr))
いずれも 4byte ビッグエンディアン長 + pickle 本体のフレーム。
"""
import os
import sys
import socket
import struct
import pickle
import random

# opcg_sim をパッケージとして解決（このファイル: opcg_sim/tools/decide_worker.py）。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from opcg_sim.src.core import cpu_ai  # noqa: E402
from opcg_sim.src.utils.loader import CardLoader  # noqa: E402

SOCK_PATH = os.environ.get("OPCG_WORKER_SOCK", "/tmp/opcg_decide.sock")


def _readn(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed mid-frame")
        buf.extend(chunk)
    return bytes(buf)


def _recv(conn):
    (n,) = struct.unpack(">I", _readn(conn, 4))
    return pickle.loads(_readn(conn, n))


def _send(conn, obj):
    body = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack(">I", len(body)) + body)


def _handle(req):
    manager, cpu_pid, difficulty, mem, profile, plan, rng_state, want_trace, read_ahead = req[:9]
    mode = req[9] if len(req) > 9 else "decide"  # 後方互換（10要素目=mode・既定 decide）
    # player は名前で解決（別 pickle すると盤面内カードと同一性が切れるため）。
    player = manager.p1 if manager.p1.name == cpu_pid else manager.p2
    # 決定性: CPython 側の RNG 状態を再現してタイブレークを一致させる。
    rng = random.Random()
    rng.setstate(rng_state)
    trace = {} if want_trace else None
    if mode == "plan":
        # Phase 3 ① 計画キャッシュ: 相手介入/TURN_END までの自分の連続手番を計画して action list を返す。
        actions = cpu_ai.plan_turn(manager, cpu_pid, difficulty, rng=rng, mem=mem, plan=plan)
        return (actions, trace, mem)
    move = cpu_ai.decide_guarded(
        manager, player, difficulty, rng=rng, mem=mem,
        plan=plan, trace=trace, trace_read_ahead=read_ahead,
    )
    return (move, trace, mem)


def main():
    # 焼き込み済みカードキャッシュをロード（コールドスタートの全件パースを回避）。
    try:
        CardLoader().load_cache()
    except Exception:
        pass  # キャッシュ未生成でも本処理（pickle 受領盤面）は master 同梱で動く。

    if os.path.exists(SOCK_PATH):
        os.remove(SOCK_PATH)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK_PATH)
    srv.listen(16)
    sys.stderr.write("[decide_worker] ready on %s (%s)\n" % (SOCK_PATH, sys.version.split()[0]))
    sys.stderr.flush()

    while True:
        try:
            conn, _ = srv.accept()
        except KeyboardInterrupt:
            break
        try:
            req = _recv(conn)
            _send(conn, ("ok", _handle(req)))
        except Exception as e:  # 1 リクエストの失敗でワーカーは落とさない（client がフォールバック）。
            try:
                _send(conn, ("err", repr(e)))
            except Exception:
                pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()
