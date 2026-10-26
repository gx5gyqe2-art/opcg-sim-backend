"""CPU 対戦の思考を裏のスレッドで回す（`/api/game/cpu/step` の非同期化・2026-10-26）。

1 コアの機械では 1 手の思考が数秒〜十数秒になりうる（世界 4 本・1 スレッドで最悪 17 秒・
`docs/reports/2026-10-25_world_parallel.md`）。思考をリクエストの中で回すとその間サーバ全体が
止まり、フロントは「考えているのか固まったのか」を知る手段が無い。そこで:

- 思考は **対局の複製**（`RsGame.decide` が Rust の `Game.fork()` の上で読む）を裏のスレッドで回す。
  Rust は木を回す間 GIL を放すので、その間も API は他のリクエスト（状況の問い合わせ）に答えられる。
- `cpu/step` は思考を起こして最大 `wait_sec` 秒だけ待つ。間に合えばその手を適用して返し（今までと同じ）、
  間に合わなければ「思考中・経過 n 秒」を返す。フロントは 1 秒ごとに `cpu/step` を呼び直す。
- 思考は `(ターン, 要求の request_id)` に紐づける。局面が変わっていたら（起こした後に別の経路で
  進んだら）その結果は捨てて読み直す。

レジストリはプロセス内 dict（`state.py` と同じく Cloud Run 単一インスタンス前提）。
"""
import asyncio
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional, Tuple

# 思考のスレッド。1 局に 1 本しか走らない（同じ対局の思考は重ねない）ので、同時に走るのは
# 同時に CPU の番が来ている対局の数まで。
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cpu-think")
_LOCK = threading.Lock()


class Think:
    """1 局の 1 回の思考。`future` の結果は `(move, trace)`。"""

    def __init__(self, key: Tuple[Any, ...], future: Future):
        self.key = key
        self.future = future
        self.started = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.started


# {game_id: Think}
THINKS: Dict[str, Think] = {}


def start_or_get(game_id: str, key: Tuple[Any, ...],
                 run: Callable[[], Tuple[Optional[Dict[str, Any]], Dict[str, Any]]]) -> Think:
    """この局面（`key`）の思考を返す。無い・別の局面のものなら新しく起こす。

    別の局面の思考がまだ走っていても止めはしない（Rust の探索は途中で止められない）。
    結果は捨てられ、走り終えればスレッドは空く。
    """
    with _LOCK:
        th = THINKS.get(game_id)
        if th is None or th.key != key:
            th = Think(key, _EXECUTOR.submit(run))
            THINKS[game_id] = th
        return th


async def wait(th: Think, wait_sec: float) -> bool:
    """思考が終わるまで最大 `wait_sec` 秒待つ（イベントループは塞がない）。終わっていれば True。"""
    if th.future.done():
        return True
    if wait_sec <= 0:
        return False
    try:
        await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(th.future)), timeout=wait_sec)
    except asyncio.TimeoutError:
        return False
    except Exception:
        # 思考の例外は `future.result()` を読む側で扱う。
        return True
    return True


def pop(game_id: str, th: Think) -> None:
    """読み終えた思考をレジストリから外す（同じ思考のときだけ）。"""
    with _LOCK:
        if THINKS.get(game_id) is th:
            del THINKS[game_id]


def drop(game_id: str) -> None:
    """対局の思考を忘れる（対局の削除・テストの後始末）。"""
    with _LOCK:
        THINKS.pop(game_id, None)


def clear_all() -> None:
    with _LOCK:
        THINKS.clear()
