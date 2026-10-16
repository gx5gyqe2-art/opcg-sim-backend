"""効果構造 JSON（`opcg_effects.json`）の自動生成が**並行しても壊れない**ことの契約。

2026-09-13 の実害（r10 のアリーナ・`docs/n_loop_ops.md` §6）: アリーナのワーカープール 4 本が
起動時に同時に生成物を作りに行き、全員が同じ `out + ".tmp"` を開いて互いの中身を上書きした
ため、壊れた JSON が `os.replace` で公開され、ほぼ全ワーカーが最初のシャードで落ちた。
**これはゲームプレイの経路（生成・アリーナ・serve の起動）が落ちる欠陥**なので必須テスト
（`cpu_infra` は付けない）。エクスポータ本体は重いので `runner` を差し替えて機構だけを試す。

守る性質:
  1. 一時ファイル名はプロセスごとに別（`tmp_path` に pid が入る）＝同時に書いても混ざらない。
  2. `ensure` は同時に来ても**1 本だけ**が作り、他は出来上がるのを待って "waited" を返す。
  3. 既にあれば何もしない（"exists"）。
  4. 作りかけで死んだプロセスの鍵（古い lock）は奪って自分で作る。
"""
import json
import os
import threading
import time

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.tools import export_effects_json as EEJ


def test_tmp_path_is_per_process():
    got = EEJ.tmp_path("/tmp/opcg_effects.json")
    assert got.startswith("/tmp/opcg_effects.json.tmp.")
    assert got.endswith(str(os.getpid()))


def test_ensure_exists_is_a_noop(tmp_path):
    out = str(tmp_path / "e.json")
    with open(out, "w") as fh:
        json.dump({"version": 1}, fh)
    calls = []
    assert EEJ.ensure(out, runner=lambda o: calls.append(o)) == "exists"
    assert calls == []


def test_ensure_runs_once_under_concurrency(tmp_path):
    """4 本が同時に来ても作るのは 1 本・残りは待って "waited"。"""
    out = str(tmp_path / "e.json")
    calls = []
    started = threading.Event()

    def runner(o):
        calls.append(o)
        started.set()
        time.sleep(0.6)                       # 作っている間に他の 3 本を待たせる
        with open(EEJ.tmp_path(o), "w") as fh:
            json.dump({"version": 1}, fh)
        os.replace(EEJ.tmp_path(o), o)

    results = {}

    def go(i):
        try:
            results[i] = EEJ.ensure(out, timeout=30.0, poll=0.05, runner=runner)
        except Exception as e:                 # 失敗も記録して assert で出す
            results[i] = f"error: {e!r}"

    threads = [threading.Thread(target=go, args=(i,)) for i in range(4)]
    threads[0].start()
    started.wait(5.0)                         # 1 本目が鍵を取ってから残りを起こす
    for t in threads[1:]:
        t.start()
    for t in threads:
        t.join(30.0)
    assert len(calls) == 1, f"エクスポータが {len(calls)} 回走った（1 回であるべき）: {results}"
    assert sorted(results.values()) == ["created", "waited", "waited", "waited"], results
    with open(out) as fh:
        assert json.load(fh) == {"version": 1}      # 公開されたのは完全なファイル
    assert not os.path.exists(out + EEJ.LOCK_SUFFIX)


def test_ensure_steals_a_stale_lock(tmp_path):
    """作りかけで死んだプロセスの鍵は（timeout より古ければ）奪って自分で作る。"""
    out = str(tmp_path / "e.json")
    lock = out + EEJ.LOCK_SUFFIX
    with open(lock, "w") as fh:
        fh.write("99999")
    os.utime(lock, (time.time() - 3600, time.time() - 3600))
    calls = []

    def runner(o):
        calls.append(o)
        with open(o, "w") as fh:
            json.dump({"version": 1}, fh)

    assert EEJ.ensure(out, timeout=1.0, poll=0.05, runner=runner) == "created"
    assert calls == [out]
    assert not os.path.exists(lock)
