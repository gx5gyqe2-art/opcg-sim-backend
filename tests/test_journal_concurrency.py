"""journal のスレッド安全性（並行回帰）。

背景: 本番でポンダリングが `asyncio.to_thread`（本物の OS スレッド）でも探索を走らせ得る。差分巻き戻し
journal の状態（active/mut_count）が**プロセス共有のグローバル**だと、あるスレッドが `transaction()` を
開いている間に別スレッドが live オブジェクトの属性を初回セットすると、その set が**別スレッドの journal に
誤記録**され、rollback で live から属性が pop される（実症状: `CardInstance has no attribute 'master'`）。

journal をスレッドローカルにしたことで、各スレッドの記録は互いに漏れない。本テストはスレッド 2 本を
**Event で決定論的に同期**させて、グローバル実装なら必ず壊れるシナリオを再現し、現実装で無傷であることを固定する
（既存テストは単一スレッドのためこの競合を検知できない＝本ファイルが唯一のガード）。
"""
import threading

import conftest  # noqa: F401
import pytest

from opcg_sim.src.core import journal
from opcg_sim.src.models.models import CardInstance
from cpu_selfplay import _load_db, build_deck


@pytest.fixture(scope="module")
def master():
    db = _load_db()
    _, cards = build_deck(db, "p1")
    return cards[0].master


def test_journal_active_is_thread_local(master):
    """あるスレッドの open な transaction は、他スレッドからは見えない（active はスレッドローカル）。"""
    a_holding = threading.Event()
    release = threading.Event()
    seen = {}

    def thread_a():
        with journal.transaction():
            a_holding.set()
            release.wait(5)
        # rollback 済み

    ta = threading.Thread(target=thread_a)
    ta.start()
    try:
        assert a_holding.wait(5), "thread A が transaction を開けなかった"
        # A が transaction を保持している今、メインスレッドからは active が見えないこと。
        seen["main_active"] = journal.is_active()
    finally:
        release.set()
        ta.join(5)
    assert seen["main_active"] is False, "他スレッドの transaction が見えた（journal がスレッドローカルでない）"
    assert journal.is_active() is False


def test_concurrent_transaction_does_not_pop_live_attr(master):
    """並行回帰（本丸）: スレッド A が transaction を保持している間にメインスレッドが live な CardInstance を
    生成（`master` を初回セット）しても、A の rollback で `master` が消えない。

    グローバル journal 実装では、メインの `__setattr__` が A の active を見て `master` を A の journal へ
    `_POP` 記録 → A の rollback が live カードから `master` を pop → AttributeError（再現する症状）。
    スレッドローカル実装では、メインの active は None なので記録されず無傷。"""
    a_holding = threading.Event()
    main_done = threading.Event()
    box = {}

    def thread_a():
        with journal.transaction():
            a_holding.set()
            main_done.wait(5)   # メインが live カードを作り終えるまで transaction を保持
        # ここで A の rollback。グローバル journal だとこの瞬間に live カードの master が pop される。

    ta = threading.Thread(target=thread_a)
    ta.start()
    try:
        assert a_holding.wait(5), "thread A が transaction を開けなかった"
        # A が active を保持している今、メイン（このスレッド）で live カードを生成。
        card = CardInstance(master, "p1")
        box["card"] = card
        box["master_at_create"] = getattr(card, "master", "MISSING")
    finally:
        main_done.set()
        ta.join(5)
    card = box["card"]
    assert box["master_at_create"] is master
    # A の rollback 後も無傷であること（並行バグなら master が pop されて MISSING になる）。
    assert getattr(card, "master", "MISSING") is master, \
        "並行 transaction の rollback が live カードから master を pop した（journal スレッド安全性の回帰）"
    # 念のため属性アクセスが例外を出さない（症状の直接確認）。
    assert card.master is master


def test_parallel_searches_do_not_corrupt_each_other(master):
    """2 スレッドが各々 transaction 内で別々の live カードの属性を書き換え→巻き戻しても、相手のカードに
    影響しない（記録がスレッド間で漏れない）。各スレッドは自分のカードだけを開始値へ復元する。"""
    results = {}
    start = threading.Event()

    def worker(tag):
        c = CardInstance(master, tag)
        c.power_buff = 0
        start.wait(5)
        ok = True
        for i in range(2000):
            with journal.transaction():
                c.power_buff = i + 1          # 書き換え（journal が旧値 0 を記録）
                if c.power_buff != i + 1:
                    ok = False
            # transaction 退出で巻き戻し → 0 に戻るはず
            if c.power_buff != 0:
                ok = False
        results[tag] = ok

    threads = [threading.Thread(target=worker, args=(t,)) for t in ("p1", "p2")]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join(15)
    assert results.get("p1") and results.get("p2"), \
        f"並行 make/unmake が干渉した: {results}"
