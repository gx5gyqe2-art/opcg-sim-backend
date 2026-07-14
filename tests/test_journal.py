"""差分巻き戻し（make/unmake）基盤 `journal.py` の検証 — ②最適化 PoC のゲート。

CPU 先読みの clone（deepcopy）が探索コストの ~86% を占める（docs/SPEC.md §2.5.2）。各ノードで
盤面全体を複製する代わりに「適用 → 評価 → 巻き戻し」で 1 手分の変更だけを記録・復元すれば複製を
省ける。本テストはその巻き戻しの**正しさ**を機械的に保証する:

  1. 不活性（transaction 外）では journaled コンテナ・__setattr__ が組み込み型と完全同一に振る舞う
     （= 通常プレイ・既存テストへ無影響）。
  2. transaction 内の任意の状態変更が、退出時に**開始時点へ完全復元**される（属性・list/set/dict）。
  3. 検証器 `deep_diff` が、journaled 化の取りこぼし（未カバーの in-place 変更）を確実に検出する。
  4. **実プレイの全手**を transaction で包んで適用→巻き戻しし、開始 deepcopy と完全一致することを
     確認する（= 実効果経路での完全性ゲート。横展開はこの照合がカバレッジを示す）。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_journal.py -q -s -p no:cacheprovider
"""
import copy
import random

import conftest  # noqa: F401
import pytest

from opcg_sim.src.core import journal
from opcg_sim.src.core.journal import (
    JournaledList, JournaledSet, JournaledDict, transaction, deep_diff,
)
import cpu_arena
import test_cpu_puzzles as P
from opcg_sim.src.core import action_api


@pytest.fixture(scope="module")
def db():
    return cpu_arena._load_db()


# ---------------------------------------------------------------------------
# 1. 不活性時は組み込み型と完全同一（記録しない・挙動を変えない）
# ---------------------------------------------------------------------------

def test_inert_containers_behave_as_builtins():
    assert journal._active is None
    lst = JournaledList([1, 2, 3])
    lst.append(4); lst.pop(0); lst[0] = 9
    assert lst == [9, 3, 4] and isinstance(lst, list)
    s = JournaledSet({1, 2})
    s.add(3); s.discard(1)
    assert s == {2, 3} and isinstance(s, set)
    d = JournaledDict({"a": 1})
    d["b"] = 2; d.pop("a")
    assert d == {"b": 2} and isinstance(d, dict)


def test_inert_records_nothing():
    """transaction 外の変更は記録されない（_active が None のまま）。"""
    lst = JournaledList([1])
    lst.append(2)
    assert journal._active is None


# ---------------------------------------------------------------------------
# 2. 基本的な巻き戻し（属性・コンテナ）
# ---------------------------------------------------------------------------

class _Obj:
    def __setattr__(self, name, value):
        if journal._active is not None:
            journal.record_attr(self, name, self.__dict__)
        object.__setattr__(self, name, value)


def test_rollback_attr_set_and_create():
    o = _Obj()
    o.x = 1
    before = copy.deepcopy(o.__dict__)
    with transaction():
        o.x = 99          # 既存属性の書き換え
        o.y = "new"       # 新規属性の作成
        assert o.x == 99 and o.y == "new"
    assert o.x == 1 and not hasattr(o, "y")
    assert o.__dict__ == before


def test_rollback_containers():
    lst = JournaledList([1, 2, 3])
    s = JournaledSet({1, 2})
    d = JournaledDict({"a": 1})
    with transaction():
        lst.append(4); lst.pop(0)
        s.add(3); s.discard(1)
        d["b"] = 2; del d["a"]
        assert lst == [2, 3, 4] and s == {2, 3} and d == {"b": 2}
    assert lst == [1, 2, 3] and s == {1, 2} and d == {"a": 1}


def test_nested_transactions_rollback_independently():
    lst = JournaledList([0])
    with transaction():
        lst.append(1)
        with transaction():
            lst.append(2)
            assert lst == [0, 1, 2]
        assert lst == [0, 1]       # 内側だけ巻き戻る
        lst.append(3)
        assert lst == [0, 1, 3]
    assert lst == [0]              # 外側も全て巻き戻る


# ---------------------------------------------------------------------------
# 3. 検証器が取りこぼしを検出する（負のテスト）
# ---------------------------------------------------------------------------

def test_deep_diff_detects_difference():
    a = JournaledList([1, 2, 3])
    b = JournaledList([1, 2, 3])
    assert deep_diff(a, b) is None
    b.append(4)
    assert deep_diff(a, b) is not None


def test_deep_diff_catches_unjournaled_mutation():
    """journaled でない**既存**コンテナを in-place 変更すると巻き戻せず、deep_diff が捕える。"""
    o = _Obj()
    o.plain = [1, 2, 3]           # わざとプレーン list（journaled でない既存コンテナ）
    before = copy.deepcopy(o)
    with transaction():
        o.plain.append(4)         # in-place 変更 → 記録されない
    # 巻き戻し後も 4 が残る → 検証器が検出
    assert o.plain == [1, 2, 3, 4]
    assert deep_diff(before, o) is not None


# ---------------------------------------------------------------------------
# 4. 実プレイ全手の適用→巻き戻し完全性ゲート
# ---------------------------------------------------------------------------

def _pid_key():
    props = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {})
    return props.get('PLAYER_ID', 'player_id')


def _apply(manager, actor, move):
    if move["kind"] == "battle":
        action_api.apply_battle_action(manager, actor, move["action_type"], move.get("card_uuid"))
    else:
        action_api.apply_game_action(manager, actor, move["action_type"], move.get("payload", {}))


@pytest.mark.cpu_infra
def test_real_playout_make_unmake_roundtrip(db):
    """実ゲームを進めつつ、各手を transaction で包んで適用→巻き戻しし、開始 deepcopy と完全一致を確認。

    手順: 各 pending で (a) deepcopy 退避 → (b) transaction 内で実手を適用し「何か変わった」ことを
    確認 → (c) 退出で巻き戻り deep_diff==None を確認 → (d) 同じ手を**素で**適用してゲームを前進。
    実効果（登場時・KO・ドン付与・戦闘・選択の既定解決等）を一通り通す。
    """
    random.seed(7)
    l1, c1 = cpu_arena.build_deck(db, "p1")
    l2, c2 = cpu_arena.build_deck(db, "p2")
    from opcg_sim.src.core.gamestate import GameManager, Player
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    deciders = {
        "p1": cpu_arena._make_decider("hard"),
        "p2": cpu_arena._make_decider("hard"),
    }
    KEY = _pid_key()
    checked = 0
    steps = 0
    while gm.winner is None and steps < 120:
        pending = gm.get_pending_request()
        if not pending:
            break
        pid = pending[KEY]
        actor = gm.p1 if gm.p1.name == pid else gm.p2
        move = deciders[pid](gm, actor)
        if move is None:
            break

        # parked resolver 状態（中断再開）も journaled 化済み＝中断/非中断どちらの手も round-trip 照合する
        # （resolver の journaled __setattr__／context・saved_targets の JournaledDict／execution_stack・
        # saved_stack・退避スタックの JournaledList／誘発 item の JournaledDict 化。§2.5.2）。
        if True:
            # (a) 退避（continuous の後方参照ごと deepcopy）。action_events は手ごとに
            #     リセットされる transient バッファなので、照合差を避けるため退避前に揃える。
            gm.action_events = JournaledList()
            before = copy.deepcopy(gm)
            # (b) transaction 内で適用
            with transaction():
                try:
                    _apply(gm, actor, move)
                except Exception as e:
                    pytest.fail(f"step{steps} apply raised under journal: {type(e).__name__}: {e}")
                changed = deep_diff(before, gm)
                # 実手なので通常は何か変わる（PASS 等の no-op もあり得るので必須にはしない）
            # (c) 巻き戻し後は完全一致
            diff = deep_diff(before, gm)
            assert diff is None, f"step{steps} ({move['action_type']}): rollback mismatch at {diff}"
            if changed is not None:
                checked += 1

        # (d) 素で適用して前進
        gm.action_events = JournaledList()
        _apply(gm, actor, move)
        steps += 1

    assert steps >= 20, f"ゲームが十分進まなかった (steps={steps})"
    assert checked >= 10, f"状態変化を伴う手の検証数が不足 (checked={checked})"


@pytest.mark.slow
def test_parked_resume_make_unmake_roundtrip(db):
    """**中断再開（parked resolver）の手**を transaction で包んで適用→巻き戻しし、開始 deepcopy と
    完全一致（deep_diff==None）を確認する。parked 状態の journaled 化（resolver の journaled
    __setattr__／context・saved_targets の JournaledDict／execution_stack・saved_stack・退避スタックの
    JournaledList／誘発 item の JournaledDict）の完全性ゲート。これが緑＝`_mu_safe` が中断局面でも
    make/unmake できる（残 clone フォールバックの大半を解消・docs/SPEC.md §2.5.2）。

    NOTE: 8 seed × 全手の make/unmake 照合で**約245秒**＝スイート最重量。CI（`-m "not slow"`）からは
    除外し、make/unmake 周辺を触ったときに**手動実行**する前提:
        OPCG_LOG_SILENT=1 python -m pytest tests/test_journal.py -q -s -m slow -p no:cacheprovider
    """
    from opcg_sim.src.core.gamestate import GameManager, Player
    KEY = _pid_key()
    parked_checked = 0
    for seed in range(0, 8):
        random.seed(seed)
        l1, c1 = cpu_arena.build_deck(db, "p1")
        l2, c2 = cpu_arena.build_deck(db, "p2")
        gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
        gm.start_game()
        deciders = {"p1": cpu_arena._make_decider("hard"), "p2": cpu_arena._make_decider("hard")}
        steps = 0
        while gm.winner is None and steps < 150:
            pending = gm.get_pending_request()
            if not pending:
                break
            pid = pending[KEY]
            actor = gm.p1 if gm.p1.name == pid else gm.p2
            move = deciders[pid](gm, actor)
            if move is None:
                break
            if gm.active_interaction is not None:  # 中断再開の手だけ照合
                gm.action_events = JournaledList()
                before = copy.deepcopy(gm)
                with transaction():
                    try:
                        _apply(gm, actor, move)
                    except Exception as e:
                        pytest.fail(f"seed{seed} step{steps} parked apply raised: {type(e).__name__}: {e}")
                diff = deep_diff(before, gm)
                assert diff is None, (
                    f"seed{seed} step{steps} ({move['action_type']}): parked rollback mismatch at {diff}")
                parked_checked += 1
            gm.action_events = JournaledList()
            _apply(gm, actor, move)
            steps += 1
    assert parked_checked >= 20, f"中断再開の検証数が不足 (parked_checked={parked_checked})"
