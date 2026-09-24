"""守る側の補助教師（dump v5 の `aux_def`／`aux_def_row`・P8・`2026-09-24_p8_defender_columns.md`）の値を固定する。

標準テスト（マーカー無し）: `opcg_sim/loop/record_gen.py` の `aux_def_from_ledger`（`_fill_def` が
本体）が「棋譜の台帳 → 次の 1 ターンで自分側に起きたこと」をどう数えるかの契約。エンジンは
要らない（**手作りの台帳**を渡す）。`aux_from_ledger`（`aux`／`aux_tok`・相手側の活動）とは
**独立の関数**——窓の探し方（A＝この行の後、相手の手番が 1 回終わるまで）は同じ規約だが、
互いを呼ばない（`test_aux_targets.py` の既存契約に触れない）。

守る性質:
  1. `aux_def` 6 枠 ×[狙われた回数, ブロックした回数, 失ったライフ（リーダー枠だけ）,
     場を離れた先]（枠の並びは自分 L ＋自分場 5＝`aux_tok` の裏返し・空枠は 0）。
  2. `left_dest` の符号（0=残った／1=戦闘でトラッシュ／2=効果でトラッシュ／3=効果で手札／
     4=効果でデッキ）——窓の終わりのゾーン差分で判定し、戦闘か効果かは「窓の中で狙われたか」
     で近似する。
  3. `aux_def_row` 2 列＝そのターン全体で自分が打った `SELECT_COUNTER`／`SELECT_BLOCKER` の回数。
  4. `step_record` の新しい `target` 欄（`ATTACK`／`ATTACK_CONFIRM` の対象 uuid）。
"""
import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.loop import record_gen as G

pytestmark = pytest.mark.cpu_infra


def _snap(turn, tp, life, field, hand=None, trash=None, life_ids=None, leader=("L1", "L2")):
    """台帳の 1 枚（`test_aux_targets._snap` に `hand`／`trash`／`life_ids` を足したもの）。"""
    hand = hand or {}
    trash = trash or {}
    life_ids = life_ids or {}
    out = {"turn": turn, "tp": tp, "life": dict(life), "power": {}, "n": {},
           "field": {}, "leader": {}, "names": {}, "hand": {}, "trash": {}, "life_ids": {}}
    for i, nm in enumerate(("p1", "p2")):
        f = field[nm]
        out["power"][nm] = float(sum(p for _u, p, _n in f))
        out["n"][nm] = float(len(f))
        out["field"][nm] = [u for u, _p, _n in f]
        out["leader"][nm] = leader[i]
        out["names"][nm] = {u: n for u, _p, n in f}
        out["names"][nm][leader[i]] = f"leader{i + 1}"
        out["hand"][nm] = set(hand.get(nm, ()))
        out["trash"][nm] = set(trash.get(nm, ()))
        out["life_ids"][nm] = set(life_ids.get(nm, ()))
    return out


def _step(actor, at, uuid=None, target=None):
    payload = {}
    if uuid:
        payload["uuid"] = uuid
    if target:
        payload["target_ids"] = [target]
    return G.step_record(actor, {"action_type": at, "payload": payload}, [])


def _def(snaps, steps, steps_idx, who):
    return G.aux_def_from_ledger(snaps, steps, steps_idx, who)


# --- 1. step_record が対象 uuid を積む -------------------------------------
def test_step_record_captures_the_attack_target():
    r = _step("p2", "ATTACK", uuid="A2", target="A1")
    assert r["at"] == "ATTACK" and r["uuid"] == "A2" and r["target"] == "A1"
    # 攻撃以外の手は target_ids が payload に在っても読まない
    r2 = G.step_record("p1", {"action_type": "PLAY", "payload": {"uuid": "H1", "target_ids": ["X"]}}, [])
    assert r2["target"] is None
    # target_ids の無い攻撃（見つからない場合の防御）は None
    assert _step("p2", "ATTACK", uuid="A2")["target"] is None


# --- 2. 狙われた・ブロックした・失ったライフ（1 つの窓で 3 つとも読む）-----
#: p1 の場は A1（5000）・C1（6000）。p2 が 3 回攻撃——L1 と A1（無傷）・C1（無防備で落ちる）。
F0 = {"p1": [("A1", 5000, "赤A"), ("C1", 6000, "赤C")], "p2": [("A2", 3000, "青B"), ("B2", 3000, "青C"),
                                                              ("D2", 3000, "青D")]}
F1 = {"p1": [("A1", 5000, "赤A")], "p2": F0["p2"]}          # 窓の終わり：C1 が落ちた


@pytest.fixture(scope="module")
def ledger():
    snaps = [
        _snap(1, "p1", {"p1": 5, "p2": 5}, F0),                              # 0: p1 PLAY
        _snap(1, "p1", {"p1": 5, "p2": 5}, F0),                              # 1: p1 TURN_END
        _snap(2, "p2", {"p1": 5, "p2": 5}, F0),                              # 2: p2 ATTACK→L1
        _snap(2, "p2", {"p1": 4, "p2": 5}, F0),                              # 3: p2 ATTACK→A1（L1 の分が通った）
        _snap(2, "p2", {"p1": 4, "p2": 5}, F0),                              # 4: p1 SELECT_BLOCKER（C1 が A1 を庇う）
        _snap(2, "p2", {"p1": 4, "p2": 5}, F0),                              # 5: p2 ATTACK→C1（別の攻撃・無防備）
        _snap(2, "p2", {"p1": 4, "p2": 5}, F1, trash={"p1": {"C1"}}),        # 6: p2 TURN_END（C1 が落ちた後）
        _snap(3, "p1", {"p1": 4, "p2": 5}, F1, trash={"p1": {"C1"}}),        # 7: 台帳の終わり
    ]
    steps = [
        _step("p1", "PLAY", uuid="H1"),
        _step("p1", "TURN_END"),
        _step("p2", "ATTACK", uuid="A2", target="L1"),
        _step("p2", "ATTACK", uuid="B2", target="A1"),
        _step("p1", "SELECT_BLOCKER", uuid="C1"),
        _step("p2", "ATTACK", uuid="D2", target="C1"),
        _step("p2", "TURN_END"),
    ]
    assert len(snaps) == len(steps) + 1
    return snaps, steps


def test_targeted_blocked_and_life_lost(ledger):
    snaps, steps = ledger
    aux_def, aux_def_row = _def(snaps, steps, [0], ["p1"])
    assert aux_def.shape == (1, G.AUX_DEF_SLOTS, G.AUX_DEF_DIM)
    # 枠は [L1, A1, C1, 空, 空, 空]（行の時点＝手 0 の直前の場）
    leader, a1, c1, e3, e4, e5 = aux_def[0]
    assert leader.tolist() == [1.0, 0.0, 1.0, 0.0]      # 狙われた 1 回・失ったライフ 1・場に残った
    assert a1.tolist() == [1.0, 0.0, 0.0, 0.0]          # 狙われたが C1 に庇われて無傷・場に残った
    assert c1.tolist() == [1.0, 1.0, 0.0, G.LEFT_DEST_TRASH_BATTLE]   # 狙われて・ブロックも宣言し・戦闘で落ちた
    assert e3.tolist() == e4.tolist() == e5.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert aux_def_row[0].tolist() == [0.0, 1.0]        # カウンター 0・ブロック宣言 1


def test_dims_match_the_constants():
    assert (G.AUX_DEF_SLOTS, G.AUX_DEF_DIM, G.AUX_DEF_ROW_DIM) == (6, 4, 2)
    assert G.AUX_DEF_COLS == ("targeted", "blocked", "life_lost", "left_dest")
    assert G.AUX_DEF_ROW_COLS == ("counters_used", "blocks")


# --- 3. 場を離れた先の 4 通り（戦闘／効果でトラッシュ・手札・デッキ・残った）
FD0 = {"p1": [("A1", 5000, "n"), ("B1", 5000, "n"), ("C1", 5000, "n"), ("D1", 5000, "n"),
             ("E1", 5000, "n")], "p2": [("X2", 5000, "n")]}
FD1 = {"p1": [("E1", 5000, "n")], "p2": FD0["p2"]}          # A1/B1/C1/D1 が全部離れ、E1 だけ残る


def test_left_dest_four_ways():
    snaps = [
        _snap(1, "p1", {"p1": 5, "p2": 5}, FD0),                                  # 0: p1 PLAY
        _snap(1, "p1", {"p1": 5, "p2": 5}, FD0),                                  # 1: p1 TURN_END
        _snap(2, "p2", {"p1": 5, "p2": 5}, FD0),                                  # 2: p2 ATTACK→A1
        _snap(2, "p2", {"p1": 5, "p2": 5}, FD1,                                   # 3: p2 TURN_END
              trash={"p1": {"A1", "B1"}}, hand={"p1": {"C1"}}),
        _snap(3, "p1", {"p1": 5, "p2": 5}, FD1,
              trash={"p1": {"A1", "B1"}}, hand={"p1": {"C1"}}),                   # 4: 台帳の終わり
    ]
    steps = [
        _step("p1", "PLAY", uuid="H1"),
        _step("p1", "TURN_END"),
        _step("p2", "ATTACK", uuid="X2", target="A1"),
        _step("p2", "TURN_END"),
    ]
    assert len(snaps) == len(steps) + 1
    aux_def, _row = _def(snaps, steps, [0], ["p1"])
    leader, a1, b1, c1, d1, e1 = aux_def[0]
    assert leader.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert a1.tolist() == [1.0, 0.0, 0.0, G.LEFT_DEST_TRASH_BATTLE]   # 狙われてトラッシュ＝戦闘
    assert b1.tolist() == [0.0, 0.0, 0.0, G.LEFT_DEST_TRASH_EFFECT]   # 狙われずにトラッシュ＝効果
    assert c1.tolist() == [0.0, 0.0, 0.0, G.LEFT_DEST_HAND]           # 手札へ＝バウンス
    assert d1.tolist() == [0.0, 0.0, 0.0, G.LEFT_DEST_DECK]           # どこにも見えない残差＝デッキ
    assert e1.tolist() == [0.0, 0.0, 0.0, 0.0]                       # 場に残った


# --- 4. 行の視点自身が打ったカウンター・ブロックの回数（ターン全体）--------
def test_aux_def_row_counts_my_counters_and_blocks():
    f = {"p1": [("A1", 5000, "n")], "p2": [("X2", 5000, "n")]}
    snaps = [
        _snap(1, "p1", {"p1": 5, "p2": 5}, f), _snap(1, "p1", {"p1": 5, "p2": 5}, f),   # 0,1
        _snap(2, "p2", {"p1": 5, "p2": 5}, f), _snap(2, "p2", {"p1": 5, "p2": 5}, f),   # 2,3
        _snap(2, "p2", {"p1": 5, "p2": 5}, f), _snap(2, "p2", {"p1": 5, "p2": 5}, f),   # 4,5
        _snap(2, "p2", {"p1": 5, "p2": 5}, f), _snap(2, "p2", {"p1": 5, "p2": 5}, f),   # 6,7
        _snap(3, "p1", {"p1": 5, "p2": 5}, f),                                          # 8: 台帳の終わり
    ]
    steps = [
        _step("p1", "PLAY", uuid="H1"),
        _step("p1", "TURN_END"),
        _step("p2", "ATTACK", uuid="X2", target="L1"),
        _step("p1", "SELECT_COUNTER", uuid="CTR1"),
        _step("p2", "ATTACK", uuid="X2", target="L1"),
        _step("p1", "SELECT_COUNTER", uuid="CTR2"),
        _step("p2", "ATTACK", uuid="X2", target="A1"),
        _step("p1", "SELECT_BLOCKER", uuid="A1"),
    ]
    assert len(snaps) == len(steps) + 1
    _aux_def, aux_def_row = _def(snaps, steps, [0], ["p1"])
    assert aux_def_row[0].tolist() == [2.0, 1.0]


# --- 5. 空の台帳は形だけの 0 ------------------------------------------------
def test_empty_ledger_is_all_zero():
    aux_def, aux_def_row = G.aux_def_from_ledger([], [], [], [])
    assert aux_def.shape == (0, G.AUX_DEF_SLOTS, G.AUX_DEF_DIM)
    assert aux_def_row.shape == (0, G.AUX_DEF_ROW_DIM)
    assert aux_def.dtype == np.float32 and aux_def_row.dtype == np.float32
