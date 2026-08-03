"""反実仮想ペア教師生成（v24・`tests/scripts/counterfactual_pair_gen.py`）の純関数テスト。

実対局・実ロールアウトは回さない。固定する性質:
  - **採掘条件**＝化粧系（PLAY/ATTACH_DON/ACTIVATE_MAIN）と進行系（ATTACK/TURN_END）が
    同時に合法な点のみ（防御窓は TURN_END を含まないため自然に除外される）
  - **代表手の選抜**＝行動種ごとに1手先 value 最良（ネットが選びがちな側を対照する）・
    種が1つしか無い点は対照が組めない（呼び出し側が捨てる契約）
  - causal_z の値域と worlds 正規化
  - sample_points はコスト制御の唯一の間引き（上限以下は全採用・決定的）
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import numpy as np
import pytest

import counterfactual_pair_gen as CF

pytestmark = pytest.mark.cpu_infra


def test_qualifies_needs_both_sides():
    assert CF.qualifies(["PLAY", "ATTACK", "TURN_END"])
    assert CF.qualifies(["ATTACH_DON", "TURN_END"])
    assert CF.qualifies(["ACTIVATE_MAIN", "ATTACK", "TURN_END", "PLAY"])
    assert not CF.qualifies(["ATTACK", "TURN_END"])          # 化粧系なし＝対照の意味がない
    assert not CF.qualifies(["PLAY", "ATTACH_DON"])          # 進行系なし（メイン決定では実質無い）
    assert not CF.qualifies(["SELECT_COUNTER", "PASS"])      # 防御窓は対象外
    assert not CF.qualifies([])


def test_pick_branches_one_representative_per_type_by_value():
    descs = [{"action_type": "PLAY", "card": "A"},
             {"action_type": "PLAY", "card": "B"},
             {"action_type": "TURN_END"},
             {"action_type": "ATTACK", "card": "C"},
             {"action_type": "SELECT_COUNTER"}]              # 対象外種は無視
    vals = [0.1, 0.5, -0.2, 0.0, 9.9]
    assert CF.pick_branches(descs, vals) == [1, 2, 3]        # PLAY は value 最良の B 側
    # value 取得失敗（None）の手は代表になれない
    assert CF.pick_branches(descs, [0.1, None, -0.2, 0.0, None]) == [0, 2, 3]


def test_pick_branches_single_type_gives_no_contrast():
    descs = [{"action_type": "TURN_END"}]
    assert len(CF.pick_branches(descs, [0.0])) < 2           # 呼び出し側が捨てる契約


def test_causal_z_range_and_normalization():
    assert CF.causal_z(0, 4) == pytest.approx(-1.0)
    assert CF.causal_z(4, 4) == pytest.approx(1.0)
    assert CF.causal_z(2, 4) == pytest.approx(0.0)
    assert CF.causal_z(3, 4) == pytest.approx(0.5)


def test_sample_points_cap_and_determinism():
    rng = np.random.default_rng(7)
    assert CF.sample_points([1, 1, 2], 5, rng) == [0, 1, 2]  # 上限以下＝全採用
    turns = [1] * 50 + [2] * 50
    a = CF.sample_points(turns, 4, np.random.default_rng(11))
    b = CF.sample_points(turns, 4, np.random.default_rng(11))
    assert a == b and len(a) == 4 and a == sorted(a)         # 決定的・昇順・重複なし
    assert len(set(a)) == 4


def test_sample_points_spreads_across_turns():
    """同一ターンに固まった候補でも、まず各ターンから1点ずつ取る（カバレッジ優先）。"""
    turns = [1] * 10 + [2] * 10 + [3] * 10 + [9]
    picked = CF.sample_points(turns, 4, np.random.default_rng(3))
    assert sorted({turns[i] for i in picked}) == [1, 2, 3, 9]   # 4点で4ターンを被覆
    picked6 = CF.sample_points(turns, 6, np.random.default_rng(3))
    assert {turns[i] for i in picked6} == {1, 2, 3, 9}          # 5点目以降は2巡目で埋める
    assert len(picked6) == 6
