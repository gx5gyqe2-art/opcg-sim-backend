"""オプションペア教師生成（v31・`tests/scripts/option_pair_gen.py`）の純関数テスト。

実対局・実ロールアウトは回さない。固定する性質:
  - **カード単位の枝**（v24 の行動種分岐では m4@2＝PLAY vs PLAY を対照できなかった修正の核心）:
    同 card_id の PLAY は代表1つ・別 card_id は別枝・TURN_END を温存枝として1つ足す
  - qualifies＝ON_PLAY 持ちの PLAY が2枚以上（m4@2 パターンの採掘条件）
  - causal_z / spread の値域（順位教師の有情報性モニタの土台）
  - build_rank_pairs が読む形式（group + value）と整合＝生成→順位学習が繋がる
"""
import conftest  # noqa: F401
import numpy as np
import pytest

import option_pair_gen as G

pytestmark = pytest.mark.cpu_infra


def _d(action_type, card=None, onplay=False):
    return {"action_type": action_type, "card": card, "onplay": onplay}


def test_option_branches_are_card_level_plus_hold():
    """PLAY はカード単位・同名は代表1つ・TURN_END を温存枝として末尾に足す。"""
    descs = [_d("PLAY", "ST30-004", True), _d("PLAY", "OP13-007", True),
             _d("PLAY", "ST30-004", True),           # 同名複製＝代表1つに畳む
             _d("ATTACK", "X"), _d("TURN_END")]
    br = G.option_branches(descs)
    assert br == [0, 1, 4]                            # イワンコフ・A&S&L・TURN_END（ATTACK は対象外）


def test_option_branches_needs_no_end_when_absent():
    descs = [_d("PLAY", "A", True), _d("PLAY", "B", True)]
    assert G.option_branches(descs) == [0, 1]


def test_qualifies_requires_two_onplay_plays():
    """ON_PLAY 持ちの PLAY が2枚以上＝『どの登場時カードを今出すか』の点（m4@2 パターン）。"""
    assert G.qualifies([_d("PLAY", "A", True), _d("PLAY", "B", True), _d("TURN_END")])
    assert not G.qualifies([_d("PLAY", "A", True), _d("PLAY", "B", False)])   # 片方バニラ
    assert not G.qualifies([_d("PLAY", "A", True), _d("TURN_END")])           # 1枚だけ


def test_causal_z_and_spread():
    assert G.causal_z(6, 6) == pytest.approx(1.0)
    assert G.causal_z(0, 6) == pytest.approx(-1.0)
    assert G.causal_z(3, 6) == pytest.approx(0.0)
    assert G.causal_z(5, 0) == 0.0
    assert G.spread([1.0, -0.5, 0.25]) == pytest.approx(1.5)
    assert G.spread([0.3, 0.3]) == pytest.approx(0.0)                          # 全枝同値＝無情報
    assert G.spread([]) == 0.0


def test_sample_points_spreads_across_turns_and_deterministic():
    turns = [2] * 5 + [3] * 5 + [4]
    a = G.sample_points(turns, 3, np.random.default_rng(1))
    b = G.sample_points(turns, 3, np.random.default_rng(1))
    assert a == b and sorted({turns[i] for i in a}) == [2, 3, 4]
    assert G.sample_points([2, 3], 5, np.random.default_rng(0)) == [0, 1]      # 上限未満は全部


def test_output_feeds_build_rank_pairs():
    """生成物（group + value）を v12.1 の build_rank_pairs がそのまま順位ペアにできる。"""
    from ref_finetune_smoke import build_rank_pairs
    child = {"value": np.array([1.0, -0.5, 0.0, 0.9, -0.9], np.float32),
             "group": np.array([10, 10, 10, 11, 11], np.int64)}
    pairs = build_rank_pairs(child, delta=0.25)
    # group10: (0>1),(0>2) が δ 超え / group11: (3>4)。勝ち idx が先。
    assert (0, 1, 10) in pairs and (3, 4, 11) in pairs
    assert all(child["value"][w] > child["value"][l] for w, l, _ in pairs)
