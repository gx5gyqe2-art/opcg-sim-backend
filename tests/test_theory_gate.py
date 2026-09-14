"""`theory_gate.py`（フェーズの中間ゲート）の**循環しないことを固める**。

**この器の値打ちは「ホールドアウトから値をもらう経路が無い」ことだけ**にある。
その性質が壊れたら、数字は当てはまりの再生産になって意味を失う。
**だから条件そのものをテストで押さえる**（値の大小は押さえない——それは測定の仕事）。
"""
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import theory_gate as G  # noqa: E402
import theory_order as T  # noqa: E402


def _sc(my_life=4, opp_life=4, my_hand=5, opp_hand=5, my_don=5, opp_don=5):
    sc = np.zeros(127, np.float32)
    sc[T.SC_MY_LIFE] = my_life; sc[T.SC_OPP_LIFE] = opp_life
    sc[G.SC_OPP_HAND] = opp_hand; sc[T.SC_MY_HAND] = my_hand
    sc[T.SC_MY_DON] = my_don; sc[G.SC_OPP_DON] = opp_don
    sc[T.SC_MY_LEADER_POWER] = 0.5; sc[T.SC_OPP_LEADER_POWER] = 0.5
    return sc


def _tok(own=(), opp=()):
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = 0.5; tok[1, T.S_POWER] = 0.5
    for j, pw in enumerate(own):
        tok[2 + j, T.S_POWER] = pw / 1e4; tok[2 + j, T.S_IS_CHAR] = 1.0
    for j, pw in enumerate(opp):
        tok[7 + j, T.S_POWER] = pw / 1e4; tok[7 + j, T.S_IS_CHAR] = 1.0
    return tok


def test_it_refuses_to_measure_on_the_records_the_theory_was_built_from():
    """**条件 1**——当てはめた棋譜で測ったら、それは当てはまりの再生産である。

    重なりは**実行前に**拒否する（走ってから気付くのでは遅い）。
    """
    assert G.overlap(["/a/w41"], ["/a/w39", "/a/w41"]) == ["/a/w41"]
    assert G.overlap(["/a/w41"], ["/a/w39"]) == []
    assert G.overlap(["/a/w41"], []) == []
    # 綴りが違っても同じ場所なら拒否する（`.` を挟む・末尾の `/`）
    assert G.overlap(["/a/./w41/"], ["/a/w41"]) == ["/a/w41"]


def test_the_score_has_no_parameter_that_comes_from_the_holdout():
    """**条件 2/3**——採点は凍結した定数だけで決まる。

    同じ盤面を 2 回採点したら必ず同じ値になり、**行の集合に依らない**
    ＝ホールドアウトの分布から値をもらう経路が無い。
    """
    sc, tok = _sc(my_life=3, opp_life=1), _tok(own=(5000,), opp=(3000,))
    a = G.state_score(sc, tok)
    b = G.state_score(sc, tok)
    assert a == b
    # 定数を明示で渡しても既定と一致する（既定がどこか別から来ていない）
    assert G.state_score(sc, tok, T.THETA, T.MU, T.LAM, G.DELTA) == a


def test_the_score_moves_the_right_way_for_each_currency():
    """4 通貨それぞれで**符号が正しい**（価格は全部正の価値を持つ）。"""
    base = G.state_score(_sc(), _tok())
    assert G.state_score(_sc(my_life=5), _tok()) > base
    assert G.state_score(_sc(opp_life=5), _tok()) < base
    assert G.state_score(_sc(my_hand=7), _tok()) > base
    assert G.state_score(_sc(my_don=7), _tok()) > base
    assert G.state_score(_sc(), _tok(own=(5000,))) > base
    assert G.state_score(_sc(), _tok(opp=(5000,))) < base


def test_the_two_sides_are_scored_from_their_own_seat():
    """**相手の場を採点するときは席を入れ替える**——相手から見た「相手のリーダー」は自分。

    席を入れ替え忘れると、盤面を丸ごとひっくり返したのに評価が反転しない。
    """
    sc = _sc(my_life=2, opp_life=4, my_hand=3, opp_hand=6, my_don=4, opp_don=7)
    flip = _sc(my_life=4, opp_life=2, my_hand=6, opp_hand=3, my_don=7, opp_don=4)
    a = G.state_score(sc, _tok(own=(4000,), opp=(7000,)))
    b = G.state_score(flip, _tok(own=(7000,), opp=(4000,)))
    assert a == pytest.approx(-b, abs=1e-9)


def test_the_auc_is_rank_only_and_therefore_fits_nothing():
    """**条件 3**——AUC は順位だけで決まる＝単調変換で動かない。

    動くなら較正が混じっており、**ホールドアウトから値をもらっている**ことになる。
    """
    s = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9]
    y = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    a = G.auc(s, y)
    assert G.auc([x * 7.0 + 3.0 for x in s], y) == pytest.approx(a)     # 線形
    assert G.auc([x ** 3 for x in s], y) == pytest.approx(a)            # 単調非線形
    assert a == pytest.approx(1.0 - G.auc([-x for x in s], y))          # 反転で 1−a


def test_the_auc_handles_ties_and_degenerate_labels():
    assert G.auc([1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]) == pytest.approx(0.5)
    assert G.auc([1.0, 2.0], [1.0, 1.0]) is None       # 片方のクラスしか無い
    assert G.auc([1.0, 2.0], [0.0, 0.0]) is None


def test_the_position_needs_a_usable_ceiling():
    """**天井の目安が下限を超えないと `position` は出さない**（読めない数字を出さない）。"""
    assert G.position({"base": 0.60, "theory": 0.70, "net": 0.80}) == pytest.approx(0.5)
    assert G.position({"base": 0.60, "theory": 0.60, "net": 0.80}) == pytest.approx(0.0)
    assert G.position({"base": 0.60, "theory": 0.80, "net": 0.80}) == pytest.approx(1.0)
    assert G.position({"base": 0.70, "theory": 0.72, "net": 0.65}) is None   # 天井が下
    assert G.position({"base": 0.60, "theory": 0.70, "net": None}) is None


def test_the_frozen_constants_are_the_ones_the_theory_ships():
    """**定数は `theory_order` の正本を使う**——器が自前の写しを持つと黙ってずれる。"""
    assert (G.LAM, G.MU, G.THETA) == (T.LAM, T.MU, T.THETA)
    assert G.PREDICTORS == ("base", "theory", "net")
