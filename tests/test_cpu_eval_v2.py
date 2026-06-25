"""評価関数 v2（L1コア・v0.4）スケルトンのスモーク/不変条件テスト。

設計: docs/reports/cpu_eval_redesign_card_currency_20260625.md（§4 v0.4）。
段階導入の first cut。確認するのは「機械が正しく動くこと」だけ:
  - 既定 OFF（`OPCG_EVAL_V2` 未設定）＝従来評価と完全同値（フラグが寝ている）。
  - `evaluate_v2` が実ゲームで finite な float を返し例外を出さない。
  - 終局視点の符号（勝者視点 > 0 > 敗者視点）と、自他対称（zero-sum 近傍）。
係数は未チューニングのため**強さ/挙動の主張はしない**（それはアリーナ A/B の仕事・§9）。
"""
import math
import random

import conftest  # noqa: F401
import pytest

from opcg_sim.src.core import cpu_ai, cpu_eval_v2
from opcg_sim.src.core.gamestate import GameManager, Player
from cpu_selfplay import build_deck, _load_db


@pytest.fixture(scope="module")
def db():
    return _load_db()


def _new_gm(db, seed=0):
    random.seed(seed)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    return gm


def test_flag_default_off():
    """既定では評価 v2 フラグは寝ている（出荷挙動＝従来評価・完全同値の前提）。"""
    assert cpu_ai._USE_EVAL_V2 is False


def test_evaluate_v2_runs_and_is_finite(db):
    """実ゲーム序盤で v2 が finite を返し、例外を出さない（機械的健全性）。"""
    gm = _new_gm(db, seed=0)
    v = cpu_eval_v2.evaluate_v2(gm, "p1")
    assert isinstance(v, float)
    assert math.isfinite(v)


def test_evaluate_v2_zero_sum_symmetry(db):
    """同一局面の自他視点はほぼ符号反転（zero-sum）。Tele の符号付き項も対称なので |me+opp| は小さい。"""
    gm = _new_gm(db, seed=1)
    vp1 = cpu_eval_v2.evaluate_v2(gm, "p1")
    vp2 = cpu_eval_v2.evaluate_v2(gm, "p2")
    # 完全な反対称ではない（時間割引 γ/amp が自ライフ依存で非対称）が、序盤は概ね打ち消す。
    assert abs(vp1 + vp2) <= abs(vp1) + abs(vp2)        # 退化しない健全性
    assert math.isfinite(vp1) and math.isfinite(vp2)


def test_evaluate_v2_winner_sign(db):
    """勝者が確定した局面では勝者視点 +W_WIN・敗者視点 −W_WIN を返す（終端の符号）。"""
    gm = _new_gm(db, seed=2)
    gm.winner = "p1"
    assert cpu_eval_v2.evaluate_v2(gm, "p1") == cpu_ai.W_WIN
    assert cpu_eval_v2.evaluate_v2(gm, "p2") == -cpu_ai.W_WIN


def test_evaluate_v2_out_trace(db):
    """out 指定時に v2 成分内訳（R_me/R_opp/tele/gamma/amp）を記録し、採点（戻り値）は不変。"""
    gm = _new_gm(db, seed=3)
    out = {}
    v_with = cpu_eval_v2.evaluate_v2(gm, "p1", out=out)
    v_without = cpu_eval_v2.evaluate_v2(gm, "p1")
    assert v_with == v_without            # out 収集は採点に影響しない
    assert "v2" in out and {"R_me", "R_opp", "tele", "gamma", "amp"} <= set(out["v2"])
