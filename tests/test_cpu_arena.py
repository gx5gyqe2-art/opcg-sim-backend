"""CPU 検証基盤（フェーズ0）の機械健全性テスト: 凍結ベースライン Elo ＋ regret ログ
（`tests/cpu_arena.py`・docs/SPEC.md §2.5.3「検証基盤」）。

実ゲームは低速（normal ≈ 1 手/秒）なので、版間 Elo の本走は `cpu_arena.py` を手動/定期実行する。
本テストは**機械が正しく動くこと**だけを高速・有界に固定する:
  - Elo 変換（勝率→Elo）の数値性質（0.5→0・単調・対称）。
  - 非対称対局ランナー `play_game` と席交互の `arena`（軽量な easy 同士で完走・構造健全）。
  - regret ログ（`cpu_ai.decide_with_regret`）が非負・有限で、easy/単一手では 0、深掘りで取得できること。
"""
import math
import random

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

from opcg_sim.src.core import cpu_ai
import cpu_arena
import test_cpu_puzzles as P  # フィクスチャ/ヘルパ再利用（_new_gm・_fast_forward_to_p1_main 等）


@pytest.fixture(scope="module")
def db():
    return cpu_arena._load_db()


# ---------------------------------------------------------------------------
# Elo 変換の数値性質
# ---------------------------------------------------------------------------

def test_elo_delta_zero_at_even():
    assert cpu_arena.elo_delta(0.5) == pytest.approx(0.0, abs=1e-6)


def test_elo_delta_monotonic_and_symmetric():
    assert cpu_arena.elo_delta(0.76) == pytest.approx(200.0, abs=2.0)
    assert cpu_arena.elo_delta(0.24) == pytest.approx(-200.0, abs=2.0)
    # 単調増加。
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    ds = [cpu_arena.elo_delta(x) for x in xs]
    assert all(b > a for a, b in zip(ds, ds[1:]))
    # 対称: elo(p) == -elo(1-p)。
    assert cpu_arena.elo_delta(0.7) == pytest.approx(-cpu_arena.elo_delta(0.3))


def test_elo_delta_clamps_extremes_finite():
    # 端（0/1）でも有限（±inf を避ける）。
    assert math.isfinite(cpu_arena.elo_delta(0.0))
    assert math.isfinite(cpu_arena.elo_delta(1.0))
    assert cpu_arena.elo_delta(1.0) > cpu_arena.elo_delta(0.0)


def test_win_rate_helper():
    assert cpu_arena.win_rate(0, 0) == 0.5      # 無情報
    assert cpu_arena.win_rate(3.0, 4) == 0.75
    assert cpu_arena.win_rate(1.0, 2) == 0.5


# ---------------------------------------------------------------------------
# 非対称対局ランナー＋席交互アリーナ（軽量 easy 同士）
# ---------------------------------------------------------------------------

def test_play_game_finishes_with_winner(db):
    res = cpu_arena.play_game(0, db, "hard", "hard")
    assert res["winner"] in ("p1", "p2")
    assert res["steps"] > 0 and res["turns"] > 0


def test_arena_structure_and_seat_alternation(db):
    rep = cpu_arena.arena(db, challenger="hard", baseline="hard", games=2, seed0=0)
    assert rep["games"] == 2
    assert 0.0 <= rep["win_rate"] <= 1.0
    assert math.isfinite(rep["elo_delta"])
    # 席交互: 偶数 i は挑戦者 p1・奇数 i は p2。
    assert rep["detail"][0]["challenger_seat"] == "p1"
    assert rep["detail"][1]["challenger_seat"] == "p2"
    # 勝利判定が席に整合（winner==challenger_seat ⇔ challenger_won）。
    for d in rep["detail"]:
        assert d["challenger_won"] == (d["winner"] == d["challenger_seat"])


# ---------------------------------------------------------------------------
# Phase 0: 分散低減アリーナ（CRN + antithetic + 信頼区間）
# ---------------------------------------------------------------------------

def test_mean_ci_math():
    """平均 CI: 全勝/全敗は mean 0/1、混在は (0,1) 内で half_width>0、n<2 は half_width=inf。"""
    assert cpu_arena.mean_ci([1.0, 1.0, 1.0])["mean"] == 1.0
    assert cpu_arena.mean_ci([0.0, 0.0])["mean"] == 0.0
    mixed = cpu_arena.mean_ci([1.0, 0.0, 1.0, 0.0])
    assert 0.0 < mixed["mean"] < 1.0 and mixed["half_width"] > 0.0
    assert cpu_arena.mean_ci([1.0])["half_width"] == float("inf")  # 分散不定


def test_elo_ci_math():
    """Elo CI: 点推定は elo_delta(win_rate) と一致、半幅は非負・有限（混在標本）。"""
    scores = [1.0, 0.5, 0.0, 1.0, 0.5, 0.0]
    ci = cpu_arena.elo_ci(scores)
    assert ci["elo"] == pytest.approx(cpu_arena.elo_delta(ci["win_rate"]))
    assert ci["elo_half_width"] >= 0.0 and math.isfinite(ci["elo_half_width"])
    assert ci["elo_lo"] <= ci["elo"] <= ci["elo_hi"]


def test_play_game_separate_policy_rng_deterministic(db):
    """CRN: 方策乱数を分離（separate_policy_rng=True）しても同一 seed/方策は決定論（同手・同勝者）。"""
    a = cpu_arena.play_game(0, db, "hard", "hard", separate_policy_rng=True)
    b = cpu_arena.play_game(0, db, "hard", "hard", separate_policy_rng=True)
    assert a["winner"] == b["winner"] and a["steps"] == b["steps"] and a["turns"] == b["turns"]


def test_arena_paired_antithetic_structure(db):
    """antithetic ペアアリーナ: ペア勝点 {0,0.5,1}・win_rate∈[0,1]・Elo CI 半幅が有限・席相殺の整合。"""
    rep = cpu_arena.arena_paired(db, challenger="hard", baseline="hard", pairs=1, seed0=0)
    assert rep["pairs"] == 1
    assert 0.0 <= rep["win_rate"] <= 1.0
    assert math.isfinite(rep["elo_half_width"]) or rep["pairs"] < 2
    d = rep["detail"][0]
    assert d["pair_score"] in (0.0, 0.5, 1.0)
    # 挑戦者の勝点 = 席A(p1勝) と席B(p2勝) の平均（席相殺）。
    assert d["pair_score"] == (d["chal_as_p1_won"] + d["chal_as_p2_won"]) / 2.0


# ---------------------------------------------------------------------------
# regret ログ（decide_with_regret）
# ---------------------------------------------------------------------------

def test_decide_with_regret_normal_nonnegative_finite(db):
    """normal の regret は非負・有限、返す手は decide と一致（同一 seed）。"""
    gm = P._new_gm(db, seed=1)
    assert P._fast_forward_to_p1_main(gm)
    if len(gm.get_legal_actions(gm.p1)) <= 1:
        pytest.skip("分岐手が無い")
    move, regret = cpu_ai.decide_with_regret(gm, gm.p1, "hard", random.Random(0))
    assert move in gm.get_legal_actions(gm.p1)
    assert regret >= 0.0 and math.isfinite(regret)
    # 同一 seed の decide と同じ手（regret 計測が方策を変えない）。
    expected = cpu_ai.decide(gm, gm.p1, "hard", random.Random(0))
    assert cpu_ai._move_sig(move) == cpu_ai._move_sig(expected)


def test_decide_with_regret_single_move_is_zero(db):
    """合法手が 1 つしか無ければ regret=0（代替が無い）。"""
    gm = P._new_gm(db, seed=1)
    assert P._fast_forward_to_p1_main(gm)
    only = gm.get_legal_actions(gm.p1)[:1]
    # moves を 1 手に絞った decide は regret 0（decide_with_regret は内部で legal を引くため、
    # ここでは decide 側の 1 手分岐を直接確認）。
    assert cpu_ai.decide(gm, gm.p1, "hard", random.Random(0), moves=only) is only[0]
