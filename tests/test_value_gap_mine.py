"""価値実現ギャップ採掘（人間ログ活用(b)）のゲート。

固定する不変条件:
  - realization_gap＝勝者の予測勝率の最低点(comeback_depth)・敗者の最高点(throwaway_peak)・gap を正しく算出。
  - 勝者判定は y 基準（p1/p2 どちらでも）。境界は producer 順（p1,p2 交互）で分離。
  - 逆転局は gap 大・圧勝局は gap 小（弱点ランキングの向き）。
  - 退行ケース（行不足・モデル None）で None。
  - 同梱モデル実経路で {f,y} から妥当域 [0,1] のギャップを返す。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_value_gap_mine.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401
import pytest

from opcg_sim.src.core import cpu_features
from value_gap_mine import realization_gap, gap_from_dump


def _row(winprob, y):
    """予測器 lambda f: f[0] が winprob を返すよう先頭特徴に埋め込んだ有効行。"""
    return {"f": [winprob] + [0.0] * (cpu_features.N_FEATURES - 1), "y": y}


_PRED = lambda f: f[0]   # noqa: E731  (決定論予測器＝先頭特徴をそのまま勝率に)


def _game(win_traj, lose_traj, p1_won=True):
    """勝者/敗者の予測勝率列から producer 順（p1,p2 交互）の value_samples を組む。"""
    (p1t, p2t) = (win_traj, lose_traj) if p1_won else (lose_traj, win_traj)
    (y1, y2) = (1, 0) if p1_won else (0, 1)
    rows = []
    for a, b in zip(p1t, p2t):
        rows.append(_row(a, y1))
        rows.append(_row(b, y2))
    return rows


def test_comeback_game_has_large_gap():
    # 勝者(p1) が途中 0.1 まで沈んでから勝つ＝深い逆転。敗者(p2) は 0.9 まで勝ちに見えた。
    g = realization_gap(_game([0.2, 0.1, 0.9], [0.8, 0.9, 0.1], p1_won=True), predict=_PRED)
    assert g["winner"] == "p1"
    assert g["comeback_depth"] == pytest.approx(0.9) and g["comeback_turn"] == 1
    assert g["throwaway_peak"] == pytest.approx(0.9) and g["throwaway_turn"] == 1
    assert g["gap"] == pytest.approx(0.9)
    assert g["n_turns"] == 3


def test_dominant_game_has_small_gap():
    g = realization_gap(_game([0.85, 0.9, 0.95], [0.15, 0.1, 0.05], p1_won=True), predict=_PRED)
    assert g["gap"] == pytest.approx(0.15)   # comeback_depth=0.15, throwaway_peak=0.15


def test_winner_detection_p2():
    g = realization_gap(_game([0.3, 0.05, 0.7], [0.7, 0.95, 0.3], p1_won=False), predict=_PRED)
    assert g["winner"] == "p2"
    assert g["comeback_depth"] == pytest.approx(0.95) and g["comeback_turn"] == 1


def test_comeback_ranks_above_dominant():
    cb = realization_gap(_game([0.2, 0.1, 0.9], [0.8, 0.9, 0.1]), predict=_PRED)
    dom = realization_gap(_game([0.85, 0.9, 0.95], [0.15, 0.1, 0.05]), predict=_PRED)
    assert cb["gap"] > dom["gap"], "逆転局が圧勝局より上位に来ない"


def test_degenerate_returns_none():
    assert realization_gap([], predict=_PRED) is None
    assert realization_gap([_row(0.5, 1)], predict=_PRED) is None          # 1行＝両者揃わず
    # 予測器が None（モデル未同梱/特徴長不一致）を返したら採掘不能。
    assert realization_gap(_game([0.2, 0.9], [0.8, 0.1]), predict=lambda f: None) is None


def test_gap_from_envelope_with_bundled_model():
    # 同梱モデル実経路: 妥当な {f,y} を入れて [0,1] 域のギャップが返る（数値はモデル依存なので域のみ検証）。
    n = cpu_features.N_FEATURES
    rows = [{"f": [0.0] * n, "y": 1}, {"f": [1.0] * n, "y": 0},
            {"f": [0.5] * n, "y": 1}, {"f": [0.2] * n, "y": 0}]
    dump = {"replay": {"value_samples": rows}}
    g = gap_from_dump(dump)   # predict 省略＝同梱 value_model.json
    assert g is not None and g["winner"] == "p1"
    assert 0.0 <= g["comeback_depth"] <= 1.0 and 0.0 <= g["throwaway_peak"] <= 1.0
    assert 0.0 <= g["gap"] <= 1.0
