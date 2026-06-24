"""価値モデルの外部セット検証ハーネス（`eval_value_on_set`・検証セット(a)）の健全性ゲート。

純粋指標（acc/logloss/Brier/ECE）の計算が正しいこと、`predict_winprob(features, model=...)` の
明示モデル経路が同梱モデル経路と**同一情報源**で一致すること、`evaluate` が読み取り専用・決定論で
正常な範囲の値を返すことを固定する。本ハーネスは `value_model.json` を一切書き換えない。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_eval_value_on_set.py -q -s -p no:cacheprovider
"""
import copy
import math
import random

import conftest  # noqa: F401
import pytest

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_features, cpu_value_model, journal
import cpu_selfplay
import eval_value_on_set as E


@pytest.fixture(scope="module")
def db():
    return cpu_selfplay._load_db()


def _game(db, seed=0):
    random.seed(seed)
    l1, c1 = cpu_selfplay.build_deck(db, "p1")
    l2, c2 = cpu_selfplay.build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    return m


# --- 純粋指標（既知値で厳密検証） --------------------------------------------------

def test_accuracy_logloss_brier_known_values():
    probs = [0.9, 0.1, 0.8, 0.4]
    ys = [1, 0, 1, 0]
    assert E.accuracy(probs, ys) == 1.0   # 全て正しい側に分類（0.4<0.5→0 も正）
    # logloss = 平均 -log(正解確率)。0.9,0.9,0.8,0.6
    exp_ll = -(math.log(0.9) + math.log(0.9) + math.log(0.8) + math.log(0.6)) / 4
    assert abs(E.logloss(probs, ys) - exp_ll) < 1e-12
    exp_brier = (0.1**2 + 0.1**2 + 0.2**2 + 0.4**2) / 4
    assert abs(E.brier(probs, ys) - exp_brier) < 1e-12


def test_accuracy_boundary_and_misclassification():
    # p=0.5 は勝ち予測扱い（>=0.5）。y=0 なら誤り。
    assert E.accuracy([0.5], [0]) == 0.0
    assert E.accuracy([0.5], [1]) == 1.0
    assert E.accuracy([0.3, 0.7], [1, 0]) == 0.0   # 両方反対


def test_calibration_buckets_and_ece():
    # 完全キャリブレーション: 予測0.0の群は実0.0、予測1.0近傍の群は実1.0。
    probs = [0.05, 0.05, 0.95, 0.95]
    ys = [0, 0, 1, 1]
    rows, ece = E.calibration(probs, ys, n_bins=10)
    assert ece < 0.06   # 平均予測≈実勝率
    # 件数の合計は全標本。
    assert sum(cnt for _, _, cnt, _, _ in rows) == 4
    # 誤キャリブレーション例: 0.9 と予測したのに実は全敗 → ECE 大。
    _, ece_bad = E.calibration([0.9, 0.9], [0, 0], n_bins=10)
    assert ece_bad > 0.8


# --- 明示モデル経路＝同梱経路と同一（単一情報源） --------------------------------

def test_explicit_model_matches_bundled(db):
    """`predict_winprob(f, model=load_model_file(同梱パス))` は既定経路と完全一致。"""
    assert cpu_value_model.is_available()
    loaded = cpu_value_model.load_model_file(cpu_value_model._MODEL_PATH)
    assert loaded is not None
    m = _game(db)
    f = cpu_features.extract_features(m, "p1")
    assert cpu_value_model.predict_winprob(f, model=loaded) == cpu_value_model.predict_winprob(f)


def test_load_model_file_rejects_bad_schema(tmp_path):
    import json
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"format": "logreg-standardized-v1", "feature_names": ["x"]}))
    assert cpu_value_model.load_model_file(str(p)) is None
    assert cpu_value_model.load_model_file(str(tmp_path / "missing.json")) is None


# --- evaluate エンドツーエンド（読み取り専用・決定論） ----------------------------

def test_evaluate_runs_readonly_and_deterministic(db):
    m = _game(db)
    before = copy.deepcopy(m)
    X = [cpu_features.extract_features(m, "p1"), cpu_features.extract_features(m, "p2")]
    Y = [1, 0]
    r1 = E.evaluate(X, Y)
    r2 = E.evaluate(X, Y)
    assert journal.deep_diff(before, m) is None, "evaluate が manager を変更した"
    assert r1 == r2, "evaluate が非決定論"
    assert r1["n"] == 2 and 0.0 <= r1["acc"] <= 1.0
    assert r1["logloss"] >= 0.0 and 0.0 <= r1["brier"] <= 1.0 and r1["ece"] >= 0.0


def test_load_rows_filters_bad_rows(tmp_path):
    import json
    p = tmp_path / "set.jsonl"
    good = {"f": [0.0] * cpu_features.N_FEATURES, "y": 1}
    bad_len = {"f": [0.0, 1.0], "y": 0}
    bad_y = {"f": [0.0] * cpu_features.N_FEATURES, "y": 2}
    p.write_text("\n".join(json.dumps(r) for r in (good, bad_len, bad_y, good)) + "\n")
    X, Y = E.load_rows(str(p))
    assert len(X) == 2 and Y == [1, 1]   # 不正2行は除外
