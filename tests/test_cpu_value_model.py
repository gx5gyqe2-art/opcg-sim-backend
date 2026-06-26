"""学習価値関数（GBDT/線形 価値モデル）の健全性ゲート。

特徴抽出（`cpu_features`）とモデル推論（`cpu_value_model`）が「壊さない・決定論・manager 非破壊・公平
（相手手札の中身を読まない）」ことを固定する。**価値モデルは将来 α-β(hard) の葉へブレンドする土台**として
残置（MCTS撤去後・現状の本番 `evaluate`/hard は不変＝モデルの消費先は未配線）。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_cpu_value_model.py -q -s -p no:cacheprovider
"""
import copy
import os
import random

import conftest  # noqa: F401
import pytest

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_features, cpu_value_model, journal
import cpu_selfplay


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


# --- 特徴抽出 -------------------------------------------------------------------

def test_features_length_deterministic_and_nonmutating(db):
    m = _game(db)
    before = copy.deepcopy(m)
    f1 = cpu_features.extract_features(m, "p1")
    f2 = cpu_features.extract_features(m, "p1")
    assert len(f1) == cpu_features.N_FEATURES == len(cpu_features.FEATURE_NAMES)
    assert f1 == f2, "特徴抽出が非決定論"
    assert journal.deep_diff(before, m) is None, "特徴抽出が manager を変更した"
    assert all(isinstance(v, float) for v in f1)


def test_features_perspective_is_asymmetric(db):
    m = _game(db)
    assert cpu_features.extract_features(m, "p1") != cpu_features.extract_features(m, "p2")


def test_features_do_not_leak_opponent_hand_contents(db):
    """see_opp_hand=False（既定）は相手手札の**中身**を読まない＝中身を入れ替えても枚数が同じなら不変。"""
    m = _game(db)
    f_before = cpu_features.extract_features(m, "p1")
    # 相手手札の中身を別カードに差し替え（枚数は保つ）。
    opp = m.p2
    if len(opp.hand) >= 2:
        opp.hand[0], opp.hand[-1] = opp.hand[-1], opp.hand[0]
    # 山札の別カードを手札へ（枚数同じになるよう1枚交換）。
    if opp.deck:
        opp.hand[0], opp.deck[0] = opp.deck[0], opp.hand[0]
    f_after = cpu_features.extract_features(m, "p1")
    assert f_before == f_after, "相手手札の中身が p1 視点の特徴に漏れている（公平性違反）"


# --- モデル推論 -----------------------------------------------------------------

def test_model_available_and_matches_feature_schema(db):
    """同梱モデルが読め、特徴スキーマ（順序/個数）と一致する（不一致なら is_available=False で安全側）。"""
    assert cpu_value_model.is_available(), "value_model.json が読めない/スキーマ不一致"


def test_predict_in_unit_range_and_rejects_bad_length(db):
    m = _game(db)
    p = cpu_value_model.predict_winprob(cpu_features.extract_features(m, "p1"))
    assert p is not None and 0.0 <= p <= 1.0
    assert cpu_value_model.predict_winprob([0.0, 1.0]) is None, "長さ不一致は None であるべき"


# --- 価値ブレンドの素材（α-β 葉への将来配線用・現状は消費先なし） ----------------

def test_blend_alpha_off_by_default(db):
    """`OPCG_VALUE_BLEND` 未設定なら blend_alpha()==0＝モデル推論を走らせない（現状の本番挙動）。"""
    os.environ.pop("OPCG_VALUE_BLEND", None)
    assert cpu_value_model.blend_alpha() == 0.0


def test_gbdt_tree_predict_and_format():
    """GBDT 推論（最小）: 木の走査＝x[f]<=t で左/否右・葉値を返す。format/spec 検証も gbdt-v1 を許容。"""
    tree = {"f": 0, "t": 1.5, "l": {"v": -2.0}, "r": {"v": 3.0}}
    assert cpu_value_model._tree_predict(tree, [1.0]) == -2.0   # <= で左
    assert cpu_value_model._tree_predict(tree, [2.0]) == 3.0    # > で右
    assert cpu_value_model._tree_predict({"v": 0.5}, [9]) == 0.5  # 葉のみ
    # gbdt-v1 のスキーマ検証が通る（trees=list・n_features 一致・feature_names 一致）。
    ok = {"format": "gbdt-v1", "feature_names": cpu_features.FEATURE_NAMES,
          "n_features": cpu_features.N_FEATURES, "trees": [{"v": 0.0}]}
    assert cpu_value_model._valid_model(ok) is True
    assert cpu_value_model._valid_model({**ok, "n_features": 999}) is False
