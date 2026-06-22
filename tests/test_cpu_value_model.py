"""学習価値関数（§2.5.7 残5・GBDT/線形 価値葉）の健全性ゲート。

特徴抽出（`cpu_features`）とモデル推論（`cpu_value_model`）・葉ブレンド（`cpu_mcts._value_boundary`）が
「壊さない・決定論・manager 非破壊・既定OFFで現状同値・公平（相手手札の中身を読まない）」ことを固定する。
**ブレンドは既定OFF（`OPCG_VALUE_BLEND` 未設定=0）**＝本番 `evaluate`/hard も MCTS 既定挙動も不変。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_cpu_value_model.py -q -s -p no:cacheprovider
"""
import copy
import os
import random

import conftest  # noqa: F401
import pytest

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_features, cpu_value_model, cpu_mcts, journal
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


def test_feature_schema_unique_and_includes_eval_concepts(db):
    """特徴名は一意で、手作り評価の主要概念（非線形ライフ/攻め圧/脅威KW/デッキ危険域/ステージ）を含む。"""
    names = cpu_features.FEATURE_NAMES
    assert len(set(names)) == len(names) == cpu_features.N_FEATURES
    for required in ("life_thin_me", "life_thin_opp", "deck_danger_me", "deck_danger_opp",
                     "attacker_n_me", "attacker_n_opp", "threat_n_me", "threat_n_opp",
                     "stage_me", "stage_opp"):
        assert required in names, f"評価概念 {required} が特徴に無い"


def test_new_features_deterministic_and_fair(db):
    """追加特徴も決定論・非破壊・相手手札の中身に依存しない（公平）ことを固定する。"""
    m = _game(db)
    before = copy.deepcopy(m)
    f1 = cpu_features.extract_features(m, "p1")
    f2 = cpu_features.extract_features(m, "p1")
    assert f1 == f2
    assert journal.deep_diff(before, m) is None
    # life_thin は min(life,2) のバケット＝薄域の非線形を表す。
    idx = cpu_features.FEATURE_NAMES.index("life_thin_me")
    assert f1[idx] == float(min(len(m.p1.life), 2))


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


# --- 葉ブレンド（既定OFF＝現状同値） ---------------------------------------------

def test_blend_off_by_default_is_pure_eval(db):
    """`OPCG_VALUE_BLEND` 未設定なら _value_boundary は従来の eval ベース値と同一（推論を走らせない）。"""
    os.environ.pop("OPCG_VALUE_BLEND", None)
    m = _game(db)
    import math
    from opcg_sim.src.core.cpu_ai import evaluate
    ev = evaluate(m, "p1", see_opp_hand=False)
    base = 0.5 * (1.0 + math.tanh(ev / cpu_mcts.MCTS_VALUE_SCALE))
    assert cpu_mcts._value_boundary(m, "p1", see_opp_hand=False) == base
    assert cpu_value_model.blend_alpha() == 0.0


def test_blend_on_changes_value_and_is_deterministic(db):
    """`OPCG_VALUE_BLEND>0` でブレンドが効き（値が変わり）・同一入力で決定論。"""
    m = _game(db)
    base = cpu_mcts._value_boundary(m, "p1", see_opp_hand=False)
    os.environ["OPCG_VALUE_BLEND"] = "0.5"
    try:
        v1 = cpu_mcts._value_boundary(m, "p1", see_opp_hand=False)
        v2 = cpu_mcts._value_boundary(m, "p1", see_opp_hand=False)
        assert v1 == v2, "ブレンド葉が非決定論"
        assert abs(v1 - base) > 1e-9, "ブレンドONで値が変わっていない"
        assert 0.0 <= v1 <= 1.0
    finally:
        os.environ.pop("OPCG_VALUE_BLEND", None)


# --- hard(α-β) 葉ブレンド（Phase 2 本体・既定OFF同値＝最重要ゲート） ---------------

def test_hard_blend_off_by_default_is_bit_identical(db):
    """`OPCG_VALUE_BLEND_HARD` 未設定なら `evaluate` は素 eval とビット一致（推論も tanh/atanh も通さない）。

    これが Phase 2 の最重要不変条件＝既定 OFF で hard の挙動・決定論・カード挙動ベースラインが完全不変。
    """
    from opcg_sim.src.core.cpu_ai import evaluate, _hard_blend_alpha
    os.environ.pop("OPCG_VALUE_BLEND_HARD", None)
    cpu_value_model.set_alpha_override(None)
    assert _hard_blend_alpha() == 0.0
    for seed in (0, 1, 2):
        m = _game(db, seed=seed)
        # 数手進めて非自明な盤面でも一致を見る。
        ev_a = evaluate(m, "p1", see_opp_hand=True)
        ev_b = evaluate(m, "p1", see_opp_hand=True)
        assert ev_a == ev_b               # 決定論
        # α=0 override も素 eval とビット一致（推論経路を一切通さない）。
        cpu_value_model.set_alpha_override(0.0)
        ev_zero = evaluate(m, "p1", see_opp_hand=True)
        cpu_value_model.set_alpha_override(None)
        assert ev_zero == ev_a, "α=0 で素 eval とビット一致でない（既定OFF同値違反）"


def test_hard_blend_on_changes_eval_uses_fair_features_and_deterministic(db):
    """α>0 で `evaluate` が変わり・決定論・winprob は常にフェア特徴（see_opp_hand に依らず同一ブレンド）。"""
    from opcg_sim.src.core.cpu_ai import evaluate
    m = _game(db, seed=3)
    ev_off = evaluate(m, "p1", see_opp_hand=True)
    cpu_value_model.set_alpha_override(0.5)
    try:
        v1 = evaluate(m, "p1", see_opp_hand=True)
        v2 = evaluate(m, "p1", see_opp_hand=True)
        assert v1 == v2, "hard ブレンドが非決定論"
        assert abs(v1 - ev_off) > 1e-9, "α>0 で値が変わっていない"
        # winprob は常に see_opp_hand=False のフェア特徴で算出＝base のカンニング差はブレンド項に出るが
        # winprob 項自体は see_opp_hand に依存しない（フェア）。
        feat_fair = cpu_features.extract_features(m, "p1", see_opp_hand=False)
        p = cpu_value_model.predict_winprob(feat_fair)
        import math
        from opcg_sim.src.core.cpu_ai import _HARD_BLEND_SCALE, _HARD_BLEND_CLIP
        base = 0.5 * (1.0 + math.tanh(ev_off / _HARD_BLEND_SCALE))
        blended = min(1.0 - _HARD_BLEND_CLIP, max(_HARD_BLEND_CLIP, 0.5 * base + 0.5 * p))
        expected = _HARD_BLEND_SCALE * math.atanh(2.0 * blended - 1.0)
        assert abs(v1 - expected) < 1e-6, "hard ブレンドの数式（tanh→混合→atanh）が設計どおりでない"
    finally:
        cpu_value_model.set_alpha_override(None)
