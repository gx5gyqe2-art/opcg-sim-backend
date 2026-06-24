"""Leave-One-Game-Out 汎化検証ハーネス（`human_value_holdout`）の健全性ゲート。

固定する不変条件:
  - **グループ混入ゼロ**: どのフォールドでも held-out 対局の行が学習側に漏れない（汎化の正直さの核）。
  - 学習不能フォールド（単一クラス対局）の安全なスキップ。
  - `train_candidate` が本番推論経路（`predict_winprob`/`_valid_model`）が読める dict を返す。
  - `main` が読み取り専用＝同梱 `value_model.json` を書き換えない・決定論。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_human_value_holdout.py -q -s -p no:cacheprovider
"""
import json
import os

import conftest  # noqa: F401

from opcg_sim.src.core import cpu_features, cpu_value_model
import human_value_holdout as H

N = cpu_features.N_FEATURES


def _vec(v: float):
    return [float(v)] * N


# --- LOGO の機構（関数注入で純粋検証） --------------------------------------------

def test_logo_excludes_heldout_group():
    """train_fn には held-out 以外の行だけが渡る（グループ混入ゼロ）ことを直接検証。"""
    # 各グループを一意な特徴値で識別。学習側に held-out の値が混ざっていないことを確認する。
    groups = [(f"g{i}", [_vec(i)], [i % 2]) for i in range(3)]

    seen_train = []

    def train_fn(X, Y):
        vals = {row[0] for row in X}          # 学習に使われたグループ識別値
        seen_train.append(vals)
        return vals                            # 「モデル」= 見た学習値の集合

    def predict_fn(f, model):
        # held-out の値が学習集合に無ければ正しく除外できている → 1.0
        return 0.0 if f[0] in model else 1.0

    probs, ys, folds = H.logo_oof(groups, train_fn, predict_fn)
    assert len(probs) == 3 and len(ys) == 3
    assert all(p == 1.0 for p in probs), "held-out 行が学習側に混入している"
    # フォールド i の学習集合に i が含まれない・他は含まれる。
    for i, vals in enumerate(seen_train):
        assert float(i) not in vals
        assert vals == {float(j) for j in range(3) if j != i}
    assert all(f["trained"] for f in folds)


def test_logo_skips_unsmall_when_single_class_train():
    """残り全部が単一クラスだと学習不能 → そのフォールドは skip されメタに記録。"""
    # 2 グループ・両方とも y=1 のみ → どのフォールドでも train は単一クラス。
    groups = [("g0", [_vec(0)], [1]), ("g1", [_vec(1)], [1])]

    def train_fn(X, Y):
        return None if len(set(Y)) < 2 else object()

    probs, ys, folds = H.logo_oof(groups, train_fn, lambda f, m: 1.0)
    assert probs == [] and ys == []
    assert all(f["trained"] is False for f in folds)


def test_logo_separable_data_generalizes():
    """線形分離なデータなら未知フォールドでも高精度（OOF で汎化を捉える）。"""
    groups = []
    for g in range(4):
        X, Y = [], []
        for k in range(6):
            X.append(_vec(5.0)); Y.append(1)     # 正例クラスタ
            X.append(_vec(-5.0)); Y.append(0)    # 負例クラスタ
        groups.append((f"g{g}", X, Y))
    probs, ys, _ = H.logo_oof(groups, H.train_candidate, H._predict_with)
    assert H.E.accuracy(probs, ys) >= 0.9


# --- train_candidate の出力が本番経路で読める ------------------------------------

def test_train_candidate_schema_and_single_class():
    assert H.train_candidate([_vec(1)], [1]) is None        # 単一クラスは学習不能
    model = H.train_candidate([_vec(5)] * 4 + [_vec(-5)] * 4, [1, 1, 1, 1, 0, 0, 0, 0])
    assert model is not None
    # 本番ローダの検証を通る＝同梱と同一スキーマ。
    assert cpu_value_model._valid_model(model)
    p = cpu_value_model.predict_winprob(_vec(5), model=model)
    assert p is not None and 0.0 <= p <= 1.0


def test_train_gbdt_candidate_schema_and_single_class():
    assert H.train_gbdt_candidate([_vec(1)], [1]) is None    # 単一クラスは学習不能
    X = [_vec(5)] * 8 + [_vec(-5)] * 8
    Y = [1] * 8 + [0] * 8
    model = H.train_gbdt_candidate(X, Y, trees=10, depth=2)
    assert model is not None and model["format"] == "gbdt-v1"
    # 本番ローダ（gbdt-v1）の検証を通り、推論経路で勝率が出る。
    assert cpu_value_model._valid_model(model)
    p = cpu_value_model.predict_winprob(_vec(5), model=model)
    assert p is not None and 0.0 <= p <= 1.0


# --- metrics の素性 ----------------------------------------------------------------

def test_metrics_basic():
    m = H.metrics([0.9, 0.1], [1, 0])
    assert m["n"] == 2 and m["acc"] == 1.0
    assert m["pos_rate"] == 0.5 and m["base_acc"] == 0.5


# --- load_groups / 読み取り専用 end-to-end -----------------------------------------

def _write_capture(path, vecs_labels):
    dump = {"replay": {"value_samples": [{"f": f, "y": y} for f, y in vecs_labels]}}
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(dump, fp)


def test_load_groups_one_file_one_group(tmp_path):
    _write_capture(str(tmp_path / "a.json"), [(_vec(1), 1), (_vec(-1), 0)])
    _write_capture(str(tmp_path / "b.json"), [(_vec(2), 1)])
    _write_capture(str(tmp_path / "empty.json"), [])     # 有効サンプル無し→除外
    groups = H.load_groups(H.expand_dir(str(tmp_path)))
    assert [g[0] for g in groups] == ["a.json", "b.json"]
    assert len(groups[0][1]) == 2 and len(groups[1][1]) == 1


def test_main_readonly_and_runs(tmp_path):
    # 分離可能な 3 対局を合成。main がエラーなく完走し、同梱モデルを書き換えないこと。
    for g in range(3):
        rows = []
        for _ in range(8):
            rows.append((_vec(5), 1)); rows.append((_vec(-5), 0))
        _write_capture(str(tmp_path / f"g{g}.json"), rows)
    before = os.path.getmtime(cpu_value_model._MODEL_PATH)
    rc = H.main(["--in", str(tmp_path), "--epochs", "50"])
    after = os.path.getmtime(cpu_value_model._MODEL_PATH)
    assert rc == 0
    assert before == after, "main が同梱 value_model.json を書き換えた（読み取り専用違反）"


def test_main_needs_two_games(tmp_path):
    _write_capture(str(tmp_path / "only.json"), [(_vec(1), 1), (_vec(-1), 0)])
    assert H.main(["--in", str(tmp_path)]) == 1     # 1 対局では LOGO 不可


def test_compare_mode_readonly_and_runs(tmp_path):
    # 分離可能な 3 対局で --compare（同梱/線形/非線形）が完走し同梱モデルを書かない。
    for g in range(3):
        rows = []
        for _ in range(8):
            rows.append((_vec(5), 1)); rows.append((_vec(-5), 0))
        _write_capture(str(tmp_path / f"g{g}.json"), rows)
    before = os.path.getmtime(cpu_value_model._MODEL_PATH)
    rc = H.main(["--in", str(tmp_path), "--compare", "--epochs", "50", "--trees", "10", "--depth", "2"])
    assert rc == 0
    assert os.path.getmtime(cpu_value_model._MODEL_PATH) == before


def test_logo_metrics_helper_separable(tmp_path):
    # 分離可能データで LOGO ヘルパが定数予測超えの acc を返す。
    groups = [(f"g{g}", [_vec(5)] * 6 + [_vec(-5)] * 6, [1] * 6 + [0] * 6) for g in range(3)]
    m = H.logo_metrics(groups, H.train_candidate)
    assert m is not None and m["acc"] >= 0.9
