"""`rate_tracking.py`（T128）の器のテスト——**`A` が「このターンの損害」を追えているか**を測る器。

**基盤健全性ではない**——ここで守っているのは「**器が正しい量を出すか**」であり、
ゲームプレイの正しさではない。だが**計測の誤りは理論の誤判断に直結する**（T125／T126 の実害）ので
必須側に置く（マーカー無し・常時実行）。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import rate_tracking as RT  # noqa: E402


def _rows(n=200, noise=0.0, seed=0):
    """`turn_harm` と同じ形の行を作る（`g`／`who`／`j`／`harm`／`slope_theory`／`t_left`）。"""
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n):
        j = i % 5
        a = 0.05 * (j + 1)
        h = a + (rng.randn() * noise)
        out.append({"g": i // 10, "who": i % 2, "j": j, "harm": float(h),
                    "slope_theory": float(a), "priced": 0.0, "t_left": 3})
    return out


# --------------------------------------------------------------------------- 1. 部品
def test_at_clamps_to_the_last_bucket():
    """輪郭の索引は**末尾で止まる**（`tau_from_profile` と同じ規約）。"""
    prof = [0.0, 0.1, 0.2]
    assert RT.at(prof, 0) == 0.0
    assert RT.at(prof, 2) == 0.2
    assert RT.at(prof, 99) == 0.2          # 超えたら末尾
    assert RT.at([], 3) == 0.0             # 輪郭が無ければ 0


def test_corr_returns_none_for_a_constant_column():
    assert RT.corr([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert RT.corr([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- 2. とどめのターンの除外（T107 の規則）
def test_the_finishing_turn_is_excluded_by_default():
    rows = _rows(20)
    rows[0]["t_left"] = 1                  # とどめのターン
    rows[1]["t_left"] = 0
    assert RT.measure(rows, [0.1] * 6, [0.1] * 6)["n"] == 18
    assert RT.measure(rows, [0.1] * 6, [0.1] * 6, include_last=True)["n"] == 20


def test_no_rows_left_returns_an_empty_result_instead_of_dividing_by_zero():
    rows = _rows(4)
    for r in rows:
        r["t_left"] = 1
    assert RT.measure(rows, [0.1] * 6, [0.1] * 6) == {"n": 0}


# --------------------------------------------------------------------------- 3. 3 つの量
def test_a_perfect_rate_is_reported_as_perfect():
    """`A` がそのターンの損害と厳密に一致するなら、水準 1・相関 1・説明力 1。"""
    out = RT.measure(_rows(200, noise=0.0), [0.15] * 6, [0.15] * 6)
    assert out["level_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert out["corr_a"] == pytest.approx(1.0, abs=1e-6)
    assert out["r2_a"] == pytest.approx(1.0, abs=1e-6)


def test_the_level_ratio_is_theory_over_actual():
    """**向き**を固定する（1 より大きい＝理論が多い）。"""
    rows = _rows(50, noise=0.0)
    for r in rows:
        r["harm"] = r["slope_theory"] / 2.0
    assert RT.measure(rows, [0.1] * 6, [0.1] * 6)["level_ratio"] == pytest.approx(2.0, abs=1e-6)


def test_the_residual_correlation_is_zero_when_a_only_knows_the_turn_number():
    """**本命の量**——`A` が `j` の関数でしかないなら、輪郭を引いた残りは定数 0 ＝相関は `None`。

    「`A` が輪郭を超える情報を持つか」を測る量なので、**持たないときに 0 と言えなければ意味が無い**。
    """
    rows = _rows(200, noise=0.3, seed=1)               # 損害は暴れるが `A` は `j` だけの関数
    prof_th = [0.05 * (j + 1) for j in range(6)]       # 理論の輪郭 = `A` そのもの
    prof = [0.05 * (j + 1) for j in range(6)]
    out = RT.measure(rows, prof, prof_th)
    assert out["corr_resid"] is None                   # `A − prof_th[j]` が恒等的に 0


def test_the_residual_correlation_is_positive_when_a_carries_extra_information():
    rng = np.random.RandomState(2)
    rows, prof, prof_th = [], [0.1] * 6, [0.1] * 6
    for i in range(400):
        extra = float(rng.randn()) * 0.02              # `j` では説明できない当たり外れ
        rows.append({"g": i // 10, "who": i % 2, "j": 2, "harm": 0.1 + extra,
                     "slope_theory": 0.1 + extra, "t_left": 3})
    out = RT.measure(rows, prof, prof_th)
    assert out["corr_resid"] == pytest.approx(1.0, abs=1e-6)


def test_r2_does_not_refit_the_predictor():
    """**当てはめない**——2 倍ずれた予測は、相関が 1 でも説明力は負になる。"""
    rows = _rows(200, noise=0.0)
    for r in rows:
        r["slope_theory"] = r["harm"] * 2.0
    out = RT.measure(rows, [0.15] * 6, [0.15] * 6)
    assert out["corr_a"] == pytest.approx(1.0, abs=1e-6)
    assert out["r2_a"] < 0.0


def test_by_j_skips_thin_buckets_and_keeps_the_counts():
    rows = _rows(200)
    rows.append({"g": 99, "who": 0, "j": 9, "harm": 0.5, "slope_theory": 0.5, "t_left": 3})
    out = RT.measure(rows, [0.15] * 6, [0.15] * 6)
    assert "9" not in out["by_j"]                      # n < 20 は出さない
    assert sum(v["n"] for v in out["by_j"].values()) == 200


# --------------------------------------------------------------------------- 4. T129: 守りの量の表
def test_the_defence_table_flags_what_theta_already_holds():
    """**二重計上の判定に要る**——`in_theta` は「その量が既に耐久 `Θ` に入っているか」。

    ブロッカーは `THETA_BODY_MODE=blockers`（既定）で `Θ` の体の項に在り、盾は T102 で `Θ` の手札の項に在る。
    **`Θ` に在る量を `A` からも引くと同じ規則を 2 か所で数える**（T97 の実害）。
    `G`（規則が強いる守りの回数）と攻撃の本数は `Θ` に**数としては**入っていない。
    """
    flags = {k: in_theta for k, _label, in_theta in RT.DEFENCE}
    for k in ("d_blk_n", "d_blk_nu", "d_hand", "d_life", "d_shield", "d_shield_rate"):
        assert flags[k] is True, k
    for k in ("d_forced", "d_attacks"):
        assert flags[k] is False, k


def test_the_defence_correlation_is_against_the_residual_and_keeps_its_sign():
    """**守りが強いほど `A` は多く見積まれる**＝外れ（実測 − `A`）との相関は**負**に出る向きを固定する。"""
    rows, resid = [], []
    for i in range(200):
        blk = float(i % 4)
        r = -0.02 * blk                        # 守りが厚い行ほど理論が多く見積もる
        rows.append({"g": i, "who": 0, "j": 2, "harm": 0.1 + r, "slope_theory": 0.1,
                     "t_left": 3, "d_blk_n": blk, "d_forced": 0.0})
        resid.append(r)
    t = RT.defence_table(rows, np.asarray(resid))
    assert t["d_blk_n"]["corr_resid"] == pytest.approx(-1.0, abs=1e-6)
    assert t["d_blk_n"]["in_theta"] is True
    assert t["d_forced"]["corr_resid"] is None          # 定数の列は相関を出さない（None）
    assert "d_hand" not in t                            # 無い列は黙って落とす（落ちない）


def test_measure_carries_the_defence_table_through():
    rows = _rows(60)
    for i, r in enumerate(rows):
        r["d_blk_n"] = float(i % 3)
    out = RT.measure(rows, [0.15] * 6, [0.15] * 6)
    assert "d_blk_n" in out["defence"]
    assert out["defence"]["d_blk_n"]["nonzero_share"] == pytest.approx(2.0 / 3.0, abs=0.02)


def test_demean_by_removes_the_turn_number_trend():
    """**ターン番号の影を抜く**——`j` ごとの平均を引いたら、各 `j` の平均は厳密に 0。"""
    js = np.array([0, 0, 1, 1, 2, 2])
    xs = np.array([1.0, 3.0, 10.0, 20.0, 5.0, 5.0])
    d = RT.demean_by(xs, js)
    assert d == pytest.approx([-1.0, 1.0, -5.0, 5.0, 0.0, 0.0])
    for j in (0, 1, 2):
        assert d[js == j].mean() == pytest.approx(0.0, abs=1e-12)


def test_a_pure_turn_number_trend_shows_up_raw_and_vanishes_once_partialled():
    """**この器が在る理由**——`j` の関数でしかない量は、生の相関では出て、`j` を抜くと消える。"""
    rows, resid, js = [], [], []
    for i in range(300):
        j = i % 5
        rows.append({"g": i, "who": 0, "j": j, "harm": 0.0, "slope_theory": 0.0,
                     "t_left": 3, "d_life": 5.0 - j, "d_forced": 0.0})
        resid.append(0.02 * j)                       # 外れも `j` だけの関数
        js.append(j)
    t = RT.defence_table(rows, np.asarray(resid), js=np.asarray(js))
    assert abs(t["d_life"]["corr_resid"]) == pytest.approx(1.0, abs=1e-6)   # 生では満点に見える
    assert t["d_life"]["corr_resid_j"] is None                              # `j` を抜くと何も残らない


def test_a_column_readable_on_only_some_rows_is_measured_on_those_rows():
    """**T130**: 守る席の手札は**その席が 1 度も打っていない局面では読めない**（T76）。
    **列を丸ごと落とさず**、読める行だけで測って **`n` で母数を開示する**
    （黙って別の母数で比べないため）。"""
    rows, resid, js = [], [], []
    for i in range(100):
        r = {"g": i, "who": 0, "j": i % 4, "harm": 0.0, "slope_theory": 0.0, "t_left": 3}
        if i >= 40:                                  # 前半 40 行は読めない
            r["d_ctr_sum"] = float(i)
        rows.append(r); resid.append(float(i)); js.append(i % 4)
    t = RT.defence_table(rows, np.asarray(resid), js=np.asarray(js))
    assert t["d_ctr_sum"]["n"] == 60                 # 読めた行だけ
    assert t["d_ctr_sum"]["mean"] == pytest.approx(np.arange(40, 100).mean())
    assert t["d_ctr_sum"]["corr_resid"] == pytest.approx(1.0, abs=1e-6)   # 残差と揃って動く


def test_the_new_columns_are_the_two_the_user_named():
    """**T130**（ユーザ指摘「カウンター値と次ターン以降に出したいカードの都合じゃない？」）:
    **決めている 2 つ**が表に在り、**`Θ` には入っていない**と印がついている。"""
    flags = {k: in_theta for k, _label, in_theta in RT.DEFENCE}
    assert flags["d_ctr_sum"] is False               # カウンター値
    assert flags["d_play_sum"] is False              # 次ターン以降に出したい度（機会費用）
    assert flags["d_cut"] is False                   # 値と機会費用を天秤にかけた結果
    assert flags["d_guard_value"] is False


def test_the_common_subset_table_lines_the_columns_up_on_one_denominator():
    """**T130**: 列ごとに読める行数が違うと**どちらが大きいか言えない**
    （2026-09-20 に 2,947 行の列と 3,247 行の列を並べて比べかけた）。
    **全部の列が読める行だけ**の表を別に出す。"""
    rows = []
    for i in range(100):
        r = {"g": i, "who": 0, "j": i % 4, "harm": 0.1, "slope_theory": 0.1,
             "t_left": 3, "d_attacks": float(i % 3)}
        if i >= 40:
            r["d_cut"] = float(i % 2)
        rows.append(r)
    out = RT.measure(rows, [0.1] * 6, [0.1] * 6)
    assert out["defence"]["d_attacks"]["n"] == 100          # 生の表は列ごとの母数
    assert out["defence"]["d_cut"]["n"] == 60
    assert out["defence_common"]["d_attacks"]["n"] == 60    # 揃えた表は 1 つの母数
    assert out["defence_common"]["d_cut"]["n"] == 60


def test_no_common_table_when_every_column_is_readable_everywhere():
    """揃える必要が無ければ出さない（同じ表を 2 回出して読み手を迷わせない）。"""
    rows = _rows(60)
    for r in rows:
        r["d_attacks"] = 2.0
    assert "defence_common" not in RT.measure(rows, [0.15] * 6, [0.15] * 6)


def test_stopped_counts_attacks_while_cut_counts_cards():
    """**T131 の訂正**: **1 本止めるのに 2 枚使うことがある**——
    **本数から枚数を引くと単位が合わない**（割引率が 0.36 まで落ちた実害）。"""
    import hand_plan as HP
    # カウンター 1000 が 2 枚。超過 1000 の攻撃 1 本は **2 枚**使って **1 本**止まる。
    items = [{"counter": 1000.0, "v": None}, {"counter": 1000.0, "v": None}]
    xs = [1000.0]
    take = 1.0                                        # 受けると高いので必ず守る
    assert HP.counters_cut(items, xs, take) == 2.0    # 枚数
    assert HP.attacks_stopped(items, xs, take) == 1.0  # 本数
