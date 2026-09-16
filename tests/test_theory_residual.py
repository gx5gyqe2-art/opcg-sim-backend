"""残差で理論の量を選別する算術（`tests/scripts/theory_residual.py`）。

基盤健全性（`cpu_infra`）。ネットも記録も要らない純関数だけを固める。要は 4 つ:

1. **選別は残差** `z − P̂` で行う（素の勝敗との相関では `V` が既に持っている分が混ざる）。
2. **帯の中だけで回帰する**（within）＝帯の固定効果を落とす。
3. **SE は対局でクラスタ**する（同じ局の行は独立でない・今日 2 回踏んだ罠）。
4. **順位は `|β|·sd`**（単位の違う量を並べるため）。`future` の量は
   `band_r2`（盤面から読めるか）を併記しないとヘッドの可否が判断できない。

**T11（`--split`）はもう 1 つの罠を潰す器**＝`future` の量は**勝敗の関数**なので、
`z` 以外に何の情報も無くても β が出る。`f = f̂(s) + u` に分けて、
**状態の関数である `f̂` が残差を持っているか**で本物かを決める。ここで固めるのは:
**(a) `f̂` は局で out-of-fold**（in-sample だと `u` を吸って artefact を持ち込む）・
**(b) artefact の世界と本物の世界を作り分けたとき判定が正しく出る**（両方向）・
**(c) 未学習の時間軸ヘッドは黙って使わせない**。
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

import theory_residual as R  # noqa: E402


def _rows(spec):
    """`spec` = [(band, seed, q, resid), …]。"""
    return [{"band": b, "seed": s, "q_x": q, "resid": r} for b, s, q, r in spec]


def test_within_slope_removes_the_band_level():
    """帯ごとに水準が違っても**帯の中の傾き**は取れる（pooled なら騙される形）。"""
    rows = _rows([("A", 1, 0.0, 0.0), ("A", 1, 1.0, 0.5),
                  ("B", 2, 10.0, 10.0), ("B", 2, 11.0, 10.5)])
    out = R.within_cluster_slope(rows, "q_x")
    assert out["beta"] == pytest.approx(0.5)       # 帯の中はどちらも傾き 0.5
    assert out["bands"] == 2 and out["n"] == 4 and out["games"] == 2


def test_cluster_se_ignores_duplicated_rows():
    """**全行を一律に複製しても SE は変わらない**（クラスタ頑健の要点）。

    行で割る素の SE なら √2 で縮む。局の構造が変わっていないので縮んではいけない。
    """
    base = _rows([("A", g, v, y) for g in (1, 2, 3, 4)
                  for v, y in ((0.0, 0.1 * g), (1.0, 0.4 * g))])
    one = R.within_cluster_slope(base, "q_x")
    twice = R.within_cluster_slope(base + [dict(r) for r in base], "q_x")
    assert one["beta"] == pytest.approx(twice["beta"])
    assert one["se"] == pytest.approx(twice["se"], rel=1e-6)
    assert one["games"] == twice["games"] == 4
    assert twice["n"] == 2 * one["n"]


def test_no_variation_and_single_game_are_refused():
    flat = _rows([("A", 1, 3.0, 0.1), ("A", 1, 3.0, 0.9)])      # q が動かない
    assert R.within_cluster_slope(flat, "q_x")["beta"] is None
    one = _rows([("A", 1, 0.0, 0.0), ("A", 1, 1.0, 1.0)])        # 1 局では SE が出せない
    assert R.within_cluster_slope(one, "q_x")["se"] is None


def test_effect_is_standardised_so_units_can_be_compared():
    """`|β|·sd`＝「1 標準偏差動いたときの残差の動き」＝単位に依らない。"""
    small = _rows([("A", 1, 0.0, 0.0), ("A", 1, 1.0, 0.1),
                   ("A", 2, 0.0, 0.0), ("A", 2, 1.0, 0.1)])
    # 同じ関係を 1000 倍の単位で書いた量（β は 1/1000 になるが effect は同じ）
    big = [{"band": r["band"], "seed": r["seed"], "q_x": r["q_x"] * 1000.0,
            "resid": r["resid"]} for r in small]
    a = R.within_cluster_slope(small, "q_x")
    b = R.within_cluster_slope(big, "q_x")
    assert a["beta"] == pytest.approx(b["beta"] * 1000.0)
    assert a["effect"] == pytest.approx(b["effect"])


def test_band_r2_says_whether_the_band_predicts_the_quantity():
    """`future` の量が**盤面から読めるか**の代理（ヘッドにできるかの第 2 因子）。"""
    predictable = _rows([("A", 1, 1.0, 0.0), ("A", 2, 1.0, 0.0),
                         ("B", 3, 9.0, 0.0), ("B", 4, 9.0, 0.0)])
    assert R.band_r2(predictable, "q_x") == pytest.approx(1.0)     # 帯で完全に決まる
    noise = _rows([("A", 1, 1.0, 0.0), ("A", 2, 9.0, 0.0),
                   ("B", 3, 1.0, 0.0), ("B", 4, 9.0, 0.0)])
    assert R.band_r2(noise, "q_x") == pytest.approx(0.0)           # 帯は何も言わない
    assert R.band_r2(_rows([("A", 1, 1.0, 0.0)]), "q_x") is None


def test_rank_splits_inputs_from_heads_and_orders_by_effect():
    """**無相関な量は効きが 0**（線形に縮めた量は標準化すると同じ効きになるので使えない）。"""
    recs = []
    for g in range(1, 11):
        for k, v in enumerate((0.0, 1.0)):
            recs.append({"band": "A", "seed": g, "resid": 0.3 * v,
                         "q_strong": v,
                         "q_weak": float((g + k) % 2),    # 残差と無相関
                         "f_future": v})
    out = R.rank(recs, min_effect=0.01)
    names = [r["name"] for r in out["ranked"]]
    assert names[0] in ("q_strong", "f_future")          # 効きの大きい順
    assert names[-1] == "q_weak"
    kinds = {r["name"]: r["kind"] for r in out["ranked"]}
    assert kinds["q_strong"] == R.KIND_NOW and kinds["f_future"] == R.KIND_FUTURE
    fut = next(r for r in out["ranked"] if r["name"] == "f_future")
    assert "band_r2" in fut                             # ヘッドの候補には第 2 因子を必ず付ける
    weak = next(r for r in out["ranked"] if r["name"] == "q_weak")
    assert weak["worth_adding"] is False                # 効きが小さい量は足さない
    assert R.verdict(out) == "add_inputs"


def test_verdict_says_when_the_net_already_has_everything():
    recs = []
    for g in range(1, 11):
        for v in (0.0, 1.0):
            recs.append({"band": "A", "seed": g, "resid": 0.0, "q_x": v})
    out = R.rank(recs, min_effect=0.01)
    assert R.verdict(out) == "net_already_has_it"
    assert R.verdict({"n": 0}) is None


def _split_rows(resid_from):
    """`f = q_a + w`（`q_a`＝状態から読める部分・`w`＝読めない残り）の合成行。

    `resid_from` で**残差を駆動するのがどちらか**を切り替える＝T11 の 2 つの世界を作る。
    `w` は局とターンの偶奇で振るので `q_a` と無相関（＝`f̂` が `w` を吸えない）。
    """
    recs = []
    for g in range(1, 41):
        for k in range(4):
            q_a = float(k)
            w = 1.0 if (g + k) % 2 == 0 else -1.0
            recs.append({"band": "A", "seed": g, "q_a": q_a, "q_b": float(g % 3),
                         "f_x": q_a + w, "resid": 0.3 * (w if resid_from == "w" else q_a)})
    return recs


def test_cross_fit_recovers_the_readable_part_only():
    """`f̂` は状態から読める部分だけを拾い、読めない残りは拾わない（局で out-of-fold）。"""
    recs = _split_rows("w")
    hat, y, fitted = R.cross_fit(recs, "f_x", ["q_a", "q_b"], folds=5)
    assert fitted == len(recs)
    # y = q_a + w（分散 1.25 + 1）から w は読めない ⇒ 説明できるのは q_a の分だけ
    r2 = 1.0 - float(((y - hat) ** 2).mean()) / float(y.var())
    assert 0.45 < r2 < 0.65
    # 拾ったのは q_a（`hat` は centered q_a とほぼ一致）
    q_c = R.center_in_band(recs, "q_a")
    assert float(np.corrcoef(hat, q_c)[0, 1]) > 0.99


def test_split_calls_the_outcome_artefact_by_its_name():
    """**読めない残りだけが残差を駆動する世界**＝`f` の β は勝敗の写し＝ヘッドの線は閉じる。"""
    sp = R.split_future(_split_rows("w"), "f_x", feats=["q_a", "q_b"], min_effect=0.01)
    assert sp["whole"]["beta"] is not None and abs(sp["whole"]["effect"]) > 0.01   # 素では効く
    assert abs(sp["hat"]["beta"]) < 0.01              # 状態から読める部分は効かない
    assert sp["unpredictable"]["beta"] == pytest.approx(0.3, abs=0.02)
    # `f̂` が十分強い（hat_r2 ≥ 0.5）ときだけ artefact と名乗れる
    assert sp["hat_r2"] >= R.MIN_HAT_R2
    assert sp["verdict"] == R.SPLIT_ARTEFACT


def test_a_weak_predictor_may_not_call_it_an_artefact():
    """**弱い `f̂` は artefact の偽陽性を作る**（読める分が `u` に残り `β(u)` が拾う）。

    `f̂` が帯の中の変動を半分も説明できていないなら判定を出さない——`β(f̂)` は
    「本物の情報」の**下限**で、`hat_r2` がその下限の被覆率だから。
    """
    recs = _split_rows("w")
    for r in recs:
        r["p_weak"] = 0.1 * r["q_a"]              # 大きく縮んだ予測（実際のネットの癖）
    sp = R.split_future(recs, "f_x", pred_key="p_weak", min_effect=0.01)
    assert sp["hat_r2"] < R.MIN_HAT_R2
    assert sp["verdict"] == R.SPLIT_WEAK          # artefact とは言わせない
    # 同じデータで強い予測器を渡せば artefact と出る＝違いは推定器の強さだけ
    for r in recs:
        r["p_strong"] = r["q_a"]
    strong = R.split_future(recs, "f_x", pred_key="p_strong", min_effect=0.01)
    assert strong["verdict"] == R.SPLIT_ARTEFACT


def test_split_calls_real_information_when_the_readable_part_carries_it():
    """**状態から読める部分が残差を駆動する世界**＝本物の情報＝ヘッドで直せる。"""
    sp = R.split_future(_split_rows("q_a"), "f_x", feats=["q_a", "q_b"], min_effect=0.01)
    assert sp["hat"]["beta"] == pytest.approx(0.3, abs=0.02)
    assert abs(sp["unpredictable"]["beta"]) < 0.01
    assert sp["verdict"] == R.SPLIT_REAL


def test_split_uses_a_given_predictor_instead_of_fitting_one():
    """`--pred-net`（r10 のヘッド）の経路＝**23 量の張る空間の外**の予測器を使う形。"""
    recs = _split_rows("q_a")
    for r in recs:
        r["p_x"] = r["q_a"]                      # 与えられた予測（ここでは完全な予測器）
    sp = R.split_future(recs, "f_x", pred_key="p_x", min_effect=0.01)
    assert sp["source"] == "p_x" and sp["rows_fitted"] == len(recs)
    assert sp["hat"]["beta"] == pytest.approx(0.3, abs=0.02)
    assert sp["verdict"] == R.SPLIT_REAL


def test_split_says_no_signal_when_neither_part_carries():
    recs = _split_rows("w")
    for r in recs:
        r["resid"] = 0.0
    sp = R.split_future(recs, "f_x", feats=["q_a", "q_b"], min_effect=0.01)
    assert sp["verdict"] == R.SPLIT_NONE
    assert R.split_verdict(None) is None
    assert R.split_future([], "f_x") is None


def test_an_untrained_time_head_is_refused():
    """未学習のヘッドは定数予測より悪い（r10 の報告で 1.698 対 0.613）＝黙って使わせない。"""
    class _Stub:
        time = False
    with pytest.raises(SystemExit):
        R.time_predictions(_Stub(), None, None, None, None)


def test_now_quantities_are_computed_from_the_state_only():
    """`G = max(0, N − L − B)` などが状態から出ること（入力列にできる条件）。"""
    sc = np.zeros(20, np.float32)
    sc[R.SC_MY_LIFE], sc[R.SC_OPP_LIFE] = 3, 2
    sc[R.SC_MY_HAND], sc[R.SC_MY_DON] = 5, 4
    sc[R.SC_MY_FIELD], sc[R.SC_OPP_FIELD] = 1, 2
    sc[R.SC_MY_LEADER_POWER], sc[R.SC_OPP_LEADER_POWER] = 0.5, 0.5   # 5000 / 5000
    sc[R.SC_TURN] = 6
    tok = np.zeros((22, 22), np.float32)
    tok[R.SLOT_OWN_FIELD.start, R.S_POWER] = 0.7          # 7000 のキャラ（超過 +2000）
    tok[R.SLOT_OPP_FIELD.start, R.S_POWER] = 0.6          # 相手 6000（自分への超過 +1000）
    tok[R.SLOT_HAND.start, R.S_COUNTER] = 1000.0
    q = R.now_quantities(sc, tok)
    assert q["A"] == 3.0                                  # 相手の場 2 ＋ リーダー
    assert q["t_hat"] == 1.0                              # 相手ライフ 2 / 自分の攻撃 2 → 1 ターン
    assert q["N"] == 3.0 and q["B"] == 0.0
    assert q["G"] == pytest.approx(max(0.0, 3.0 - 3.0 - 0.0))
    assert q["x_max"] == pytest.approx(2000.0)            # 7000 − 5000
    assert q["c_in"] == pytest.approx(1.00)               # +1000 を止めるのは 1 枚
    assert q["n_dead_attackers"] == 0.0
    # 飽和点（Θ=1.15 で 2000）を超えていない（**f32 の丸めを 10 の桁で吸収している**）
    assert q["x_over_sat"] == pytest.approx(0.0)
