"""`crossing_bridge.py`（T52・交点の橋）の算術を固める。

**当てはめない・回帰しない**器なので、しきい値・傾き・交点・予測勝者の定義がそのまま出ること、
残差が「実際に届いた側の τ 対 その側の残りターン」で作られること、単位の検算が比そのままであることを値で押さえる。
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

import crossing_bridge as CB  # noqa: E402
import price_realised as PR  # noqa: E402
import theory_order as T  # noqa: E402


def _sc(opp_life=3, opp_hand=4):
    sc = np.zeros(70, np.float32)
    sc[T.SC_OPP_LIFE], sc[T.SC_OPP_HAND] = opp_life, opp_hand
    sc[T.SC_MY_LEADER_POWER], sc[T.SC_OPP_LEADER_POWER] = 0.5, 0.5
    return sc


def test_the_threshold_is_the_opponents_endurance_in_price_units():
    tok = np.zeros((22, 24), np.float32)
    assert CB.threshold(_sc(3, 4), tok) == pytest.approx(3 * T.LAM + 4 * T.MU)
    tok[7, T.S_POWER], tok[7, T.S_IS_CHAR], tok[7, T.S_IS_BLOCKER] = 0.6, 1.0, 1.0      # アクティブなブロッカー 6000
    assert CB.threshold(_sc(3, 4), tok) == pytest.approx(3 * T.LAM + 4 * T.MU + PR.NU_MEAS["leader_to_sat"])
    tok[7, T.S_IS_REST] = 1.0                                                            # レスト中は数えない
    assert CB.threshold(_sc(3, 4), tok) == pytest.approx(3 * T.LAM + 4 * T.MU)


def test_the_theory_slope_is_the_priced_attack_flow_of_the_board():
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER], tok[1, T.S_POWER] = 0.5, 0.5
    lead_only = CB.theory_slope(tok, 5000.0)
    assert lead_only == pytest.approx(T.attack_value_don(5000.0, 5000.0, True))
    tok[2, T.S_POWER], tok[2, T.S_IS_CHAR], tok[2, T.S_CAN_ATTACK] = 0.8, 1.0, 1.0
    assert CB.theory_slope(tok, 5000.0) == pytest.approx(lead_only + T.attack_value_don(8000.0, 5000.0, True))


def test_the_crossing_picks_the_side_that_reaches_its_threshold_first():
    tau_me, tau_opp, pred = CB.predict(0.4, 0.4, 0.2, 0.1)
    assert (tau_me, tau_opp, pred) == (pytest.approx(2.0), pytest.approx(4.0), True)
    assert CB.predict(0.4, 0.4, 0.1, 0.2)[2] is False
    assert CB.predict(0.4, 0.4, 0.1, 0.1)[2] is True                          # 同数なら手番の自席
    assert CB.predict(0.4, 0.4, 0.0, 0.1)[0] == pytest.approx(0.4 / CB.SLOPE_FLOOR)   # 傾き 0 は届かない
    assert CB.harm_of({"opp_life": 0.136, "opp_hand": -0.05, "opp_body": 0.02, "my_life": -9, "my_hand": 9,
                       "my_body": 9, "don": 9}) == pytest.approx(0.106)


def _row(won, tau_me, tau_opp, t_me, t_opp):
    return {"who": 0, "won": won, "t_me_act": t_me, "t_opp_act": t_opp, "theta_me": 0.5, "theta_opp": 0.5,
            "tau_me_hist": tau_me, "tau_opp_hist": tau_opp, "pred_hist": tau_me <= tau_opp,
            "tau_me_theory": tau_me, "tau_opp_theory": tau_opp, "pred_theory": tau_me <= tau_opp}


def test_the_residual_uses_the_side_that_actually_reached_and_the_ledger_check_is_a_plain_ratio():
    rows = [_row(True, 2.0, 6.0, 2, 4)] * 30 + [_row(False, 5.0, 3.0, 4, 2)] * 30
    ledger = [{"F_end": 0.8, "theta_start": 1.0, "F_priced_end": 0.4}] * 10
    out = CB.summarise(rows, ledger)
    o = out["by_slope"]["hist"]
    assert o["sign_accuracy"] == 1.0
    assert o["bias"] == pytest.approx(0.5)                                     # 負けた行だけ 3 − 2 = 1
    assert o["sigma_T"] == pytest.approx(0.5)
    assert o["win_by_D"][">3"]["win_rate"] == 1.0 and o["win_by_D"]["-3..-1"]["win_rate"] == 0.0
    assert out["ledger"]["F_end_over_theta_start"] == pytest.approx(0.8)
    assert out["ledger"]["F_priced_over_F_real"] == pytest.approx(0.5)


def test_the_harm_profile_extends_its_last_value_and_the_profile_crossing_interpolates():
    """損害の輪郭は `j` ごとの平均・薄い先は最後の値を伸ばす。輪郭に沿った交点は端数を比例配分する。"""
    th = [{"j": 0, "harm": 0.05, "slope_theory": 0.1}] * 20 + [{"j": 1, "harm": 0.15, "slope_theory": 0.2}] * 20 + \
         [{"j": 2, "harm": 0.25, "slope_theory": 0.3}] * 5                  # j=2 は標本不足
    prof, prof_th = CB.harm_profile(th, j_max=4)
    assert prof == pytest.approx([0.05, 0.15, 0.15, 0.15, 0.15])
    assert prof_th == pytest.approx([0.1, 0.2, 0.2, 0.2, 0.2])
    assert CB.tau_from_profile(0.05, 0, prof) == pytest.approx(1.0)          # 1 ターン目でちょうど
    assert CB.tau_from_profile(0.125, 0, prof) == pytest.approx(1.5)         # 0.05 + 0.15/2
    assert CB.tau_from_profile(0.30, 1, prof) == pytest.approx(2.0)          # j=1 から 0.15 + 0.15
    assert CB.tau_from_profile(0.30, 1, prof, scale=2.0) == pytest.approx(1.0)
    rows = [_row(True, 2.0, 6.0, 2, 4) | {"j_me": 0, "j_opp": 0, "slope_theory_me": 0.2, "slope_theory_opp": 0.1}] * 30
    out = CB.summarise(rows, [], th)
    assert out["harm_profile"]["harm_by_turn"][:2] == pytest.approx([0.05, 0.15])
    assert set(out["by_slope"]) == {"hist", "theory", "curve", "curve_scaled"}
    assert out["by_slope"]["curve"]["sign_accuracy"] == 1.0                 # Θ が同じなら τ も同じ＝手番の自席


def test_my_endurance_mirrors_the_threshold_and_the_curve_d_is_the_tau_difference(tmp_path):
    """**T75**: 自分の耐久はしきい値の鏡（自ライフ・自手札・自分のアクティブなブロッカー）。交点の `D` = τ_opp − τ_me（正なら自分が先に届く）。
    輪郭は測る記録と別のセットのもの（`cross`）。"""
    sc = _sc(3, 4); sc[T.SC_MY_LIFE], sc[T.SC_MY_HAND] = 5, 2
    tok = np.zeros((22, 24), np.float32)
    assert CB.threshold_of_me(sc, tok) == pytest.approx(5 * T.LAM + 2 * T.MU)
    tok[2, T.S_POWER], tok[2, T.S_IS_CHAR], tok[2, T.S_IS_BLOCKER] = 0.6, 1.0, 1.0        # 自分のアクティブなブロッカー
    assert CB.threshold_of_me(sc, tok) == pytest.approx(5 * T.LAM + 2 * T.MU + PR.NU_MEAS["leader_to_sat"])
    prof = [0.1] * 12
    got = CB.curve_d_of_row(sc, tok, 0, prof)
    assert got["tau_me"] == pytest.approx(CB.threshold(sc, tok) / 0.1) and got["tau_opp"] == pytest.approx(CB.threshold_of_me(sc, tok) / 0.1)
    assert got["d"] == pytest.approx(got["tau_opp"] - got["tau_me"]) and got["d"] > 0            # 相手の耐久（3 ライフ）の方が小さい → 自分が先
    assert CB.own_turn_index(1) == 0 and CB.own_turn_index(2) == 0 and CB.own_turn_index(3) == 1 and CB.own_turn_index(8) == 3
    # 輪郭の表と cross の規則
    fx = tmp_path / "harm_profile.json"
    fx.write_text('{"real": [0.0, 0.1, 0.2], "syn": [0.0, 0.05, 0.1]}', encoding="utf-8")
    d_real = tmp_path / "w_real"; d_real.mkdir(); (d_real / "meta_n_record.json").write_text('{"decks": "user"}', encoding="utf-8")
    d_syn = tmp_path / "w_syn"; d_syn.mkdir(); (d_syn / "meta_n_record.json").write_text('{"decks": "synth"}', encoding="utf-8")
    assert CB.record_kind([str(d_real)]) == "real" and CB.record_kind([str(d_syn)]) == "syn"
    assert CB.record_kind([str(d_real), str(d_syn)]) is None and CB.record_kind([str(tmp_path)]) is None
    assert CB.profile_for([str(d_real)], "cross", str(fx)) == [0.0, 0.05, 0.1]                     # 実デッキには合成の輪郭
    assert CB.profile_for([str(d_syn)], "cross", str(fx)) == [0.0, 0.1, 0.2]
    assert CB.profile_for([str(d_syn)], "syn", str(fx)) == [0.0, 0.05, 0.1]
    assert CB.profile_for([str(tmp_path)], "cross", str(fx)) is None                               # 種類が判らなければ無い
    assert CB.profile_for([str(d_syn)], "cross", str(tmp_path / "missing.json")) is None


def test_theta_hand_mode_prices_the_hand_by_quality_instead_of_the_count(monkeypatch):
    """**T76**: 耐久の手札項は `μ × 枚数`（`count`・旧）か **札 1 枚あたりの実価格 × 枚数**（`quality`）。
    1 枚あたりの価格はその席の手札の札ごとの `max(ΔH_play, ΔG_guard)` の平均で、手札が空なら `μ` に落ちる。"""
    sc = _sc(3, 4); sc[T.SC_MY_LIFE], sc[T.SC_MY_HAND] = 5, 2
    tok = np.zeros((22, 24), np.float32)
    # 手札項だけが `g` で置き換わる（ライフ・体はそのまま）
    assert CB.threshold(sc, tok, g_hand=0.2) == pytest.approx(3 * T.LAM + 4 * 0.2)
    assert CB.threshold_of_me(sc, tok, g_hand=0.2) == pytest.approx(5 * T.LAM + 2 * 0.2)
    assert CB.threshold(sc, tok, g_hand=None) == pytest.approx(CB.threshold(sc, tok, g_hand=T.MU))
    # `D` は両席の `g` を別々に受ける（1 行から読めるのは自分の手札だけ＝もう片方は `μ`）
    prof = [0.1] * 12
    got = CB.curve_d_of_row(sc, tok, 0, prof, g_hand_of_me=0.2)
    assert got["theta_opp"] == pytest.approx(CB.threshold_of_me(sc, tok, g_hand=0.2))
    assert got["theta_me"] == pytest.approx(CB.threshold(sc, tok))                 # 相手側は μ のまま
    # 1 枚あたりの価格＝札ごとの `dtotal` の平均（器は差し替えて算術だけ見る）
    import hand_plan as HP
    monkeypatch.setattr(HP, "search_context",
                        lambda *a, **k: {"hand_items": [{"cid": "A"}, {"cid": "B"}], "caps": (1,), "xs": [], "take": 0.0})
    monkeypatch.setattr(HP, "card_deltas", lambda rest, card, caps, xs, take: {"dtotal": {"A": 0.1, "B": 0.3}[card["cid"]]})
    assert CB.hand_price_mean(sc, tok, np.zeros(24, np.int32), {}, None) == pytest.approx(0.2)
    monkeypatch.setattr(HP, "search_context", lambda *a, **k: {"hand_items": [], "caps": (1,), "xs": [], "take": 0.0})
    assert CB.hand_price_mean(sc, tok, np.zeros(24, np.int32), {}, None) == pytest.approx(T.MU)
    # 切替の検査
    with pytest.raises(ValueError):
        CB.set_theta_hand_mode("nope")
    assert CB.THETA_HAND_MODE == "cuttable"          # **既定は T77 で `cuttable` になった**（ユーザ決定 2026-09-17）


def test_the_hand_carries_two_values_cuttable_for_the_threshold_and_playable_for_the_rate(monkeypatch):
    """**T77**（ユーザ提案「手札に 2 つの価値を持たせる」）: **守る価値は耐久へ**（切れる札だけが `μ`＝`F` が切らせた札を `μ` で数えるから）・
    **出す価値は速さへ**（今のドンで出せる体の攻撃の価格を 1 ターンの損害に足す）。"""
    sc = _sc(3, 4); sc[T.SC_MY_LIFE], sc[T.SC_MY_HAND], sc[T.SC_MY_DON] = 5, 2, 4
    tok = np.zeros((22, 24), np.float32)
    import hand_plan as HP
    # 耐久側: 4 枚中 2 枚がカウンターを持つ → 1 枚あたりは μ の半分（＝μ × 切れる枚数）
    items = [{"cid": "A", "counter": 2000.0}, {"cid": "B", "counter": 0.0},
             {"cid": "C", "counter": 1000.0}, {"cid": "D", "counter": 0.0}]
    monkeypatch.setattr(HP, "search_context", lambda *a, **k: {"hand_items": items, "caps": (1,), "xs": [], "take": 0.0})
    g = CB.hand_price_mean(sc, tok, np.zeros(24, np.int32), {}, None, part="cuttable")
    assert g == pytest.approx(T.MU * 0.5)
    assert CB.threshold(sc, tok, g_hand=g) == pytest.approx(3 * T.LAM + 4 * T.MU * 0.5)   # 切れない札は耐久ではない

    # 速さ側: ドン 4 で出せる体の攻撃の価格の和（費用の重い札は入らない・イベントは体を持たない）
    class _Cards:
        DB = {"P2": {"power": 5000.0, "cost": 2}, "P3": {"power": 6000.0, "cost": 3},
              "P9": {"power": 9000.0, "cost": 9}, "EV": {"event": True, "cost": 1}}

        def info(self, cid):
            return dict(self.DB.get(cid) or {})
    cards = _Cards()
    hand = [{"cid": c, "cost": cards.DB[c]["cost"], "counter": 0.0} for c in ("P2", "P3", "P9", "EV")]
    olp = 5000.0
    one = T.attack_value_don(5000.0, olp, True, T.THETA, T.MU)
    two = T.attack_value_don(6000.0, olp, True, T.THETA, T.MU)
    assert CB.playable_attack_price(hand, cards, 5, olp) == pytest.approx(one + two)      # ドン 5 なら 2 + 3 の 2 体
    assert CB.playable_attack_price(hand, cards, 4, olp) == pytest.approx(max(one, two))  # 4 では 1 体だけ（費用 9 は出せない）
    assert CB.playable_attack_price(hand, cards, 2, olp) == pytest.approx(one)            # ドン 2 なら安い方だけ
    assert CB.playable_attack_price(hand, cards, 0, olp) == pytest.approx(0.0)
    assert CB.playable_attack_price([{"cid": "EV", "cost": 1, "counter": 0.0}], cards, 4, olp) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        CB.set_slope_mode("nope")
    assert CB.SLOPE_MODE == "hand" and CB.THETA_HAND_MODE == "cuttable"   # **既定は T77 の 2 値化**（ユーザ決定 2026-09-17）


def test_the_endurance_counts_bodies_the_same_way_the_harm_side_does():
    """**T82**（T81 の結論）: **`F` と `Θ` は同じものに同じ値段を付ける**。`F` の体の項は `price_realised.side_nu_meas`
    で**場の全キャラ**を数える（レストもブロッカー以外も・付与ドンを外した素のパワーで）ので、`THETA_BODY_MODE=all` なら
    `Θ` の体の項も**同じ関数の値**になる。`blockers`（旧）はアクティブなブロッカーだけ。"""
    sc = _sc(3, 4)
    sc[T.SC_MY_LEADER_POWER] = 0.5
    tok = np.zeros((22, 24), np.float32)
    tok[7, T.S_POWER], tok[7, T.S_IS_CHAR] = 0.6, 1.0                                  # ブロッカーでないキャラ
    tok[8, T.S_POWER], tok[8, T.S_IS_CHAR], tok[8, T.S_IS_BLOCKER] = 0.5, 1.0, 1.0     # アクティブなブロッカー
    tok[9, T.S_POWER], tok[9, T.S_IS_CHAR], tok[9, T.S_IS_BLOCKER] = 0.5, 1.0, 1.0
    tok[9, T.S_IS_REST] = 1.0                                                          # レスト中のブロッカー
    base = 3 * T.LAM + 4 * T.MU
    assert CB.THETA_BODY_MODE == "blockers"                                            # 既定は旧のまま
    assert CB.threshold(sc, tok) == pytest.approx(base + PR.NU_MEAS["leader_to_sat"])   # アクティブなブロッカー 1 体だけ
    try:
        assert CB.set_theta_body_mode("all") == "all"
        # **`F` が使う関数そのもの**と一致する（3 体ぜんぶ）
        assert CB.threshold(sc, tok) == pytest.approx(base + PR.side_nu_meas(tok, T.SLOT_OPP_FIELD, 5000.0))
        assert CB.threshold(sc, tok) > base + PR.NU_MEAS["leader_to_sat"]              # 体を出すほど耐久が増える
        with pytest.raises(ValueError):
            CB.set_theta_body_mode("なにか")
    finally:
        CB.set_theta_body_mode("blockers")
