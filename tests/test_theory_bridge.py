"""`theory_bridge.py`（理論と勝敗を繋ぐ橋・T28）の符号と「できない」の扱いを固める。

**符号が 1 つ狂うと結論が反転する器**なので、向きを値で押さえる。
守り側の**「守れなかった行を誤りと数えない」**は `measurement.md` §1 そのもの。
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

import theory_bridge as B  # noqa: E402
import theory_order as T  # noqa: E402


def _tok(opp_lead=5000, my_lead=5000, blocker=False):
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = my_lead / 1e4
    tok[1, T.S_POWER] = opp_lead / 1e4
    if blocker:
        tok[2, T.S_POWER] = 0.3
        tok[2, T.S_IS_CHAR] = 1.0
        tok[2, B.GA.S_BLOCKER] = 1.0
    return tok


def _sc(don=5):
    sc = np.zeros(127, np.float32)
    sc[T.SC_MY_DON] = don
    return sc


def test_a_step_is_never_positive():
    """`s_t` は「最善からの取りこぼし」なので **0 以下**。正が出たら向きが逆。"""
    tok, sc = _tok(opp_lead=8000), _sc()
    for played in ("take", "guard"):
        got = B.guard_step(tok, sc, played, free=9000.0, paid=[])
        assert got["s"] <= 0.0


def test_doing_what_the_theory_advises_costs_nothing():
    """助言どおりに打った行は **0**——減点は「違う手を打った」ときだけ。"""
    tok, sc = _tok(opp_lead=6000), _sc()          # 超過 1000＝c(1000)=1.00 < Θ=1.15 → 守る
    got = B.guard_step(tok, sc, "guard", free=9000.0, paid=[])
    assert got["theory_says"] == "guard"
    assert got["s"] == pytest.approx(0.0)


def test_taking_when_the_theory_says_guard_is_penalised():
    """安く守れたのに受けた＝取りこぼし。**`Θ·μ − c(x)·μ` ちょうど**になる。"""
    tok, sc = _tok(opp_lead=6000), _sc()
    got = B.guard_step(tok, sc, "take", free=9000.0, paid=[])
    expect = -(T.THETA * T.MU - T.c_of(1000.0) * T.MU)
    assert got["s"] == pytest.approx(expect)
    assert got["s"] < 0.0


def test_a_guard_you_could_not_afford_is_not_a_mistake():
    """**「〜しない」ではなく「〜できない」**（`measurement.md` §1）。

    カウンターを持っていなければ受けるしかない。そこを減点すると
    **能力の限界を選択の誤りとして数える**ことになる。
    """
    tok, sc = _tok(opp_lead=6000), _sc(don=0)
    poor = B.guard_step(tok, sc, "take", free=0.0, paid=[])     # 守る手段が無い
    rich = B.guard_step(tok, sc, "take", free=9000.0, paid=[])  # 守れたのに受けた
    assert poor["can_guard"] is False and poor["s"] == pytest.approx(0.0)
    assert rich["can_guard"] is True and rich["s"] < 0.0


def test_a_blocker_makes_the_guard_free_of_don():
    """ブロッカーはレストするだけ＝**ドンを使わない**ので予算に関係しない。"""
    tok, sc = _tok(opp_lead=9000, blocker=True), _sc(don=0)
    got = B.guard_step(tok, sc, "take", free=0.0, paid=[])
    assert got["can_guard"] is True


def test_a_huge_attack_is_taken_not_guarded():
    """**`c(x) > Θ` なら受けるのが正しい**——高すぎる攻撃に払うのは損。"""
    tok, sc = _tok(opp_lead=14000), _sc()
    got = B.guard_step(tok, sc, "take", free=20000.0, paid=[])
    assert got["theory_says"] == "take"
    assert got["s"] == pytest.approx(0.0)


def test_no_incoming_attack_is_not_a_decision():
    assert B.guard_step(_tok(opp_lead=1000), _sc(), "take", free=0.0, paid=[]) is None


def _seat(s_atk=0.0, s_grd=0.0, n_atk=5, n_grd=5, z=1.0, silent=0):
    return {"z": z, "s_atk": s_atk, "s_grd": s_grd, "n_atk": n_atk, "n_grd": n_grd,
            "n_silent": silent, "v0": [0.1]}


def test_the_pairing_is_my_loss_minus_the_opponents():
    """`ΔS > 0` は**自席の方が取りこぼしが少ない**＝理論に近い打ち方をした、の向き。"""
    per = {(1, 0): _seat(s_atk=-0.1, z=1.0), (1, 1): _seat(s_atk=-0.5, z=0.0)}
    got = B.pair_games(per)
    assert len(got) == 1 and got[0]["dS"] == pytest.approx(0.4)
    assert got[0]["z"] == 1.0


def test_the_per_row_version_divides_each_seat_by_its_own_count():
    """**手数で正規化した版**——先手は手番が 1 つ多いので生の和は席順を拾いうる。"""
    per = {(1, 0): _seat(s_atk=-1.0, n_atk=10, n_grd=0),
           (1, 1): _seat(s_atk=-1.0, n_atk=5, n_grd=0, z=0.0)}
    got = B.pair_games(per)[0]
    assert got["dS"] == pytest.approx(0.0)              # 生の和では差が無い
    assert got["dS_per_row"] == pytest.approx(-0.1 + 0.2)   # 1 手あたりでは差が出る
    assert got["dn"] == 5


def test_the_verdict_reads_the_normalised_slope():
    """**判定は正規化した版**で出す（生の和は手数の差に汚染されうる）。"""
    rng = np.random.default_rng(0)
    pairs = []
    for i in range(200):
        d = float(rng.normal())
        pairs.append({"seed": i, "z": 1.0 if d > 0 else 0.0, "dS": d, "dS_per_row": d,
                      "dS_atk": d, "dS_grd": d, "n": 10, "dn": 0, "silent": 0, "v0": 0.1})
    assert B.summarise(pairs, reps=50)["verdict"] == "bridge_holds"
    for p in pairs:                                     # 向きだけ反転させる
        p["dS_per_row"] = -p["dS_per_row"]
    assert B.summarise(pairs, reps=50)["verdict"] == "bridge_inverted"


def _rec():
    return {"seed": 1, "who": 0, "z": 1.0, "s_atk": 0.0, "s_grd": 0.0,
            "n_atk": 0, "n_grd": 0, "n_silent": 0, "v0": [], "band": {}}


def test_a_row_is_added_to_its_own_band():
    """**T28-b の要点**——行ごとに帯を決めてから足す。

    局ごとに足してから帯で切ると、**決着後の雑さが接戦帯に混ざる**（T28 で踏んだ）。
    """
    r = _rec()
    B._add(r, "close", -0.1, "atk")
    B._add(r, "decided", -0.9, "atk")
    assert r["band"]["close"]["s"] == pytest.approx(-0.1)
    assert r["band"]["decided"]["s"] == pytest.approx(-0.9)
    assert r["band"]["close"]["n"] == 1 and r["band"]["decided"]["n"] == 1
    assert r["s_atk"] == pytest.approx(-1.0)          # 全帯の合計も保つ


def test_the_two_sides_are_kept_apart_inside_a_band():
    """攻めと守りは帯の中でも分ける——**どちらの半分が効いているか**を見るため。"""
    r = _rec()
    B._add(r, "close", -0.2, "atk")
    B._add(r, "close", -0.5, "grd")
    b = r["band"]["close"]
    assert b["s_atk"] == pytest.approx(-0.2) and b["s_grd"] == pytest.approx(-0.5)
    assert b["n"] == 2


def test_pairing_within_a_band_needs_both_seats_present_there():
    """**片方の席にその帯の行が無い局は使えない**（差が定義できない）。"""
    a, b = _rec(), _rec()
    b["who"] = 1; b["z"] = 0.0
    B._add(a, "close", -0.1, "atk")
    per = {(1, 0): a, (1, 1): b}
    assert B.pair_by_band(per, "close") == []       # 席 1 に close の行が無い
    B._add(b, "close", -0.5, "atk")
    got = B.pair_by_band(per, "close")
    assert len(got) == 1 and got[0]["dS_per_row"] == pytest.approx(0.4)


def test_the_two_silent_modes_differ_in_the_denominator():
    """**P2 の暫定値**——無言の行を母数に入れるかどうかで 1 手あたりの値が変わる。

    `zero` は「取りこぼし 0」として母数に入れる（薄まる）・`exclude` は入れない。
    **実測でこの選択は結論を動かした**ので、テストで違いを固定する。
    """
    z = _rec()
    B._add(z, "close", -0.4, "atk")
    B._add(z, "close", 0.0, "atk")                  # zero が足す無言の行
    e = _rec()
    B._add(e, "close", -0.4, "atk")                 # exclude は足さない
    assert z["band"]["close"]["s"] / z["band"]["close"]["n"] == pytest.approx(-0.2)
    assert e["band"]["close"]["s"] / e["band"]["close"]["n"] == pytest.approx(-0.4)


def test_a_blocker_row_is_always_comfortably_affordable():
    """**T28-c**——ブロッカーはレストするだけでドンを使わないので、余裕は無限。"""
    got = B.guard_step(_tok(opp_lead=14000, blocker=True), _sc(don=0), "take",
                       free=0.0, paid=[])
    assert got["margin"] == float("inf") and got["comfortable"] is True


def test_the_margin_is_how_much_the_guard_overshoots_the_attack():
    """余裕＝**守る力 − 来る攻撃**。ここが小さい行を減点すると**貧しい席を罰する**。"""
    tok, sc = _tok(opp_lead=8000), _sc()          # 超過 3000
    got = B.guard_step(tok, sc, "take", free=5000.0, paid=[])
    assert got["margin"] == pytest.approx(2000.0)
    assert got["comfortable"] is True             # 既定の閾値 2000 にちょうど届く
    tight = B.guard_step(tok, sc, "take", free=3500.0, paid=[])
    assert tight["can_guard"] is True and tight["comfortable"] is False


def test_the_comfort_threshold_is_a_provisional_value_you_can_sweep():
    """**閾値は暫定値**なので振れる（§0.4 の規則 2＝感度を付ける）。"""
    tok, sc = _tok(opp_lead=8000), _sc()
    got = B.guard_step(tok, sc, "take", free=5000.0, paid=[], margin_comfort=6000.0)
    assert got["can_guard"] is True and got["comfortable"] is False


def test_a_row_that_could_not_be_guarded_is_never_comfortable():
    got = B.guard_step(_tok(opp_lead=9000), _sc(don=0), "take", free=0.0, paid=[])
    assert got["can_guard"] is False and got["comfortable"] is False


# ---- T40: 選んだ手の変化量 `g` と、その累積 `ΔG`（2026-09-15） ----

def test_the_gain_of_a_guard_row_is_minus_what_was_actually_paid():
    """`g` は**実際に払った費用の符号**——`s` と違って誰の責任かを問わない。"""
    tok, sc = _tok(), _sc(don=5)
    x = 2000.0
    got_g = B.guard_step(tok, sc, "guard", free=x + 1000.0, paid=[], theta=1.15, mu=0.0551, guard_g="paid")
    got_t = B.guard_step(tok, sc, "take", free=x + 1000.0, paid=[], theta=1.15, mu=0.0551, guard_g="paid")
    assert got_g["g"] == pytest.approx(-B.c_of(got_g["x"]) * 0.0551)
    assert got_t["g"] == pytest.approx(-1.15 * 0.0551)
    # 払えなかった行: `s` は 0（誤りでない）だが `g` は受けた損をそのまま持つ
    # （`x = 0` は 0 パワーでも守れるので、守れない行は相手リーダーを大きくして作る）
    poor = B.guard_step(_tok(opp_lead=8000), sc, "take", free=0.0, paid=[], theta=1.15, mu=0.0551, guard_g="paid")
    assert poor["can_guard"] is False
    assert poor["s"] == 0.0 and poor["g"] == pytest.approx(-1.15 * 0.0551)


def test_under_delta_the_guard_window_counts_only_the_gap_to_the_attackers_price():
    """**T62**: `g = 攻め手の価格 − 払った額`（価格＝`min(Θ·μ, c(x)·μ)`）。最安の応答なら 0・高い方を選べば差分が負・
    払えずに受けた行も差分（攻め手には払えるかが見えない）。`paid` と `delta` の両方を返し、既定はモジュール定数。"""
    x = 2000.0                                              # loose: c(2000) = 1.28 > Θ 1.15 → 受けるのが最安
    tok, sc = _tok(opp_lead=5000 + x), _sc(don=5)            # 来る攻撃の超過 x は相手リーダーと自リーダーの差
    th, mu = 1.15, 0.0551
    g = B.guard_step(tok, sc, "guard", free=x + 1000.0, paid=[], theta=th, mu=mu, guard_g="delta")
    t = B.guard_step(tok, sc, "take", free=x + 1000.0, paid=[], theta=th, mu=mu, guard_g="delta")
    assert t["g"] == pytest.approx(0.0, abs=1e-9)                                  # 最安どおり
    assert g["g"] == pytest.approx((th - B.c_of(x)) * mu, abs=1e-9) and g["g"] < 0  # 高い方を選んだ分だけ負
    assert g["g_paid"] == pytest.approx(-B.c_of(x) * mu) and g["g_delta"] == g["g"]
    assert t["price"] == pytest.approx(th * mu)
    poor = B.guard_step(_tok(opp_lead=8000), sc, "take", free=0.0, paid=[], theta=th, mu=mu, guard_g="delta")
    assert poor["can_guard"] is False and poor["s"] == 0.0
    assert poor["g"] == pytest.approx(min(th * mu, B.c_of(poor["x"]) * mu) - th * mu)   # 守れないが差分は載る
    before = B.GUARD_G_MODE
    try:
        assert B.set_guard_g_mode("paid") == "paid"
        assert B.guard_step(tok, sc, "guard", free=x + 1000.0, paid=[], theta=th, mu=mu)["g"] == pytest.approx(-B.c_of(x) * mu)
        with pytest.raises(ValueError):
            B.set_guard_g_mode("なにか")
    finally:
        B.set_guard_g_mode(before)
    assert B.GUARD_G_MODE == before


def test_gain_is_accumulated_beside_the_deviation_not_instead_of_it():
    rec = _rec()
    B._add(rec, "close", -0.02, "atk", g=0.05)
    B._add(rec, "close", -0.01, "grd", g=-0.06)
    assert rec["s_atk"] == pytest.approx(-0.02) and rec["g_atk"] == pytest.approx(0.05)
    assert rec["s_grd"] == pytest.approx(-0.01) and rec["g_grd"] == pytest.approx(-0.06)
    b = rec["band"]["close"]
    assert b["g"] == pytest.approx(-0.01) and b["g_atk"] == pytest.approx(0.05)
    assert b["g_grd"] == pytest.approx(-0.06)
    # `grdc`（余裕で払えた行の別勘定）は席の合計には二重に足さない
    B._add(rec, "close", -0.01, "grdc", g=-0.06)
    assert rec["g_grd"] == pytest.approx(-0.06) and b["g"] == pytest.approx(-0.01)
    assert b["g_grdc"] == pytest.approx(-0.06)


def test_the_pairing_of_gain_is_my_gain_minus_the_opponents_per_row():
    a = dict(_seat(n_atk=4, n_grd=4, z=1.0), g_atk=0.40, g_grd=-0.20)
    b = dict(_seat(n_atk=2, n_grd=2, z=1.0), g_atk=0.10, g_grd=-0.10)
    per = {(1, 0): a, (1, 1): b}
    p = B.pair_games(per)[0]
    assert p["dG"] == pytest.approx(0.20 - 0.0)
    assert p["dG_atk"] == pytest.approx(0.30) and p["dG_grd"] == pytest.approx(-0.10)
    assert p["dG_per_row"] == pytest.approx(0.20 / 8 - 0.0 / 4)
    # 旧い席（`g` を持たない）でも落ちない＝0 として扱う
    old = {(2, 0): _seat(z=1.0), (2, 1): _seat(z=0.0)}
    assert B.pair_games(old)[0]["dG"] == 0.0


def test_the_calibration_table_is_quantiles_with_their_raw_win_rate():
    """**較正表は当てはめない**——等分位ごとの実勝率をそのまま並べる。"""
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(200):
        x = float(rng.normal())
        z = 1.0 if rng.random() < 1.0 / (1.0 + np.exp(-3.0 * x)) else 0.0
        pairs.append({"dG_per_row": x, "z": z})
    cal = B.calibration(pairs, "dG_per_row", bins=5)
    assert len(cal["bins"]) == 5 and sum(r["n"] for r in cal["bins"]) == 200
    assert cal["bins"][0]["x_mean"] < cal["bins"][-1]["x_mean"]
    assert cal["monotone"] is True and cal["spread"] > 0.5
    assert B.calibration(pairs[:5], "dG_per_row", bins=5) is None      # 局数が足りなければ出さない


def test_summarise_reports_the_gain_family_and_its_own_verdict():
    rng = np.random.default_rng(1)
    pairs = []
    for i in range(120):
        g = float(rng.normal())
        z = 1.0 if rng.random() < 1.0 / (1.0 + np.exp(-4.0 * g)) else 0.0
        pairs.append({"seed": i, "z": z, "dS": 0.0, "dS_per_row": 0.0, "dS_atk": 0.0,
                      "dS_grd": 0.0, "dG": g * 8, "dG_per_row": g, "dG_atk": g, "dG_grd": 0.0,
                      "n": 8, "dn": 0, "silent": 0, "v0": 0.1})
    out = B.summarise(pairs, reps=50)
    assert out["gain_verdict"] == "bridge_holds"
    assert out["dG_per_row"]["auc"] > 0.7 and out["dG_per_row"]["slope"] > 0
    assert out["calibration"]["dG_per_row"]["monotone"] is True
    assert out["verdict"] == "undecided"          # `ΔS` は全部 0＝分散ゼロで判定できない


def test_the_bridge_weights_each_row_by_the_slope_of_the_race():
    """**T49**: `ΔG` の行は `κ = w(D)/w̄` で重み付く——同じ行の候補には共通なので順位は動かない。
    `flat` なら 1（それ以前の橋）。`D` の帯は端が閉じている。"""
    tok, sc = _tok(opp_lead=5000), _sc()
    sc[T.SC_MY_LIFE], sc[T.SC_OPP_LIFE], sc[T.SC_MY_HAND], sc[T.SC_OPP_HAND] = 3, 3, 4, 4
    assert T.clock_of_row(sc, tok, mode="flat")["kappa"] == 1.0
    k = T.clock_of_row(sc, tok, mode="clock")["kappa"]
    assert k == pytest.approx(T.w_of_d(T.clock_of_row(sc, tok)["d"]) / T.W_BAR)
    assert B._d_bin(-5) == "<-3" and B._d_bin(-2) == "-3..-1" and B._d_bin(0) == "-1..1"
    assert B._d_bin(2) == "1..3" and B._d_bin(9) == ">3"
    assert B._TO_W_MODE() in T.W_MODES


# ---- T58（2026-09-16・ユーザ決定「それでいきましょうか」）: 数える価格は `exercise`・決める価格は `option` ----

def test_the_ledger_rereads_the_played_move_under_exercise_and_leaves_the_choice_alone():
    """`g`（数える）は帳簿の規約で 1 回だけ読み直し、`s`（決める）が使う `FLOW_PRICING` は動かさない。
    読み直しの間だけ規約が替わり、抜けたら必ず元に戻る。"""
    import effect_value as EV
    assert EV.LEDGER_FLOW_PRICING == "exercise" and EV.FLOW_PRICING == "option"
    seen = []

    def score():
        seen.append(EV.FLOW_PRICING)
        return 0.25

    assert B.ledger_value(score, 0.75) == 0.25                     # 帳簿の規約（exercise）で読み直した値
    assert seen == ["exercise"] and EV.FLOW_PRICING == "option"    # 中だけ替わり・外は option のまま
    assert B.ledger_value(score, 0.75, mode="option") == 0.75      # 同じ規約なら呼び直さない
    assert seen == ["exercise"]
    assert B.ledger_value(lambda: None, 0.75) == 0.75              # 読み直しが None なら決める側の値に戻す


def test_the_ledger_convention_is_a_module_constant_with_a_switch():
    import effect_value as EV
    before = EV.LEDGER_FLOW_PRICING
    try:
        assert EV.set_ledger_flow_pricing("option") == "option"
        assert B.ledger_value(lambda: 0.25, 0.75) == 0.75          # 既定を option に戻せば読み直さない
        with pytest.raises(ValueError):
            EV.set_ledger_flow_pricing("なにか")
    finally:
        EV.set_ledger_flow_pricing(before)
    assert EV.LEDGER_FLOW_PRICING == before


def test_a_played_body_is_booked_when_it_starts_working():
    """**T84**（ユーザ決定 2026-09-18）: 出した体の価格の計上時点（`PLAY_BOOK_MODE`）。
    規則から**次の自席ターンに繰り延べるのは「登場したターンには何もできない体」だけ**——
    召喚酔いで殴れず（`battle.rs::declare_attack`）・アクティブなので殴られず（`attackable`）・
    ブロッカーでないのでブロックもできない体。**速攻**（今から殴れる）・**ブロッカー**（相手の次のターンから守れる）・
    **イベント／ステージ**（その場で解決）・**リーダー**・体を持たない札は `now` のまま。"""
    class _Cards:
        def __init__(self, d):
            self.d = d

        def info(self, cid):
            return self.d.get(cid)

    cards = _Cards({"plain": {"power": 5000}, "rush": {"power": 5000, "rush": True},
                    "blk": {"power": 5000, "blocker": True}, "ev": {"event": True, "power": 0},
                    "stg": {"stage": True, "power": 0}, "ld": {"leader": True, "power": 5000},
                    "noBody": {"power": 0}})
    assert B.play_starts_next_turn("plain", cards) is True
    for cid in ("rush", "blk", "ev", "stg", "ld", "noBody"):
        assert B.play_starts_next_turn(cid, cards) is False, cid
    assert B.play_starts_next_turn("missing", cards) is False      # 知らない札は繰り延べない
    assert B.play_starts_next_turn(None, cards) is False
    assert B.play_starts_next_turn("plain", None) is False         # カード表が無ければ繰り延べない
    before = B.PLAY_BOOK_MODE
    assert before == "next"                                        # **既定は規則どおりの計上時点**（ユーザ決定 2026-09-18）
    try:
        assert B.set_play_book_mode("now") == "now"                # 旧の計上時点にも戻せる（以前の数字と比べるとき）
        with pytest.raises(ValueError):
            B.set_play_book_mode("なにか")
        assert B.PLAY_BOOK_MODE == "now"
    finally:
        B.set_play_book_mode(before)
    assert B.PLAY_BOOK_MODE == "next"


def test_the_attach_is_counted_once_in_the_ledger():
    """**T85**: ドン付与の増分は**殴る行の価格に既に入っている**（記録の `slot_power` は自席のターンに付与ドンを
    載せる）ので、帳簿（`g`・`ΔG`）では**付与の行を 0** にして同じ移転を 1 回だけ数える（T62 と同じ型の直し）。
    **決める価格 `s` は増分のまま**（T58 の分離）。切替の既定は旧（`increment`）。"""
    assert B.ATTACH_LEDGER_MODE == "in_attack"      # **既定は 1 回だけ数える**（ユーザ決定 2026-09-18）
    assert B.move_family(["DON_BOX", "cid", [], [], None]) == "attach"      # 対象なし＝純粋な付与
    assert B.move_family(["DON_BOX", "cid", ["t"], [], None]) == "attack"   # 対象あり＝殴る手（こちらは 0 にしない）
    try:
        assert B.set_attach_ledger_mode("increment") == "increment"        # 旧の規約にも戻せる
        with pytest.raises(ValueError):
            B.set_attach_ledger_mode("なにか")
        assert B.ATTACH_LEDGER_MODE == "increment"
    finally:
        B.set_attach_ledger_mode("in_attack")
    assert B.ATTACH_LEDGER_MODE == "in_attack"


def test_the_guard_window_can_be_priced_against_every_attack_of_the_turn():
    """**T86**: 守りの窓の「攻め手の価格」は `max_attack`（旧・そのターン最大の攻撃 1 本）か
    **`all_attacks`**（そのターンに相手が打った攻撃の価格の和＝帳簿の攻めの行と同じ数）。
    旧は `actual`（`spent`＝窓の間に消えた札の総額＝**全部の攻撃への支払い**）と釣り合っておらず、
    **攻め手の行が既に数えた移転を守り側でもう一度数えていた**（記録: 攻撃のあったターンの 73%／67% が 2 本以上）。"""
    assert B.GUARD_PRICE_MODE == "max_attack"          # 既定は旧のまま（採否はユーザ判定）
    try:
        assert B.set_guard_price_mode("all_attacks") == "all_attacks"
        with pytest.raises(ValueError):
            B.set_guard_price_mode("なにか")
    finally:
        B.set_guard_price_mode("max_attack")
    # 切り出した確定処理（T86）: `g` は `price − actual`・帯と型に同じ値が入る
    kn, stats, rec = {}, {"grd_rows": 0, "grd_by_life": {}, "grd_comfortable": 0}, {}
    seen = []
    got = {"g": 0.25, "g_paid": -0.5, "g_delta": 0.25, "price": 0.75, "s": -0.1,
           "theory_says": "take", "can_guard": True, "comfortable": False}
    B._finish_guard(got, "take", 3.0, 1.0, "close", 2.0, 0, 4, rec, kn, stats,
                    lambda *a, **k: seen.append((a, k)))
    assert stats["grd_rows"] == 1 and stats["grd_by_life"]["3"]["n"] == 1
    assert stats["grd_by_life"]["3"]["g_delta"] == pytest.approx(0.25)
    assert kn[4]["g0"] == pytest.approx(0.25) and kn[4]["g_fam"]["guard"] == pytest.approx(0.25)
    assert seen and seen[0][1]["g"] == pytest.approx(0.5)        # `κ = 2` を掛けた値が帯に入る
    kn2, stats2 = {}, {"grd_rows": 0, "grd_by_life": {}, "grd_comfortable": 0}
    B._finish_guard(got, "take", 3.0, 1.0, "close", 1.0, 1, 5, rec, kn2, stats2, lambda *a, **k: None)
    assert kn2[5]["g0"] == pytest.approx(-0.25)                  # 席 1 は符号が反転する


def test_the_ledger_can_be_written_in_what_was_actually_lost():
    """**T87**（T86 の結論）: 帳簿を**実際に失われた額**で書く切替。`realised_harm` は
    `attack_response.parts` の相手ライフ・相手手札・相手の体の和＝**交点の橋が `F` を積むのに使う式と同じ**
    （`crossing_bridge.harm_of`）。ライフ 1 枚 ＋ 手札 1 枚なら `λ + μ` ちょうど。"""
    import numpy as np
    import theory_order as T
    sc, sc2 = np.zeros(70, np.float32), np.zeros(70, np.float32)
    sc[T.SC_OPP_LIFE], sc2[T.SC_OPP_LIFE] = 4, 3
    sc[T.SC_OPP_HAND], sc2[T.SC_OPP_HAND] = 5, 4
    tok, tok2 = np.zeros((22, 24), np.float32), np.zeros((22, 24), np.float32)
    assert B.realised_harm(sc, tok, sc2, tok2) == pytest.approx(T.LAM + T.MU)
    assert B.realised_harm(sc, tok, sc, tok) == pytest.approx(0.0)          # 何も失っていなければ 0
    import crossing_bridge as CB
    from attack_response import parts
    assert B.realised_harm(sc, tok, sc2, tok2) == pytest.approx(CB.harm_of(parts(sc, tok, sc2, tok2)))
    assert B.LEDGER_HARM_MODE == "realised"     # **既定は実現**（ユーザ決定 2026-09-18・`exercise` の規約どおり）
    try:
        assert B.set_ledger_harm_mode("price") == "price"                   # 旧の規約にも戻せる（比較用）
        with pytest.raises(ValueError):
            B.set_ledger_harm_mode("なにか")
    finally:
        B.set_ledger_harm_mode("realised")
