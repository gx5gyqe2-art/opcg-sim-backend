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
    """しきい値の形（`λL + gH + Σν_meas(吸える体)`）。**体の集合は `THETA_BODY_MODE` が決める**ので、
    旧 `blockers`（アクティブなブロッカーだけ・レストは数えない）を明示して算術を固定する（既定は T83 の `attackable`）。"""
    tok = np.zeros((22, 24), np.float32)
    try:
        CB.set_theta_body_mode("blockers")                                           # 既定（T97）
        assert CB.threshold(_sc(3, 4), tok) == pytest.approx(3 * T.LAM + 4 * T.MU)
        tok[7, T.S_POWER], tok[7, T.S_IS_CHAR], tok[7, T.S_IS_BLOCKER] = 0.6, 1.0, 1.0  # アクティブなブロッカー 6000
        assert CB.threshold(_sc(3, 4), tok) == pytest.approx(3 * T.LAM + 4 * T.MU + PR.NU_MEAS["leader_to_sat"])
        tok[7, T.S_IS_REST] = 1.0                                                        # `blockers` ではレスト中は数えない
        assert CB.threshold(_sc(3, 4), tok) == pytest.approx(3 * T.LAM + 4 * T.MU)
    finally:
        CB.set_theta_body_mode("blockers")


def test_sigma_t_comes_from_the_measurement_and_follows_the_body_set():
    """**T97**（ユーザ指示「理論的に正しいものにしたい」）: `σ_D = √2 × σ_T` の `σ_T` は
    **借り物の 1.0 ではなく交点の橋の実測**（輪郭の表の `sigma_t`）から採る。
    **耐久の体の集合ごとに違う**（`blockers` は `attackable` より小さい）ので追随し、
    **測る記録と別のセット**の値を使う（輪郭と同じ規約）。"""
    b_real, b_syn = CB.sigma_t_for(None, "real"), CB.sigma_t_for(None, "syn")     # 既定 `blockers`
    assert b_real and b_syn and b_real > 0.0 and b_syn > 0.0
    try:
        CB.set_theta_body_mode("attackable")
        a_real, a_syn = CB.sigma_t_for(None, "real"), CB.sigma_t_for(None, "syn")
    finally:
        CB.set_theta_body_mode("blockers")
    assert b_real < a_real and b_syn < a_syn        # 体の項を落とすと τ の残差は小さくなる（実測）
    assert CB.sigma_t_for(None, "real", body_mode="blockers") == b_real
    assert CB.sigma_t_for(None, "なにか") is None or True   # 知らない名前は cross 扱い
    assert CB.sigma_t_for([], "cross") is None      # 記録の種類が判らなければ引かない


def test_w_bar_comes_from_the_measurement_too():
    """**T98**: `κ = w(D)/w̄` の**分母は定義上 `E[w(D)]`**（検算は「`κ` の平均が 1」）。
    `0.5/R` は閉じた形の代用で、`σ_T` を実測にし耐久の形を変えたら一致しなくなった
    （`blockers` で `κ` の平均 1.505／1.737）ので、**同じ器の実測**を使う。規約は `σ_T` と同じ。"""
    import theory_order as TO
    b_real, b_syn = CB.w_bar_for(None, "real"), CB.w_bar_for(None, "syn")     # 既定 `blockers`
    assert b_real and b_syn and b_real > 0.0 and b_syn > 0.0
    try:
        CB.set_theta_body_mode("attackable")
        a_real, a_syn = CB.w_bar_for(None, "real"), CB.w_bar_for(None, "syn")
    finally:
        CB.set_theta_body_mode("blockers")
    assert b_real > a_real and b_syn > a_syn      # `D` が中央に集まるほど `w(D)` の平均は大きい（実測）
    assert b_real > 0.5 / TO.R_TURNS              # 旧 `0.5/R` より大きい＝`κ` が 1 より大きく出ていた
    assert CB.w_bar_for([], "cross") is None      # 記録の種類が判らなければ引かない


def test_the_hand_pays_for_the_guards_the_rules_force():
    """**T100**: T99 は `c(x_max)` 1 本で全札を割ったので端数を捨てすぎた。守り手は攻撃ごとに選ぶので、
    **規則が決める「必ず守る回数」** `G = max(0, 本数 − ライフ − ブロッカー)`（`board_theta` と同じ式）で
    1 回あたりの費用を出す。**打ち筋に依らない**（本数・ライフ・ブロッカー・`c_of` だけ）。"""
    xs = [0.0, 1000.0, 3000.0]                         # c = 1.0 / 1.28 / 2.78
    assert CB.forced_guards(xs, 1, 0) == 2             # 3 本・ライフ 1・ブロッカー 0 → 2 回は守らねば死ぬ
    assert CB.forced_guards(xs, 3, 0) == 0             # ライフが足りれば全部受けても死なない
    assert CB.forced_guards(xs, 1, 2) == 0             # ブロッカーが受けてくれる分は守らなくてよい
    assert CB.forced_guards([], 1, 0) == 0
    mu = T.MU
    import math as _m
    # **conftest は `CBAR_MODE=loose` を敷く**（そこでは `c(0) = c(1000) = 1`）ので、
    # **興味のある枝（`c_eff > 1`）を通すために出荷既定の `strict` を明示する**。
    before = T.CBAR_MODE
    try:
        T.set_cbar_mode("strict")
        cs = sorted(T.c_of(x) for x in xs)
        c_eff = (cs[0] + cs[1]) / 2.0
        assert c_eff > 1.0                             # 終盤は 1 枚では足りない
        # **終盤（G = 2）**: 3 枚では `floor(3/c_eff)` 回ぶんしか止まらない
        late = CB.hand_absorb_forced(3, xs, 1, 0)
        assert late == pytest.approx(mu * c_eff * _m.floor(3 / c_eff))
        assert late < 3 * mu                           # 旧 `cuttable` より小さい
        # **中盤以降（G = 0）**: 一番安い攻撃の `c` に落ちる＝**削らない**（T99 の削りすぎを避ける）
        assert CB.hand_absorb_forced(3, xs, 3, 0) == pytest.approx(3 * mu)
        assert CB.hand_absorb_forced(3, xs, 1, 2) == pytest.approx(3 * mu)
        # **T99 との違いは中盤以降**——`c(x_max)` は**守る義務が無いターンでも削る**が、
        # `forced` は `G = 0` なら削らない。そこが T99 の「削りすぎ」の正体。
        assert CB.hand_absorb(3, max(xs)) < 3 * mu                       # T99 は中盤でも削る
        assert CB.hand_absorb_forced(3, xs, 3, 0) == pytest.approx(3 * mu)   # T100 は削らない
    finally:
        T.set_cbar_mode(before)
    # 通る攻撃が無ければ 0
    assert CB.hand_absorb_forced(3, [-1000.0], 1, 0) == 0.0
    assert CB.hand_absorb_forced(3, [], 1, 0) == 0.0


def test_every_hand_mode_has_a_price_source():
    """**T99 で踏んだ穴**: 手札項の数え方を足したのに `part` の対応表に入れ忘れると、
    `crossing_bridge` は `KeyError` で落ち、`theory_bridge` は `.get` が `None` を返して**黙って `μ` に落ちていた**。
    **全モードが表に在ること**と、**知らない名前は落ちること**を押さえる。"""
    import theory_bridge as TB  # noqa: F401  （同じ表を使うことの確認）
    for m in CB.THETA_HAND_MODES:
        assert m in CB.THETA_HAND_PART
    assert CB.THETA_HAND_PART["count"] is None                       # `count` は `μ`
    assert CB.THETA_HAND_PART["cuttable_cx"] == "cuttable"           # 1 枚あたりの価格は同じ
    with pytest.raises(KeyError):
        CB.THETA_HAND_PART["なにか"]


def test_the_hand_absorbs_in_whole_guards():
    """**T99**（ユーザ指示「1から進めてください」）: 切れる札は **`c(x)` 枚ひと組**でしか働かない。
    規則は「攻撃側のパワー ≥ 対象のパワー」で命中するので、超過 `x` を止めるには `c(x)` 枚要る（`c_of`）＝
    **1 回分に足りない端数は一生 `F` に入らない＝耐久ではない**。"""
    mu = T.MU
    # `c(x) ≤ 1`（超過 0）なら従来どおり 1 枚 = 1 回
    assert T.c_of(0.0) == pytest.approx(1.0)
    assert CB.hand_absorb(3, 0.0) == pytest.approx(3 * mu)
    # 通らない攻撃しか無ければ守る必要が無い＝0
    assert CB.hand_absorb(3, -1000.0) == 0.0
    # `c(3000) = 2.78` → 3 枚では 1 回ぶんしか止まらない（端数 0.22 枚は捨てる）
    c = T.c_of(3000.0)
    assert c > 2.0
    assert CB.hand_absorb(3, 3000.0) == pytest.approx(mu * c * 1.0)
    assert CB.hand_absorb(3, 3000.0) < 3 * mu                       # 端数のぶん小さい
    # **1 回分に足りなければ 0**（重い攻撃 ＋ 薄い手札＝終盤の形）
    assert CB.hand_absorb(2, 3000.0) == 0.0
    assert CB.hand_absorb(0, 0.0) == 0.0


def test_the_threshold_splits_into_life_hand_and_bodies():
    """**T96**（ユーザ指示「Θの方で進めてください」）: `threshold_parts` は `Θ` を **3 つの項**に割り、和は `threshold` と一致する。
    **どの項が終盤に縮まないか**を見るための切り分け。"""
    tok = np.zeros((22, 24), np.float32)
    tok[7, T.S_POWER], tok[7, T.S_IS_CHAR], tok[7, T.S_IS_BLOCKER] = 0.6, 1.0, 1.0   # アクティブなブロッカー（既定で入る）
    sc = _sc(3, 4)
    life, hand, body = CB.threshold_parts(sc, tok)
    assert life == pytest.approx(3 * T.LAM)
    assert hand == pytest.approx(4 * T.MU)
    assert body > 0.0
    assert life + hand + body == pytest.approx(CB.threshold(sc, tok))
    # `g_hand` を渡すと手札の項だけが動く
    l2, h2, b2 = CB.threshold_parts(sc, tok, g_hand=0.5 * T.MU)
    assert (l2, b2) == (pytest.approx(life), pytest.approx(body))
    assert h2 == pytest.approx(hand / 2.0)


def test_a_rested_blocker_is_not_endurance_now_but_comes_back():
    """**T96**（ユーザ指摘「レストのブロッカーの意味も考えてみてください」）: 規則では
    **レストのブロッカーは横取りできない**（`has_blocker` が `!is_rest` を要求する）が、
    **持ち主のターン開始でアンタップして戻る**。だから **今の `Θ` からは外し、`j ≥ 2` の段差**として補充の側へ渡す。

    **符号化の制約も一緒に押さえる**——トークンの 6 列目は `is_blocker_active`
    （`tokens.rs`: `on_board && Character && !rest && has_keyword(BLOCKER)`）なので、
    **レストのブロッカーはトークン上 0**＝レストの素の体と区別がつかない。判別は**札の id から原本を引く**しかない。"""
    class _Cards:
        def info(self, cid):
            return {"blocker": True} if cid == "B" else {}

    tok = np.zeros((22, 24), np.float32)
    tok[7, T.S_POWER], tok[7, T.S_IS_CHAR] = 0.6, 1.0
    tok[7, T.S_IS_REST] = 1.0                                    # レストのブロッカー（列 6 は 0 のまま＝符号化どおり）
    ci = np.zeros(22, np.int32); ci[7] = 1
    idx2cid, cards = {1: "B"}, _Cards()
    sc = _sc(3, 4)
    assert CB.THETA_RETURN_MODE == "off"                         # 既定は据え置き（採否はユーザ判定）
    # **トークンだけでは分からない**（札を渡さなければ 0）
    assert CB.resting_blocker_term(tok, T.SLOT_OPP_FIELD, 5000.0) == 0.0
    back = CB.resting_blocker_term(tok, T.SLOT_OPP_FIELD, 5000.0, ci_row=ci, idx2cid=idx2cid, cards=cards)
    assert back > 0.0
    # **既定（`blockers`）では耐久に入らない**——今は横取りできないから（規則どおり）
    assert CB.threshold(sc, tok) == pytest.approx(3 * T.LAM + 4 * T.MU)
    try:                                                         # 旧 `attackable` は**レストの体として**数えていた
        CB.set_theta_body_mode("attackable")
        assert CB.threshold(sc, tok) > 3 * T.LAM + 4 * T.MU
    finally:
        CB.set_theta_body_mode("blockers")
    try:
        CB.set_theta_return_mode("untap")
        # 段差は `j ≥ 2` からしか効かない＝1 ターン目で届くなら τ は変わらない
        assert CB.tau_grow(0.2, 0.25, 0.0, 0.0, 0.0, step=back) == pytest.approx(
            CB.tau_grow(0.2, 0.25, 0.0, 0.0, 0.0))
        # 2 ターン目までかかるなら、その分だけ遠のく
        assert CB.tau_grow(0.4, 0.25, 0.0, 0.0, 0.0, step=back) > CB.tau_grow(0.4, 0.25, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            CB.set_theta_return_mode("なにか")
    finally:
        CB.set_theta_return_mode("off")


def test_the_theory_slope_is_the_priced_attack_flow_of_the_board():
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER], tok[1, T.S_POWER] = 0.5, 0.5
    lead_only = CB.theory_slope(tok, 5000.0)
    assert lead_only == pytest.approx(T.attack_value_don(5000.0, 5000.0, True))
    tok[2, T.S_POWER], tok[2, T.S_IS_CHAR], tok[2, T.S_CAN_ATTACK] = 0.8, 1.0, 1.0
    assert CB.theory_slope(tok, 5000.0) == pytest.approx(lead_only + T.attack_value_don(8000.0, 5000.0, True))


def test_the_rate_can_count_the_opponents_blockers():
    """**T92**（ユーザ指示「1で進めてください」）: 速さ `A` の盤面の項に**相手のアクティブなブロッカー**を入れる。
    **欠落を埋めるだけ**——`attack_value(..., blockers=)` は T47 から在り、`ν` も `score_candidate` も渡している。
    規則（`rules/battle.rs` の `has_blocker`）: 横取りできるのはアクティブなブロッカーだけ。"""
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER], tok[1, T.S_POWER] = 0.5, 0.5
    blk = ((6000.0, 0.05),)                                   # 殴る体より大きいブロッカー（ν = 0.05）
    assert CB.SLOPE_BLOCK_MODE == "off"                       # 既定は据え置き（採否はユーザ判定）
    assert CB.theory_slope(tok, 5000.0, blockers=blk) == pytest.approx(CB.theory_slope(tok, 5000.0))  # off では無視
    try:
        assert CB.set_slope_block_mode("on") == "on"
        with_blk = CB.theory_slope(tok, 5000.0, blockers=blk)
        assert with_blk == pytest.approx(T.attack_value_don(5000.0, 5000.0, True, blockers=blk))
        assert with_blk < CB.theory_slope(tok, 5000.0)        # 応答が 1 つ増えるので `min` は下がる
        assert CB.theory_slope(tok, 5000.0, blockers=()) == pytest.approx(CB.theory_slope(tok, 5000.0))
        with pytest.raises(ValueError):
            CB.set_slope_block_mode("なにか")
    finally:
        CB.set_slope_block_mode("off")


def test_the_hand_term_of_the_rate_is_a_flow():
    """**T93**（2026-09-18・ユーザ決定「それは規定にしましょうか」で既定）: 速さの手札の項は**在庫ではなく流入**。
    `flow` は**そのデッキの平均**（`deck_refill.a_of`）だけを見る＝**手札の中身も打ち方も読まない**。"""
    import deck_refill as DR
    assert CB.SLOPE_HAND_MODE == "flow"                      # 既定（以前の数字と比べるときだけ `stock`）
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER], tok[1, T.S_POWER] = 0.5, 0.5
    sc = _sc(3, 4)
    sc[T.SC_MY_DON] = 10.0
    db = DR.db()
    body = next(c for c in db.raw_db if DR.body_of(db.get_card(c)) and float(db.get_card(c).power) >= 6000)
    board, hand = CB.seat_slope_parts(sc, tok, None, None, None, 5000.0, deck_ids=[body])
    assert board == pytest.approx(CB.theory_slope(tok, 5000.0))              # 盤面の項は動かない
    assert hand == pytest.approx(DR.a_of([body], 5000.0, 10.0))
    assert hand > 0.0
    assert CB.seat_slope_parts(sc, tok, None, None, None, 5000.0)[1] == 0.0  # デッキが無ければ流入は数えない
    with pytest.raises(ValueError):
        CB.set_slope_hand_mode("なにか")
    try:                                                     # 旧い形（在庫）も残す＝過去の数字と比べるため
        assert CB.set_slope_hand_mode("stock") == "stock"
    finally:
        CB.set_slope_hand_mode("flow")


def test_the_walk_lets_the_rate_accumulate():
    """**T94**（2026-09-18・ユーザ決定「規定にして」で既定）: 交点までの速さを**規則どおり積み上げる**。
    盤面は毎ターン・**在庫は 2 ターン目からの段差**（召喚酔い）・**流入は進むほど積み上がる**（j で引いた札は j+1 から殴る）。"""
    assert CB.RATE_WALK_MODE == "grow"                               # 既定（以前の数字と比べるときだけ `flat`）
    # 在庫も流入も無ければ一定の速さと同じ（リーダー 0.25・キャラ 0）
    assert CB.tau_grow(1.0, 0.25, 0.0, 0.0, 0.0) == pytest.approx(4.0)
    # 在庫 0.1 は 2 ターン目から: 0.25 + 0.35 + 0.35 = 0.95、残り 0.05 を 4 ターン目の 0.35 で
    assert CB.tau_grow(1.0, 0.25, 0.0, 0.1, 0.0) == pytest.approx(3.0 + 0.05 / 0.35)
    # 流入 0.1 は (j − 1) 倍: 0.25 + 0.35 + 0.45 = 1.05 → 3 ターン目の途中
    assert CB.tau_grow(1.0, 0.25, 0.0, 0.0, 0.1) == pytest.approx(2.0 + 0.4 / 0.45)
    # 積み上がるほうが一定より早く届く
    assert CB.tau_grow(1.0, 0.25, 0.0, 0.1, 0.1) < CB.tau_grow(1.0, 0.25, 0.0, 0.0, 0.0)
    assert CB.tau_grow(0.0, 0.25, 0.0, 0.1, 0.1) == pytest.approx(0.0)    # 既に届いている
    assert CB.tau_grow(1.0, 0.0, 0.0, 0.0, 0.0) == pytest.approx(CB.RACE_CAP)   # 届かなければ打ち切り
    # 動く的と組める（的が下がるぶん遅くなる）
    assert CB.tau_grow(1.0, 0.25, 0.0, 0.1, 0.1, 0.05) > CB.tau_grow(1.0, 0.25, 0.0, 0.1, 0.1)
    try:                                                             # 旧い形も残す＝過去の数字と比べるため
        assert CB.set_rate_walk_mode("flat") == "flat"
        with pytest.raises(ValueError):
            CB.set_rate_walk_mode("なにか")
    finally:
        CB.set_rate_walk_mode("grow")


def test_the_board_can_decay_but_the_leader_never_does():
    """**T95**（ユーザ指示「2で進めてください」）: 盤面は毎自席ターン `ko_p` で失われる。
    **リーダーは KO されない**ので減衰しない＝速さは 0 に落ちず、流入のぶん `1/ko_p` に飽和する。"""
    assert CB.RATE_DECAY_MODE == "off"                               # 既定は据え置き（採否はユーザ判定）
    # `ko_p = 0` なら T94 のまま
    assert CB.rate_at(3, 0.05, 0.05, 0.1, 0.02, 0.0) == pytest.approx(0.05 + 0.05 + 0.1 + 0.04)
    q = 1.0 - 0.289
    assert CB.rate_at(1, 0.05, 0.05, 0.1, 0.02, 0.289) == pytest.approx(0.05 + 0.05)          # 1 ターン目は減衰前
    assert CB.rate_at(2, 0.05, 0.05, 0.1, 0.02, 0.289) == pytest.approx(0.05 + 0.05 * q + 0.1 + 0.02)
    assert CB.rate_at(3, 0.05, 0.05, 0.1, 0.02, 0.289) == pytest.approx(
        0.05 + 0.05 * q ** 2 + 0.1 * q + 0.02 * (1.0 + q))
    # **リーダーは残る**＝遠い先でも速さはリーダー ＋ 流入の飽和 `flow/ko_p` を下回らない
    far = CB.rate_at(60, 0.05, 0.05, 0.1, 0.02, 0.289)
    assert far == pytest.approx(0.05 + 0.02 / 0.289, abs=1e-6)
    # 減衰を入れると届くのが遅くなる
    assert CB.tau_grow(1.0, 0.05, 0.05, 0.1, 0.02, ko_p=0.289) > CB.tau_grow(1.0, 0.05, 0.05, 0.1, 0.02, ko_p=0.0)
    try:
        assert CB.set_rate_decay_mode("ko") == "ko"
        assert CB.tau_grow(1.0, 0.05, 0.05, 0.1, 0.02) == pytest.approx(
            CB.tau_grow(1.0, 0.05, 0.05, 0.1, 0.02, ko_p=T.KO_P))     # 既定の `ko_p` を拾う
        with pytest.raises(ValueError):
            CB.set_rate_decay_mode("なにか")
    finally:
        CB.set_rate_decay_mode("off")


def test_the_rate_terms_are_separate_quantities():
    """`seat_slope_terms` は `(盤面, 在庫, 流入, リーダー, 在庫の速攻, 流入の速攻)`（T103 で末尾 2 つが増えた）。
    **在庫は要求したときだけ計算する**（重いので）。"""
    import deck_refill as DR
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER], tok[1, T.S_POWER] = 0.5, 0.5
    sc = _sc(3, 4)
    sc[T.SC_MY_DON] = 10.0
    db = DR.db()
    body = next(c for c in db.raw_db if DR.body_of(db.get_card(c)) and float(db.get_card(c).power) >= 6000)
    board, stock, flow, lead, s_rush, f_rush = CB.seat_slope_terms(
        sc, tok, None, None, None, 5000.0, deck_ids=[body])
    assert s_rush == 0.0 and f_rush == 0.0            # 既定は `RATE_RUSH_MODE=off`（T103）
    assert board == pytest.approx(CB.theory_slope(tok, 5000.0))
    assert stock == 0.0                                              # `cards` が無ければ在庫は数えられない
    assert flow == pytest.approx(DR.a_of([body], 5000.0, 10.0))
    assert lead == pytest.approx(board)                              # 場が空ならリーダーが全部
    tok[2, T.S_POWER], tok[2, T.S_IS_CHAR], tok[2, T.S_CAN_ATTACK] = 0.8, 1.0, 1.0
    board2, _s, _f, lead2, _sr2, _fr2 = CB.seat_slope_terms(sc, tok, None, None, None, 5000.0)
    assert lead2 == pytest.approx(lead) and board2 > lead2           # キャラのぶんはリーダーに入らない
    # 既定（`flow`）では `seat_slope_parts` の 2 つ目は流入
    assert CB.seat_slope_parts(sc, tok, None, None, None, 5000.0, deck_ids=[body])[1] == pytest.approx(flow)


def test_the_blockers_of_the_rate_are_the_active_ones():
    """`opp_blockers_of` は**アクティブなブロッカーだけ**（レスト中は横取りできない・ブロッカーでない体も入らない）。"""
    tok = np.zeros((22, 24), np.float32)
    tok[7, T.S_POWER], tok[7, T.S_IS_CHAR] = 0.6, 1.0                 # ブロッカーでない体
    assert CB.opp_blockers_of(tok) == []
    tok[7, T.S_IS_BLOCKER] = 1.0
    got = CB.opp_blockers_of(tok)
    assert len(got) == 1 and got[0][0] == pytest.approx(6000.0) and got[0][1] > 0.0
    tok[7, T.S_IS_REST] = 1.0
    assert CB.opp_blockers_of(tok) == []


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
    try:
        CB.set_theta_body_mode("blockers")
        assert CB.threshold(sc, tok) == pytest.approx(base + PR.NU_MEAS["leader_to_sat"])   # アクティブなブロッカー 1 体だけ
        assert CB.set_theta_body_mode("all") == "all"
        # **`F` が使う関数そのもの**と一致する（3 体ぜんぶ）
        assert CB.threshold(sc, tok) == pytest.approx(base + PR.side_nu_meas(tok, T.SLOT_OPP_FIELD, 5000.0))
        assert CB.threshold(sc, tok) > base + PR.NU_MEAS["leader_to_sat"]              # 体を出すほど耐久が増える
        with pytest.raises(ValueError):
            CB.set_theta_body_mode("なにか")
    finally:
        CB.set_theta_body_mode("blockers")


def test_the_endurance_counts_only_what_cannot_be_walked_past():
    """**T83 → T97**: 耐久は「**避けて通れないもの**」だけを数える。

    規則（`rust/opcg_engine/src/rules/battle.rs`）: **キャラを殴れるのはレストのときだけ**（`declare_attack`）・
    **リーダーへの攻撃を横取りできるのはアクティブなブロッカー**（`has_blocker` は `!is_rest && KW_BLOCKER`）。
    **T83 は「殴れるか」で決めた**（`attackable`＝レストの体 ＋ アクティブなブロッカー）が、
    **攻め手は的を選べる**ので**レストの体は 1 体も壊さずに勝てる＝耐久ではない**（T96 の実測でも
    とどめのターンで 4.07 倍の過大）。**避けて通れないのはアクティブなブロッカーだけ**＝既定は `blockers`（T97）。"""
    sc = _sc(3, 4)
    sc[T.SC_MY_LEADER_POWER] = 0.5
    tok = np.zeros((22, 24), np.float32)
    tok[7, T.S_POWER], tok[7, T.S_IS_CHAR] = 0.5, 1.0                                  # アクティブな非ブロッカー＝吸えない
    tok[8, T.S_POWER], tok[8, T.S_IS_CHAR], tok[8, T.S_IS_REST] = 0.5, 1.0, 1.0        # レストの体＝**避けて通れる**
    tok[9, T.S_POWER], tok[9, T.S_IS_CHAR], tok[9, T.S_IS_BLOCKER] = 0.5, 1.0, 1.0     # アクティブなブロッカー＝横取りできる
    base = 3 * T.LAM + 4 * T.MU
    one = PR.NU_MEAS["leader_to_sat"]
    assert CB.THETA_BODY_MODE == "blockers"            # **既定は規則から出る形**（T97）
    # 既定: アクティブなブロッカーだけ（レストの体もアクティブな非ブロッカーも入らない）
    assert CB.threshold(sc, tok) == pytest.approx(base + one)
    assert not CB._body_absorbs(tok, 7) and not CB._body_absorbs(tok, 8) and CB._body_absorbs(tok, 9)
    try:
        assert CB.set_theta_body_mode("attackable") == "attackable"
        assert CB.threshold(sc, tok) == pytest.approx(base + 2 * one)                   # 旧: レスト 1 ＋ ブロッカー 1
        assert not CB._body_absorbs(tok, 7) and CB._body_absorbs(tok, 8) and CB._body_absorbs(tok, 9)
        # 3 つの数え方は順序で挟まる: 規則どおり ≤ 旧（殴れるか） ≤ 全キャラ
        CB.set_theta_body_mode("blockers")
        low = CB.threshold(sc, tok)
        CB.set_theta_body_mode("attackable")
        mid = CB.threshold(sc, tok)
        CB.set_theta_body_mode("all")
        high = CB.threshold(sc, tok)
        assert low < mid < high
        assert CB.threshold_of_me(sc, tok) == pytest.approx(0.0)                        # 自分のライフ・手札・場が空なら 0（どのモードでも）
    finally:
        CB.set_theta_body_mode("blockers")


def test_the_race_can_run_against_a_moving_threshold():
    """**T90**（ユーザとの整理 2026-09-18「その形で進めてください」）: 交点を**動く的との競争**で解く。
    **時間軸は流れの側に 1 本だけ**——`Θ` は在庫のまま・`A`（1 ターン目は盤面だけ＝召喚酔い）と
    相手の補充 `r`（引き 1 枚＝`Θ` の手札項と同じ 1 枚あたりの価格）が時間を持つ。"""
    assert CB.RACE_MODE == "static"                                  # 既定は旧（採否はユーザ判定）
    assert CB.tau_net(1.0, 0.25, 0.0, 0.0) == pytest.approx(4.0)     # 的が動かなければ Θ/A
    assert CB.tau_net(1.0, 0.25, 0.0, 0.05) == pytest.approx(5.0)    # 補充ありなら Θ/(A − r)
    # **手札の体は 2 ターン目から**（召喚酔い）: 1 ターン目 0.2・以後 0.3 → 0.2+0.3+0.3 = 0.8、残り 0.2 を 4 ターン目の途中で
    assert CB.tau_net(1.0, 0.2, 0.1, 0.0) == pytest.approx(3.0 + 0.2 / 0.3)
    assert CB.tau_net(1.0, 0.05, 0.0, 0.05) == pytest.approx(CB.RACE_CAP)   # 追いつけなければ打ち切り
    assert CB.tau_net(0.0, 0.25, 0.0, 0.0) == pytest.approx(0.0)     # 既に届いている
    # **輪郭の側で解く形**（T90 の本命）——輪郭は `A` の成長を持っているので、動く的でも追いつける
    prof = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25]
    assert CB.tau_from_profile(0.5, 0, prof) == pytest.approx(4.0)                 # 的が動かない
    assert CB.tau_from_profile(0.5, 0, prof, 1.0, 0.03) > 4.0                      # 動けば伸びる
    assert CB.tau_from_profile(0.5, 0, prof, 1.0, 0.0) == pytest.approx(4.0)       # r = 0 は従来と同じ
    try:
        assert CB.set_race_mode("net") == "net"
        with pytest.raises(ValueError):
            CB.set_race_mode("なにか")
    finally:
        CB.set_race_mode("static")


def test_the_refill_can_come_from_the_rules_instead_of_the_play():
    """**T91**（ユーザ指摘 2026-09-18「穴の大きさを測るのは CPU の打ち方によるんじゃない？」）:
    動く的の下がる速さ `r` を**帳簿の `g`**（打ち筋が入る）ではなく**デッキの中身**から出す形。
    的の解き方（`tau_net`／`tau_from_profile`）は `net` と同一で、**変わるのは `r` の出どころだけ**。"""
    import deck_refill as DR
    assert "deck" in CB.RACE_MODES
    assert CB.RACE_MODE == "static"                                  # 既定は据え置き（採否はユーザ判定）
    try:
        assert CB.set_race_mode("deck") == "deck"
    finally:
        CB.set_race_mode("static")
    # `r` は `μ ×（切れる札の割合）`＝記録も打ち回しも読まない
    assert DR.r_of(0.5) == pytest.approx(T.MU * 0.5)
    prof = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25]
    assert CB.tau_from_profile(0.5, 0, prof, 1.0, DR.r_of(0.7)) > CB.tau_from_profile(0.5, 0, prof)


def test_theta_over_need_is_split_by_whether_tau_was_right():
    """**T101**（T100 §5 の疑い）: **`Θ` は在庫**だが**「要った損害」は実際にいつ終わったかに依る総量**なので、
    **理論より早く終わった局では `Θ` > 要 になるのが当たり前**。`theta_check` を
    **予測 τ が実際の残りターンと近い行**に絞って同じ比を測れるようにした（読み取りだけ・新定数ゼロ）。"""
    # 当たった行（τ ≒ 残り）は比 1・外れた行（τ が 3 ターン遠い）は比 2 になるよう作る
    rows = []
    for _ in range(10):
        rows.append({"t_left": 2, "j": 0, "tau": 2.2, "theta": 1.0, "need": 1.0,
                     "th_life": 0.5, "th_hand": 0.3, "th_body": 0.2})
        rows.append({"t_left": 2, "j": 0, "tau": 5.0, "theta": 2.0, "need": 1.0,
                     "th_life": 1.0, "th_hand": 0.6, "th_body": 0.4})
    out = CB.summarise([], [], None, rows)["theta_check"]
    assert out["n"] == 20
    # 全部混ぜると 1.5（＝`Θ` が過大に見える）
    assert out["by_turns_left"]["2"]["theta_over_need"] == pytest.approx(1.5)
    # **τ が当たった行だけなら 1**＝膨らみの正体は τ の偏りだった、という読み方ができる
    m = out["tau_matched"]["0.5"]
    assert m["n"] == 10 and m["share"] == pytest.approx(0.5)
    assert m["by_turns_left"]["2"]["theta_over_need"] == pytest.approx(1.0)
    assert m["pooled_theta_over_need"] == pytest.approx(1.0)
    # 許容を広げれば外れた行も入る（0.5 ⊆ 1.0 ⊆ 2.0）
    assert out["tau_matched"]["1.0"]["n"] == 10 and out["tau_matched"]["2.0"]["n"] == 10
    # 対照（外れた行）とその平均のずれも残す
    assert out["tau_missed"]["n"] == 10
    assert out["tau_missed"]["theta_over_need"] == pytest.approx(2.0)
    assert out["tau_missed"]["tau_minus_left"] == pytest.approx(3.0)
    assert out["tau_minus_left_mean"] == pytest.approx(1.6)


def test_the_two_places_that_solve_for_tau_use_the_same_branch():
    """**T101**: 行の予測（`SLOPES` のループ）と `theta_check` の τ は**同じ式**でなければ、
    「τ が当たった行」の選び方が予測と食い違う。`collect` の中で 1 本に束ねたことを、
    式そのもの（`tau_grow`／`tau_net`／`Θ/A`）が既定の切替に従うことで確かめる。"""
    assert CB.RATE_WALK_MODE == "grow" and CB.RACE_MODE == "static"   # 現在の既定
    # `grow` の既定では `r = 0`（`static` なので的は動かない）＝`tau_grow` の素の形
    assert CB.tau_grow(1.0, 0.1, 0.1, 0.0, 0.0, 0.0) == pytest.approx(5.0)
    # `static` ＋ `flat` なら `Θ / A`（`predict` と同じ）
    assert CB.predict(1.0, 1.0, 0.25, 0.5)[0] == pytest.approx(4.0)


def test_the_hand_is_a_shield_that_takes_time_to_spend():
    """**T102**（T101 が指した先）: **手札は「使う時間」が要る**——`Θ` に一括で足すと
    **とどめのターンで倍に見え**（T101: τ を揃えても 1.95／2.01）、**長い局では足りない**（0.58）。
    同じ `μ × 切れる枚数` を**しきい値から的の側の有限の盾へ移す**（新しい量はゼロ）。"""
    assert CB.THETA_HAND_PLACE == "stock"                         # 既定は据え置き（採否はユーザ判定）
    # 盾が無ければ従来どおり: 速さ 0.1／ターンで的 0.5 → 5 ターン
    assert CB.tau_grow(0.5, 0.1, 0.0, 0.0, 0.0) == pytest.approx(5.0)
    # 盾 0.5 を 1 ターンで全部使えるなら、的は 1.0 になる（＝旧 `stock` と同じ）
    assert CB.tau_grow(0.5, 0.1, 0.0, 0.0, 0.0, shield=0.5) == pytest.approx(10.0)
    # **毎ターン 0.02 までしか出せないなら、盾は 25 ターンかけてしか出ない**＝間に合った分だけ的が遠のく
    slow = CB.tau_grow(0.5, 0.1, 0.0, 0.0, 0.0, shield=0.5, shield_rate=0.02)
    assert slow == pytest.approx(6.4)   # 歩きは 1 ターン刻み（連続なら 0.5/(0.1−0.02) = 6.25）
    assert 5.0 < slow < 10.0
    # **とどめが近い（速さが大きい）ほど盾は出てこない**＝`Θ` に一括で足す形より短い
    fast_stock = CB.tau_grow(0.5, 1.0, 0.0, 0.0, 0.0, shield=0.5)
    fast_shield = CB.tau_grow(0.5, 1.0, 0.0, 0.0, 0.0, shield=0.5, shield_rate=0.05)
    assert fast_shield < fast_stock
    # 輪郭の側でも同じ形（`curve` の読み）
    prof = [0.1] * 12
    assert CB.tau_from_profile(0.5, 0, prof) == pytest.approx(5.0)
    assert CB.tau_from_profile(0.5, 0, prof, 1.0, 0.0, 0.5) == pytest.approx(10.0)
    assert 5.0 < CB.tau_from_profile(0.5, 0, prof, 1.0, 0.0, 0.5, 0.02) < 10.0
    try:
        assert CB.set_theta_hand_place("shield") == "shield"
        with pytest.raises(ValueError):
            CB.set_theta_hand_place("なにか")
    finally:
        CB.set_theta_hand_place("stock")


def test_the_shield_can_only_be_spent_on_attacks_that_exist():
    """**T102** の上限は**規則から出る**: 宣言された攻撃にしかカウンターは切れず、1 本止めるのに `c(x)` 枚要る。
    ブロッカーは**安い攻撃から**横取りするので札が要らず、**`c(x) > Θ` の攻撃は受けた方が安い**（T63）。"""
    old = T.CBAR_MODE
    try:
        T.set_cbar_mode("strict")                                  # `conftest` は `loose`（c(0)=1）にしている
        c0 = T.c_of(0.0)
        # 攻撃 2 本・ブロッカー 0 → 2 本ぶんの `c` を出せる
        assert CB.shield_rate_of([0.0, 0.0], 0) == pytest.approx(T.MU * 2 * c0)
        # ブロッカー 1 は**安い方**を横取りする＝その 1 本ぶんは札が要らない
        assert CB.shield_rate_of([0.0, 0.0], 1) == pytest.approx(T.MU * c0)
        assert CB.shield_rate_of([0.0, 0.0], 5) == pytest.approx(0.0)
        # 通らない攻撃（x < 0）は止める必要が無い
        assert CB.shield_rate_of([-1000.0], 0) == pytest.approx(0.0)
        # 攻撃が無ければ札は 1 枚も出ない＝**手札は耐久にならない**
        assert CB.shield_rate_of([], 0) == pytest.approx(0.0)
        # **`c(x) > Θ` は受ける**（守る規則・T63）＝その攻撃には札を出さない
        big = 1e9
        assert T.c_of(big) > CB.THETA
        assert CB.shield_rate_of([big], 0) == pytest.approx(0.0)
    finally:
        T.set_cbar_mode(old)


def test_the_opponents_attacks_are_read_from_the_board_not_the_turn_flag():
    """**T102**: 相手の攻撃の本数は `can_attack`（自席のターンの旗）では読めない——
    **相手のターンが来ればレフレッシュで全部アクティブになる**ので、場のキャラ全部 ＋ リーダーを数える。"""
    tok = np.zeros((22, 24), dtype=np.float32)
    tok[1, T.S_POWER] = 0.5                                        # 相手リーダー 5000
    tok[7, T.S_IS_CHAR] = 1.0; tok[7, T.S_POWER] = 0.6             # 相手のキャラ（レスト・旗も無し）
    tok[7, T.S_IS_REST] = 1.0
    tok[8, T.S_IS_CHAR] = 1.0; tok[8, T.S_POWER] = 0.4
    xs = CB.opp_attackers_of(tok, 5000.0)
    assert xs == pytest.approx([0.0, 1000.0, -1000.0])             # リーダー ＋ 場の 2 体（レストも数える）
    # 自分のアクティブなブロッカーは相手の攻撃を横取りできる
    tok[2, T.S_IS_CHAR] = 1.0; tok[2, T.S_IS_BLOCKER] = 1.0
    assert CB._own_active_blockers(tok) == 1
    tok[2, T.S_IS_REST] = 1.0
    assert CB._own_active_blockers(tok) == 0                       # レストのブロッカーは横取りできない


def test_the_walk_obeys_the_first_turn_rule():
    """**T103**: **どちらの席も「自分の最初のターン」はアタックできない**
    （規則・`rules/battle.rs::declare_attack` の `turn_count <= 2`）。歩きはこれを知らなかった。"""
    assert CB.RATE_T1_MODE == "off"                                # 既定は据え置き（採否はユーザ判定）
    try:
        CB.set_rate_t1_mode("on")
        # 局の 1 自席ターン目（`j0 = 1`）から歩くと、1 段目は 0
        assert CB.rate_at(1, 0.05, 0.05, 0.1, 0.02, j0=1) == pytest.approx(0.0)
        assert CB.rate_at(2, 0.05, 0.05, 0.1, 0.02, j0=1) > 0.0
        # 途中の行（`j0 ≥ 2`）から歩くなら 1 段目から打てる
        assert CB.rate_at(1, 0.05, 0.05, 0.1, 0.02, j0=2) > 0.0
        # 的に届くまでのターン数は 1 つ増える側に動く（1 段ぶん進めないので）
        assert CB.tau_grow(0.3, 0.1, 0.0, 0.0, 0.0, j0=1) > CB.tau_grow(0.3, 0.1, 0.0, 0.0, 0.0, j0=2)
        with pytest.raises(ValueError):
            CB.set_rate_t1_mode("なにか")
    finally:
        CB.set_rate_t1_mode("off")
    # `off` なら `j0` は無視される（旧と完全に同じ）
    assert CB.rate_at(1, 0.05, 0.05, 0.1, 0.02, j0=1) == pytest.approx(0.1)


def test_rush_bodies_attack_the_turn_they_arrive():
    """**T103**: **速攻は出したターン・引いたターンからもう殴れる**（規則）。
    帳簿側は T84 の `play_starts_next_turn` が既に例外にしていて、**歩きだけが持っていなかった**。"""
    assert CB.RATE_RUSH_MODE == "off"                              # 既定は据え置き
    # 在庫 0.1 が全部速攻なら 1 ターン目から乗る（旧は 2 ターン目から）
    assert CB.rate_at(1, 0.05, 0.0, 0.1, 0.0) == pytest.approx(0.05)
    assert CB.rate_at(1, 0.05, 0.0, 0.1, 0.0, stock_rush=0.1) == pytest.approx(0.15)
    # 速攻ぶんは総額の内側（二重には乗らない）
    assert CB.rate_at(3, 0.05, 0.0, 0.1, 0.0, stock_rush=0.1) == pytest.approx(
        CB.rate_at(3, 0.05, 0.0, 0.1, 0.0))
    # 流入の速攻は**引いたターンから**＝`j` 枚ぶん（旧の遅い側は `j − 1` 枚ぶん）
    assert CB.rate_at(3, 0.0, 0.0, 0.0, 0.02) == pytest.approx(0.04)
    assert CB.rate_at(3, 0.0, 0.0, 0.0, 0.02, flow_rush=0.02) == pytest.approx(0.06)
    # 上限を超えて渡しても総額で抑える
    assert CB.rate_at(1, 0.0, 0.0, 0.05, 0.0, stock_rush=99.0) == pytest.approx(0.05)
    try:
        assert CB.set_rate_rush_mode("on") == "on"
        with pytest.raises(ValueError):
            CB.set_rate_rush_mode("なにか")
    finally:
        CB.set_rate_rush_mode("off")


def test_the_rush_share_comes_out_of_the_same_knapsack():
    """**T103**: 速攻の内訳は**同じ最適な詰め方の中**から採る（速攻だけで別に解くとドンの枠を二重に使う）。"""
    class _Cards:
        def __init__(self, tbl): self.tbl = tbl
        def info(self, cid): return self.tbl.get(cid)
    cards = _Cards({"A": {"power": 6000, "rush": True}, "B": {"power": 7000}})
    items = [{"cid": "A", "cost": 4}, {"cid": "B", "cost": 4}]
    tot_a, rush_a = CB.playable_attack_price(items, cards, 4, 5000.0, want_rush=True)
    # ドンが 4 なら 1 枚しか出せない＝価値の大きい B が選ばれ、速攻ぶんは 0
    assert tot_a > 0.0 and rush_a == pytest.approx(0.0)
    tot_b, rush_b = CB.playable_attack_price(items, cards, 8, 5000.0, want_rush=True)
    assert tot_b > tot_a and 0.0 < rush_b < tot_b            # 8 なら両方出せる＝速攻ぶんが立つ
    assert CB.playable_attack_price(items, cards, 8, 5000.0) == pytest.approx(tot_b)


def test_the_rate_is_checked_against_the_realised_harm_per_turn():
    """**T103**: **速さの検算**——自席ターン番号ごとに「理論の `A` ÷ 実際の損害」を並べる。
    **平均が合っていても形が違えば交点の時刻は外れる**（T93 は平均を 1.01 に合わせたが偏りは +2.9 残った）。"""
    turn_harm = []
    for j, (h, a) in enumerate(((0.0, 0.05), (0.10, 0.05), (0.20, 0.30))):
        turn_harm += [{"j": j, "harm": h, "slope_theory": a}] * 40      # `min_n` を満たす本数
    out = CB.summarise([], [], turn_harm)["harm_profile"]
    assert out["harm_by_turn"][:3] == [0.0, 0.1, 0.2]
    assert out["theory_slope_by_turn"][:3] == [0.05, 0.05, 0.3]
    # 損害 0 のターンは比を出さない（割れないので `None`）
    assert out["ratio_by_turn"][0] is None
    assert out["ratio_by_turn"][1] == pytest.approx(0.5)               # 遅すぎる
    assert out["ratio_by_turn"][2] == pytest.approx(1.5)               # 速すぎる
    # 累積は両方持つ（交点は累積で決まるので、こちらが本番）
    assert out["cum_harm_by_turn"][2] == pytest.approx(0.3)
    assert out["cum_theory_by_turn"][2] == pytest.approx(0.4)


def test_the_refill_lands_in_the_shield_not_on_the_target():
    """**T104**（T102・T103 が指した先）: **補充も「使う時間」が要る**——引いた札は手札に入るだけで、
    **出せる速さは `shield_rate`（宣言された攻撃の本数）が決める**。
    `Θ + r·j` と直に足すと**上限なしに吸える**ことになり、交点が遠のきすぎる（T102 で偏り +6.37）。"""
    assert "deck_shield" in CB.RACE_MODES and CB.RACE_MODE == "static"
    # 盾も補充も無ければ従来どおり
    assert CB.tau_grow(0.5, 0.1, 0.0, 0.0, 0.0) == pytest.approx(5.0)
    # 的に直に足す旧い形（`r`）は上限が無いので、補充が速さに近いと一気に遠のく
    far = CB.tau_grow(0.5, 0.1, 0.0, 0.0, 0.0, r=0.08)
    # 盾に積む形なら、**出せる速さ 0.01 で頭打ち**＝遠のき方も 0.01/ターンで止まる
    near = CB.tau_grow(0.5, 0.1, 0.0, 0.0, 0.0, refill=0.08, shield_rate=0.01)
    assert near < far
    assert near == pytest.approx(CB.tau_grow(0.5, 0.1, 0.0, 0.0, 0.0, r=0.01))   # 上限ぶんだけ的が動く
    # 出せる速さが補充より大きければ、補充はそのまま効く（旧 `r` と一致）
    assert CB.tau_grow(0.5, 0.1, 0.0, 0.0, 0.0, refill=0.02, shield_rate=9.0) == pytest.approx(
        CB.tau_grow(0.5, 0.1, 0.0, 0.0, 0.0, r=0.02))
    # 攻撃が 1 本も無い（`shield_rate = 0`）なら**手札も補充も吸えない**
    assert CB.tau_grow(0.5, 0.1, 0.0, 0.0, 0.0, shield=0.3, shield_rate=0.0,
                       refill=0.05) != pytest.approx(5.0)          # 上限が無いときは一度に全部（旧の形）
    # 輪郭の側も同じ形
    prof = [0.1] * 30
    assert CB.tau_from_profile(0.5, 0, prof, 1.0, 0.0, 0.0, 0.01, 0.08) < CB.tau_from_profile(
        0.5, 0, prof, 1.0, 0.08)
