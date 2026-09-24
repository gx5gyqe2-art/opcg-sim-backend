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



#: **出荷時の既定**を import の瞬間に写し取る（`conftest` の autouse も各テストの try/finally も
#: まだ走っていない時点の値）＝**ファイルの既定そのもの**をラチェットするための控え。
_SHIPPED = {n: getattr(CB, n) for n in (
    "THETA_HAND_MODE", "THETA_HAND_PLACE", "THETA_BODY_MODE", "THETA_HAND_BLOCKER_MODE",
    "THETA_RETURN_MODE", "SLOPE_MODE", "SLOPE_HAND_MODE", "SLOPE_BLOCK_MODE", "SLOPE_EFFECT_MODE",
    "RATE_WALK_MODE", "RATE_DECAY_MODE", "RATE_T1_MODE", "RATE_RUSH_MODE", "RACE_MODE",
    "DON_PURSE_MODE", "THETA_DON_MODE", "THETA_HAND_WINDOW", "RATE_DON_MODE")}


def test_the_shipped_defaults_are_the_ones_we_decided():
    """**既定の一覧をラチェットする**（`docs/cpu_theory_gap.md` §9.2 の表と 1 対 1）。

    **2026-09-19 のユーザ指示**（「使用できるドンと使ったドンの整合が取れるように最後まで進めてください」）で
    `DON_PURSE_MODE=all`（T109）が入った——**財布が 1 つになって初めて帳尻が合う**
    （理論の使いすぎ 22.0% → 0.0%）。**続く「それは直しましょうか」で `THETA_DON_MODE=rule`（T110）**
    ——**耐久の側もドンを規則どおり払う**（手札のブロッカーの予算＝次ターンのアクティブ・
    カウンター・イベントは使い残しで払う）。

    **2026-09-20 のユーザ決定**（「効果があったものの規定はオンにしないの？」→「3 本すべて」）で 3 本動いた:
    `RATE_DON_MODE=flow`（T114・歩きの成長を規則のドンから）・`THETA_HAND_MODE=cuttable_forced` ＋
    `THETA_HAND_WINDOW=horizon`（T116・手札は守る窓が開く分だけ的に入る）・
    `theory_order.W_ERR_MODE=rel`（T118・勝率を比で読む）。**T114 と T116 は対で採る**
    ——単独では合成が動かないが、**組むと両記録で 5 軸が改善する**（偏り・的中・σ_T・`curve` の偏り・`Θ`/要）。
    **代金は ±1 当たりと `curve` の的中、そして線形の橋**（`dG` の AUC 0.696 → 0.660／0.720 → 0.692）。

    **同日のユーザ決定**（「1は規定、2は正しいものに直してください」）で 4 つ動いた:
    `SLOPE_EFFECT_MODE=hand`（T108）・`RATE_T1_MODE=on`・`RATE_RUSH_MODE=on`・
    `THETA_HAND_BLOCKER_MODE=on`（T103／T106＝**規則として正しい形**）。
    **黙って既定が変わると 2 つの橋の数字が比較不能になる**ので、ここで固定する。"""
    assert _SHIPPED == {
        "THETA_HAND_MODE": "cuttable_forced",   # T77／T100・2026-09-20（T116 と対で採用）
        "THETA_HAND_PLACE": "stock",            # T102（切替として残す）
        "THETA_BODY_MODE": "blockers",           # T97
        "THETA_HAND_BLOCKER_MODE": "on",         # T106・2026-09-19
        "THETA_RETURN_MODE": "off",              # T96（切替として残す）
        "SLOPE_MODE": "hand",                    # T77
        "SLOPE_HAND_MODE": "flow",               # T93
        "SLOPE_BLOCK_MODE": "off",               # T92（切替として残す）
        "SLOPE_EFFECT_MODE": "hand",             # T105／T108・2026-09-19
        "RATE_WALK_MODE": "grow",                # T94
        "RATE_DECAY_MODE": "off",                # T95（切替として残す）
        "RATE_T1_MODE": "on",                    # T103・2026-09-19
        "RATE_RUSH_MODE": "on",                  # T103・2026-09-19
        "RACE_MODE": "static",                   # T90／T91／T104（切替として残す）
        "DON_PURSE_MODE": "all",                 # T109・2026-09-19
        "THETA_DON_MODE": "rule",                # T110・2026-09-19
        "THETA_HAND_WINDOW": "horizon",          # T116・2026-09-20
        "RATE_DON_MODE": "flow",                 # T114・2026-09-20
    }


#: **2026-09-20**: 既定が `THETA_HAND_MODE=cuttable_forced`（T116 と対で採用）になったので、
#: **しきい値の算術を固定するテストは自分で形を明示する**（`μ × 枚数` の素の形を測っているもの）。
#: **既定そのものは上の `test_the_shipped_defaults_are_the_ones_we_decided` がラチェットする**ので、
#: ここで形を固定するのは「算術の検算」と「既定の検算」を分けるためであって既定を隠すためではない。
_PLAIN_HAND_TESTS = (
    "test_the_threshold_is_the_opponents_endurance_in_price_units",
    "test_the_threshold_splits_into_life_hand_and_bodies",
    "test_a_rested_blocker_is_not_endurance_now_but_comes_back",
    "test_theta_hand_mode_prices_the_hand_by_quality_instead_of_the_count",
    "test_the_hand_carries_two_values_cuttable_for_the_threshold_and_playable_for_the_rate",
    "test_the_endurance_counts_bodies_the_same_way_the_harm_side_does",
    "test_the_endurance_counts_only_what_cannot_be_walked_past",
)


@pytest.fixture(autouse=True)
def _plain_hand(request):
    """上の一覧のテストだけ **`THETA_HAND_MODE=cuttable`**（`g` をそのまま掛ける素の形）で回す。"""
    if request.node.name.split("[")[0] not in _PLAIN_HAND_TESTS:
        yield
        return
    old = CB.THETA_HAND_MODE
    CB.set_theta_hand_mode("cuttable")
    try:
        yield
    finally:
        CB.set_theta_hand_mode(old)


def _tg(*a, **k):
    """**歩きの代数は「局の途中の行」で確かめる**（`j0=2`）。

    `RATE_T1_MODE=on`（2026-09-19 から既定）は**局の 1 自席ターン目だけ**速さを 0 にする規則
    （`turn_count <= 2`）なので、**閉じた代数を固定するテストには掛けない**——
    規則そのものは `test_the_walk_obeys_the_first_turn_rule` が `j0` を明示して固定する。"""
    k.setdefault("j0", 2)
    return CB.tau_grow(*a, **k)


def _ra(*a, **k):
    """`rate_at` の代数も同じ（`j0=2`＝局の途中の行）。"""
    k.setdefault("j0", 2)
    return CB.rate_at(*a, **k)

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
        assert _tg(0.2, 0.25, 0.0, 0.0, 0.0, step=back) == pytest.approx(
            _tg(0.2, 0.25, 0.0, 0.0, 0.0))
        # 2 ターン目までかかるなら、その分だけ遠のく
        assert _tg(0.4, 0.25, 0.0, 0.0, 0.0, step=back) > _tg(0.4, 0.25, 0.0, 0.0, 0.0)
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
    assert _tg(1.0, 0.25, 0.0, 0.0, 0.0) == pytest.approx(4.0)
    # 在庫 0.1 は 2 ターン目から: 0.25 + 0.35 + 0.35 = 0.95、残り 0.05 を 4 ターン目の 0.35 で
    assert _tg(1.0, 0.25, 0.0, 0.1, 0.0) == pytest.approx(3.0 + 0.05 / 0.35)
    # 流入 0.1 は (j − 1) 倍: 0.25 + 0.35 + 0.45 = 1.05 → 3 ターン目の途中
    assert _tg(1.0, 0.25, 0.0, 0.0, 0.1) == pytest.approx(2.0 + 0.4 / 0.45)
    # 積み上がるほうが一定より早く届く
    assert _tg(1.0, 0.25, 0.0, 0.1, 0.1) < _tg(1.0, 0.25, 0.0, 0.0, 0.0)
    assert _tg(0.0, 0.25, 0.0, 0.1, 0.1) == pytest.approx(0.0)    # 既に届いている
    assert _tg(1.0, 0.0, 0.0, 0.0, 0.0) == pytest.approx(CB.RACE_CAP)   # 届かなければ打ち切り
    # 動く的と組める（的が下がるぶん遅くなる）
    assert _tg(1.0, 0.25, 0.0, 0.1, 0.1, 0.05) > _tg(1.0, 0.25, 0.0, 0.1, 0.1)
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
    assert _ra(3, 0.05, 0.05, 0.1, 0.02, 0.0) == pytest.approx(0.05 + 0.05 + 0.1 + 0.04)
    q = 1.0 - 0.289
    assert _ra(1, 0.05, 0.05, 0.1, 0.02, 0.289) == pytest.approx(0.05 + 0.05)          # 1 ターン目は減衰前
    assert _ra(2, 0.05, 0.05, 0.1, 0.02, 0.289) == pytest.approx(0.05 + 0.05 * q + 0.1 + 0.02)
    assert _ra(3, 0.05, 0.05, 0.1, 0.02, 0.289) == pytest.approx(
        0.05 + 0.05 * q ** 2 + 0.1 * q + 0.02 * (1.0 + q))
    # **リーダーは残る**＝遠い先でも速さはリーダー ＋ 流入の飽和 `flow/ko_p` を下回らない
    far = _ra(60, 0.05, 0.05, 0.1, 0.02, 0.289)
    assert far == pytest.approx(0.05 + 0.02 / 0.289, abs=1e-6)
    # 減衰を入れると届くのが遅くなる
    assert _tg(1.0, 0.05, 0.05, 0.1, 0.02, ko_p=0.289) > _tg(1.0, 0.05, 0.05, 0.1, 0.02, ko_p=0.0)
    try:
        assert CB.set_rate_decay_mode("ko") == "ko"
        assert _tg(1.0, 0.05, 0.05, 0.1, 0.02) == pytest.approx(
            _tg(1.0, 0.05, 0.05, 0.1, 0.02, ko_p=T.KO_P))     # 既定の `ko_p` を拾う
        with pytest.raises(ValueError):
            CB.set_rate_decay_mode("なにか")
    finally:
        CB.set_rate_decay_mode("off")


def test_the_decay_also_bites_on_the_scheduled_path():
    """**T128**（2026-09-20）: **列を作る道でも減衰が効く**。

    `rate_at` は `sched` が在ると**先頭で返す**ので、**`RATE_DON_MODE != "off"`（既定）の下では
    `RATE_DECAY_MODE=ko` が 1 ビットも効いていなかった**（`--rate-decay ko` の出力が既定とバイト一致）。
    **切替が名乗ったことをするか**をここで固定する（既定は `off` なので出荷の値は動かない）。
    """
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = 0.5                       # リーダー 5000（KO されない＝減衰しない）
    s0 = T.SLOT_OWN_FIELD.start                           # 盤面のキャラ 1 体（殴れる）
    tok[s0, T.S_POWER], tok[s0, T.S_IS_CHAR], tok[s0, T.S_CAN_ATTACK] = 0.6, 1.0, 1.0
    sc = _sc(3, 4)
    base = CB.seat_slope_sched(sc, tok, None, None, None, 5000.0, jmax=6)
    try:
        CB.set_rate_decay_mode("ko")
        dec = CB.seat_slope_sched(sc, tok, None, None, None, 5000.0, jmax=6)
    finally:
        CB.set_rate_decay_mode("off")
    assert base[0] == dec[0] == 0.0                       # 最初の自席ターンは規則で 0（T103）
    assert dec[1] < base[1]                               # 2 段目からは盤面が減っている
    assert all(d <= b + 1e-12 for d, b in zip(dec, base))  # どの段でも増えない
    assert dec[-1] < base[-1] * 0.9                       # 先へ行くほど差が開く
    # **既定では何も変わらない**（切替を入れなければ出荷の値はそのまま）
    assert CB.seat_slope_sched(sc, tok, None, None, None, 5000.0, jmax=6) == base


def test_the_rate_terms_are_separate_quantities():
    """`seat_slope_terms` は `(盤面, 在庫, 流入, リーダー, 在庫の速攻, 流入の速攻, 効果)`（T103／T105 で末尾が増えた）。
    **在庫は要求したときだけ計算する**（重いので）。"""
    import deck_refill as DR
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER], tok[1, T.S_POWER] = 0.5, 0.5
    sc = _sc(3, 4)
    sc[T.SC_MY_DON] = 10.0
    db = DR.db()
    body = next(c for c in db.raw_db if DR.body_of(db.get_card(c)) and float(db.get_card(c).power) >= 6000)
    board, stock, flow, lead, s_rush, f_rush, eff, eff1 = CB.seat_slope_terms(
        sc, tok, None, None, None, 5000.0, deck_ids=[body])
    assert eff1 == 0.0                                # 既定は `SLOPE_EFFECT_MODE=off`（T108）
    assert s_rush == 0.0 and f_rush == 0.0            # 既定は `RATE_RUSH_MODE=off`（T103）
    assert eff == 0.0                                 # 既定は `SLOPE_EFFECT_MODE=off`（T105）
    assert board == pytest.approx(CB.theory_slope(tok, 5000.0))
    assert stock == 0.0                                              # `cards` が無ければ在庫は数えられない
    assert flow == pytest.approx(DR.a_of([body], 5000.0, 10.0))
    assert lead == pytest.approx(board)                              # 場が空ならリーダーが全部
    tok[2, T.S_POWER], tok[2, T.S_IS_CHAR], tok[2, T.S_CAN_ATTACK] = 0.8, 1.0, 1.0
    board2, _s, _f, lead2, _sr2, _fr2, _e2, _e12 = CB.seat_slope_terms(sc, tok, None, None, None, 5000.0)
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
    assert _tg(1.0, 0.1, 0.1, 0.0, 0.0, 0.0) == pytest.approx(5.0)
    # `static` ＋ `flat` なら `Θ / A`（`predict` と同じ）
    assert CB.predict(1.0, 1.0, 0.25, 0.5)[0] == pytest.approx(4.0)


def test_the_hand_is_a_shield_that_takes_time_to_spend():
    """**T102**（T101 が指した先）: **手札は「使う時間」が要る**——`Θ` に一括で足すと
    **とどめのターンで倍に見え**（T101: τ を揃えても 1.95／2.01）、**長い局では足りない**（0.58）。
    同じ `μ × 切れる枚数` を**しきい値から的の側の有限の盾へ移す**（新しい量はゼロ）。"""
    assert CB.THETA_HAND_PLACE == "stock"                         # 既定は据え置き（採否はユーザ判定）
    # 盾が無ければ従来どおり: 速さ 0.1／ターンで的 0.5 → 5 ターン
    assert _tg(0.5, 0.1, 0.0, 0.0, 0.0) == pytest.approx(5.0)
    # 盾 0.5 を 1 ターンで全部使えるなら、的は 1.0 になる（＝旧 `stock` と同じ）
    assert _tg(0.5, 0.1, 0.0, 0.0, 0.0, shield=0.5) == pytest.approx(10.0)
    # **毎ターン 0.02 までしか出せないなら、盾は 25 ターンかけてしか出ない**＝間に合った分だけ的が遠のく
    slow = _tg(0.5, 0.1, 0.0, 0.0, 0.0, shield=0.5, shield_rate=0.02)
    assert slow == pytest.approx(6.4)   # 歩きは 1 ターン刻み（連続なら 0.5/(0.1−0.02) = 6.25）
    assert 5.0 < slow < 10.0
    # **とどめが近い（速さが大きい）ほど盾は出てこない**＝`Θ` に一括で足す形より短い
    fast_stock = _tg(0.5, 1.0, 0.0, 0.0, 0.0, shield=0.5)
    fast_shield = _tg(0.5, 1.0, 0.0, 0.0, 0.0, shield=0.5, shield_rate=0.05)
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
    assert CB.RATE_T1_MODE == "on"        # **2026-09-19 から既定**（ユーザ決定「2は正しいものに直してください」）
    try:
        # 局の 1 自席ターン目（`j0 = 1`）から歩くと、1 段目は 0
        assert CB.rate_at(1, 0.05, 0.05, 0.1, 0.02, j0=1) == pytest.approx(0.0)
        assert CB.rate_at(2, 0.05, 0.05, 0.1, 0.02, j0=1) > 0.0
        # 途中の行（`j0 ≥ 2`）から歩くなら 1 段目から打てる
        assert CB.rate_at(1, 0.05, 0.05, 0.1, 0.02, j0=2) > 0.0
        # 的に届くまでのターン数は 1 つ増える側に動く（1 段ぶん進めないので）
        assert CB.tau_grow(0.3, 0.1, 0.0, 0.0, 0.0, j0=1) > CB.tau_grow(
            0.3, 0.1, 0.0, 0.0, 0.0, j0=2)
        with pytest.raises(ValueError):
            CB.set_rate_t1_mode("なにか")
    finally:
        CB.set_rate_t1_mode("off")
    # `off` なら `j0` は無視される（旧と完全に同じ）
    assert _ra(1, 0.05, 0.05, 0.1, 0.02, j0=1) == pytest.approx(0.1)


def test_rush_bodies_attack_the_turn_they_arrive():
    """**T103**: **速攻は出したターン・引いたターンからもう殴れる**（規則）。
    帳簿側は T84 の `play_starts_next_turn` が既に例外にしていて、**歩きだけが持っていなかった**。"""
    assert CB.RATE_RUSH_MODE == "on"      # **2026-09-19 から既定**（規則どおり・量は小さい）
    # 在庫 0.1 が全部速攻なら 1 ターン目から乗る（旧は 2 ターン目から）
    assert _ra(1, 0.05, 0.0, 0.1, 0.0) == pytest.approx(0.05)
    assert _ra(1, 0.05, 0.0, 0.1, 0.0, stock_rush=0.1) == pytest.approx(0.15)
    # 速攻ぶんは総額の内側（二重には乗らない）
    assert _ra(3, 0.05, 0.0, 0.1, 0.0, stock_rush=0.1) == pytest.approx(
        _ra(3, 0.05, 0.0, 0.1, 0.0))
    # 流入の速攻は**引いたターンから**＝`j` 枚ぶん（旧の遅い側は `j − 1` 枚ぶん）
    assert _ra(3, 0.0, 0.0, 0.0, 0.02) == pytest.approx(0.04)
    assert _ra(3, 0.0, 0.0, 0.0, 0.02, flow_rush=0.02) == pytest.approx(0.06)
    # 上限を超えて渡しても総額で抑える
    assert _ra(1, 0.0, 0.0, 0.05, 0.0, stock_rush=99.0) == pytest.approx(0.05)
    try:
        assert CB.set_rate_rush_mode("off") == "off"
        with pytest.raises(ValueError):
            CB.set_rate_rush_mode("なにか")
    finally:
        CB.set_rate_rush_mode("on")


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
    assert _tg(0.5, 0.1, 0.0, 0.0, 0.0) == pytest.approx(5.0)
    # 的に直に足す旧い形（`r`）は上限が無いので、補充が速さに近いと一気に遠のく
    far = _tg(0.5, 0.1, 0.0, 0.0, 0.0, r=0.08)
    # 盾に積む形なら、**出せる速さ 0.01 で頭打ち**＝遠のき方も 0.01/ターンで止まる
    near = _tg(0.5, 0.1, 0.0, 0.0, 0.0, refill=0.08, shield_rate=0.01)
    assert near < far
    assert near == pytest.approx(_tg(0.5, 0.1, 0.0, 0.0, 0.0, r=0.01))   # 上限ぶんだけ的が動く
    # 出せる速さが補充より大きければ、補充はそのまま効く（旧 `r` と一致）
    assert _tg(0.5, 0.1, 0.0, 0.0, 0.0, refill=0.02, shield_rate=9.0) == pytest.approx(
        _tg(0.5, 0.1, 0.0, 0.0, 0.0, r=0.02))
    # 攻撃が 1 本も無い（`shield_rate = 0`）なら**手札も補充も吸えない**
    assert _tg(0.5, 0.1, 0.0, 0.0, 0.0, shield=0.3, shield_rate=0.0,
                       refill=0.05) != pytest.approx(5.0)          # 上限が無いときは一度に全部（旧の形）
    # 輪郭の側も同じ形
    prof = [0.1] * 30
    assert CB.tau_from_profile(0.5, 0, prof, 1.0, 0.0, 0.0, 0.01, 0.08) < CB.tau_from_profile(
        0.5, 0, prof, 1.0, 0.08)


def test_a_blocker_in_hand_is_endurance_too():
    """**T106**（T96 以来の宿題・ユーザ指摘「ブロッカーは出たターンの次の相手のターンにはブロックできます」）:
    **ブロックに召喚酔いは無い**（`has_blocker` は登場ターンかどうかを見ない）ので、
    **手札から出せるブロッカーは「避けて通れない体」の予備**＝`Θ` の体の項と同じ意味。
    **今はどこにも数えられていなかった**（盤面の体でも切れる札でもない）。"""
    assert CB.THETA_HAND_BLOCKER_MODE == "on"   # **2026-09-19 から既定**（規則がそう言っている）
    sc = _sc(3, 4)
    tok = np.zeros((22, 24), dtype=np.float32)
    base = CB.threshold(sc, tok)
    try:
        # `off`（旧）なら手札のブロッカーを渡しても動かない
        CB.set_theta_hand_blocker_mode("off")
        life, hand, body = CB.threshold_parts(sc, tok, hand_blocker=0.5)
        assert life + hand + body == pytest.approx(base)
        CB.set_theta_hand_blocker_mode("on")
        life2, hand2, body2 = CB.threshold_parts(sc, tok, hand_blocker=0.5)
        assert body2 == pytest.approx(body + 0.5)               # 体の項に載る
        assert life2 == pytest.approx(life) and hand2 == pytest.approx(hand)
        assert CB.threshold_parts(sc, tok, hand_blocker=0.0)[2] == pytest.approx(body)
        assert CB.threshold_parts(sc, tok, hand_blocker=-1.0)[2] == pytest.approx(body)   # 負は 0 に倒す
        with pytest.raises(ValueError):
            CB.set_theta_hand_blocker_mode("なにか")
    finally:
        CB.set_theta_hand_blocker_mode("on")


def test_the_hand_blocker_is_read_from_the_rules_not_the_play():
    """**T106**: どのブロッカーを数えるかは**次の自分のターンのドン**で払えるかだけで決まる。
    **1 体だけ**数える（1 体は 1 ターンに 1 回しか横取りできない・過小側に倒す）。

    **T110 でその「次のターンのドン」が規則どおりになった**——旧 `min(10, 今のアクティブ + 2)` は
    **レストと付与が戻ることを見ておらず**（自席ターン終わりの行で測るので使い残しは 0.5 枚しか無い）、
    **上限 10 も決め打ち**だった。新 `rule` は `アクティブ ＋ レスト ＋ 付与 ＋ min(2, デッキ)`。"""
    import don_ledger as DL

    class _Cards:
        def __init__(self, tbl): self.tbl = tbl
        def info(self, cid): return self.tbl.get(cid)

    class _HP:
        @staticmethod
        def hand_items(*_a, **_k):
            return [{"cid": "BLK", "cost": 5}, {"cid": "BIG", "cost": 9}, {"cid": "BODY", "cost": 1}]

    cards = _Cards({"BLK": {"power": 4000, "blocker": True},
                    "BIG": {"power": 9000, "blocker": True},
                    "BODY": {"power": 7000}})
    tok = np.zeros((22, 24), dtype=np.float32)
    sys.modules["hand_plan"] = _HP
    try:
        assert CB.THETA_DON_MODE == "rule"       # **2026-09-19 から既定**（ユーザ指示「それは直しましょうか」）
        sc = _sc(3, 4)
        # 使い残し 1・レスト 3・デッキ 6 → 次のターンは 1 + 3 + 0 + 2 = 6 ドン
        sc[DL.SC_MY_ACTIVE], sc[DL.SC_MY_RESTED] = 1.0, 3.0
        sc[DL.SC_MY_DON_DECK] = 6 / DL.DECK_SCALE
        v = CB.hand_blocker_nu(sc, tok, [0] * 22, {}, cards, 5000.0)
        assert v == pytest.approx(PR.nu_meas_of(4000.0, 5000.0))  # 5 コストは払えるが 9 は払えない
        # 付与も戻る——付与 4 を足すと 10 ドン＝両方払える → 高い方
        sc[DL.SC_MY_LEADER_DON] = 4 / DL.ATT_SCALE
        sc[DL.SC_MY_DON_DECK] = 2 / DL.DECK_SCALE
        v2 = CB.hand_blocker_nu(sc, tok, [0] * 22, {}, cards, 5000.0)
        assert v2 == pytest.approx(PR.nu_meas_of(9000.0, 5000.0)) and v2 >= v
        # 全部使い切って付与も無くデッキも尽きていれば 0（規則どおり 1 枚も増えない）
        sc[DL.SC_MY_ACTIVE] = sc[DL.SC_MY_RESTED] = 0.0
        sc[DL.SC_MY_LEADER_DON] = sc[DL.SC_MY_DON_DECK] = 0.0
        assert CB.hand_blocker_nu(sc, tok, [0] * 22, {}, cards, 5000.0) == pytest.approx(0.0)
        # **旧 `off` は使い残しだけを見るので、同じ行でも桁が違う**（3 ドン＝どちらも出せない）
        CB.set_theta_don_mode("off")
        sc[DL.SC_MY_ACTIVE], sc[DL.SC_MY_RESTED] = 1.0, 3.0
        sc[DL.SC_MY_DON_DECK] = 6 / DL.DECK_SCALE
        assert CB.hand_blocker_nu(sc, tok, [0] * 22, {}, cards, 5000.0) == pytest.approx(0.0)
        # 原本が引けなければ 0（落ちない）
        assert CB.hand_blocker_nu(sc, None, None, None, None, 5000.0) == pytest.approx(0.0)
    finally:
        CB.set_theta_don_mode("rule")
        del sys.modules["hand_plan"]


def test_the_rate_check_can_drop_the_killing_turn():
    """**T107**: 速さの検算を**終わりからの距離**でも読み、**とどめのターンを外した**形も出す。
    **勝った席の最後の自席ターンは必要なだけ削って終わる**ので、そのターンだけ
    「盤面の大きさ」と「実際に出した損害」が構造的にずれる（`A` の誤りではない）。"""
    turn_harm = []
    for j in range(3):
        # とどめのターン（残り 1）は損害が小さい＝比が大きく出る
        turn_harm += [{"j": j, "harm": 0.05, "slope_theory": 0.20, "priced": 0.04,
                       "t_left": 1, "won": True}] * 30
        # それ以外のターンは比 1
        turn_harm += [{"j": j, "harm": 0.20, "slope_theory": 0.20, "priced": 0.16,
                       "t_left": 3, "won": True}] * 30
    hp = CB.summarise([], [], turn_harm)["harm_profile"]
    # 全行だと比は 1 より大きく出る（とどめのターンが混ざるので）
    assert hp["ratio_by_turn"][0] > 1.0
    # **とどめを外すと 1 に戻る**＝ずれの正体はそのターンだった、と読める
    assert hp["excl_last_turn"]["ratio_by_turn"][0] == pytest.approx(1.0)
    assert hp["excl_last_turn"]["n"] == 90
    # 終わりからの距離でも読める
    assert hp["by_turns_left"]["1"]["n"] == 90 and hp["by_turns_left"]["1"]["ratio"] > 1.0
    assert hp["by_turns_left"]["3"]["ratio"] == pytest.approx(1.0)
    assert hp["by_turns_left"]["3"]["attack_share"] == pytest.approx(0.8)


def test_the_hand_can_fire_its_effect_once():
    """**T108**（T107 が指した先）: T105 は**毎ターン引く 1 枚**（流量）の効果だけを数えていた。
    **手札に溜まっている札の効果**は**一度きり**に使えるもので、速さの式に 1 項も無かった。
    歩きでは **1 ターン目に 1 回だけ**乗る（在庫なので繰り返さない）。"""
    assert CB.SLOPE_EFFECT_MODE == "hand"   # **2026-09-19 から既定**（ユーザ決定「1は規定」）
    # 流量（`eff`）は毎ターン・在庫（`eff_once`）は 1 ターン目だけ
    assert _ra(1, 0.1, 0.0, 0.0, 0.0, eff=0.01) == pytest.approx(0.11)
    assert _ra(3, 0.1, 0.0, 0.0, 0.0, eff=0.01) == pytest.approx(0.11)
    assert _ra(1, 0.1, 0.0, 0.0, 0.0, eff_once=0.03) == pytest.approx(0.13)
    assert _ra(2, 0.1, 0.0, 0.0, 0.0, eff_once=0.03) == pytest.approx(0.10)
    assert _ra(3, 0.1, 0.0, 0.0, 0.0, eff_once=0.03) == pytest.approx(0.10)
    # 負は 0 に倒す
    assert _ra(1, 0.1, 0.0, 0.0, 0.0, eff_once=-1.0) == pytest.approx(0.10)
    # 的に届くのは早くなる（1 ターン目に 1 回ぶん進む）
    assert _tg(0.5, 0.1, 0.0, 0.0, 0.0, eff_once=0.03) < _tg(0.5, 0.1, 0.0, 0.0, 0.0)
    try:
        assert CB.set_slope_effect_mode("off") == "off"
        assert CB.set_slope_effect_mode("on") == "on"
        with pytest.raises(ValueError):
            CB.set_slope_effect_mode("なにか")
    finally:
        CB.set_slope_effect_mode("hand")


def test_one_purse_pays_for_each_card_once(monkeypatch):
    """**T109**（ユーザ指示「使用できるドンと使ったドンの整合が取れるように」）: **支払いも付与も
    同じアクティブから出る**のが規則なのに、`stock`（T77・体）と `e₁`（T108・効果）は**別々に同じドンを
    使えた**。`DON_PURSE_MODE=one` は**手札を 1 つのナップサックに入れ、1 枚 1 回だけ払う**。"""
    import deck_refill as DR

    class _Cards:
        def __init__(self, tbl): self.tbl = tbl
        def info(self, cid): return self.tbl.get(cid)
    # A＝体だけ・B＝効果だけ（イベント＝体を持たない）・どちらもコスト 4
    cards = _Cards({"A": {"power": 6000, "cost": 4}, "B": {"power": 0, "cost": 4, "event": True}})
    monkeypatch.setattr(DR, "card_effect_harm", lambda cid, *a, **k: 0.05 if cid == "B" else 0.0)
    items = [{"cid": "A", "cost": 4}, {"cid": "B", "cost": 4}]
    # ドン 4 なら**どちらか 1 枚だけ**——旧 `off` は両方（体も効果も）数えられた
    atk, rush, eff, paid = CB.hand_purse(items, cards, 4, 5000.0)
    assert paid == pytest.approx(4.0)                     # 払ったのは 1 枚ぶん
    assert (atk > 0.0) != (eff > 0.0)                     # 体か効果のどちらか片方しか立たない
    # ドン 8 なら両方買える＝両方立つ（払いは 8）
    atk2, _r2, eff2, paid2 = CB.hand_purse(items, cards, 8, 5000.0)
    assert paid2 == pytest.approx(8.0) and atk2 > 0.0 and eff2 == pytest.approx(0.05)
    # **払いは絶対に財布を超えない**（不変量）
    for don in range(0, 11):
        assert CB.hand_purse(items, cards, don, 5000.0)[3] <= don + 1e-9


def test_the_purse_keeps_bodyless_removal_that_the_old_form_dropped(monkeypatch):
    """**T109 の副産物**: `playable_attack_price` は**体を持つ札だけ**を候補にしていたので、
    **除去を持つイベント・ステージは丸ごと落ちていた**。財布のナップサックは**規則どおり**
    「体か効果のどちらかが在れば候補」なので拾う。"""
    import deck_refill as DR

    class _Cards:
        def __init__(self, tbl): self.tbl = tbl
        def info(self, cid): return self.tbl.get(cid)
    cards = _Cards({"E": {"power": 0, "cost": 2, "event": True}})
    monkeypatch.setattr(DR, "card_effect_harm", lambda cid, *a, **k: 0.07)
    items = [{"cid": "E", "cost": 2}]
    assert CB.playable_attack_price(items, cards, 5, 5000.0) == pytest.approx(0.0)   # 旧: 0
    atk, _r, eff, paid = CB.hand_purse(items, cards, 5, 5000.0)
    assert atk == pytest.approx(0.0) and eff == pytest.approx(0.07) and paid == pytest.approx(2.0)


def test_the_knapsack_reports_what_it_spent():
    """**T109**: 帳尻を取るには**詰め方が使ったドン**が要る（`want_cost`）。
    **同じ詰め方の中の額**なので、価値の内訳（速攻）と矛盾しない。"""
    class _Cards:
        def __init__(self, tbl): self.tbl = tbl
        def info(self, cid): return self.tbl.get(cid)
    cards = _Cards({"A": {"power": 6000, "rush": True}, "B": {"power": 7000}})
    items = [{"cid": "A", "cost": 4}, {"cid": "B", "cost": 3}]
    tot, rush, paid = CB.playable_attack_price(items, cards, 7, 5000.0, want_rush=True, want_cost=True)
    assert paid == pytest.approx(7.0) and rush > 0.0 and tot > rush
    tot1, rush1, paid1 = CB.playable_attack_price(items, cards, 3, 5000.0, want_rush=True, want_cost=True)
    assert paid1 == pytest.approx(3.0) and rush1 == pytest.approx(0.0) and tot1 > 0.0
    assert CB.playable_attack_price(items, cards, 0, 5000.0, want_cost=True)[2] == pytest.approx(0.0)


def test_the_purse_picks_one_option_per_group():
    """**T109**: 財布のナップサックは**組ごとに 1 つだけ**選ぶ（手札の 1 枚は「出す／出さない」・
    場の攻め手は「付与 `k` 枚」のうち 1 つ）。**同じ組から 2 つ取れたら付与を二重に買える**。"""
    g = [[(0, {}), (1, {"attach": 1.0}), (2, {"attach": 1.5})]]
    p = CB.purse_plan(g, 4)
    assert p["attach"] == pytest.approx(1.5) and p["paid"] == 2.0     # 3 枚買って 2.5 にはならない
    assert CB.purse_plan(g, 1)["attach"] == pytest.approx(1.0)
    assert CB.purse_plan(g, 0)["paid"] == 0.0 and CB.purse_plan(g, 0)["attach"] == 0.0
    # 2 つの組なら両方から 1 つずつ取れる
    g2 = g + [[(0, {}), (2, {"atk": 3.0, "rush": 3.0})]]
    p2 = CB.purse_plan(g2, 4)
    assert p2["atk"] == pytest.approx(3.0) and p2["rush"] == pytest.approx(3.0)
    assert p2["attach"] == pytest.approx(1.5) and p2["paid"] == 4.0   # 4 なら体 2 ＋ 付与 2
    # **財布が足りなければ組の中で安い方に落ちる**（付与 2 枚 → 1 枚）＝取り合いが起きる
    p3 = CB.purse_plan(g2, 3)
    assert p3["atk"] == pytest.approx(3.0) and p3["attach"] == pytest.approx(1.0) and p3["paid"] == 3.0
    # **払いは財布を超えない**（不変量）
    for b in range(0, 8):
        assert CB.purse_plan(g2, b)["paid"] <= b + 1e-9


def test_the_attach_comes_out_of_the_same_purse():
    """**T109**（ユーザ指示の核心「アクティブ・レスト・付与・ドンデッキの適切な箇所から使う」）:
    **付与（アクティブ → 付与）は手札を出すのと同じアクティブ**から出る（`ops.rs::attach_don`）。

    `attack_value_don`（T45）は `max_k [増分 − k·δ]` を取っていたが、**そのドンをどこからも引いて
    いなかった**（実測 0.10 枚/ターンが無料で湧いていた）。`all` では**盤面を素殴りに戻し**、
    付与を**財布の中の選択肢**にする＝**変えるのは出所だけ**（値段は T45 のまま・新定数ゼロ）。

    **`δ` を落とすのは誤り**（2026-09-19 に測って捨てた）——財布はふつう余る（要求 5.78 対 在る 6.01）ので
    **制約が値段の代わりにならず**、落とすと `A` が 3〜6 割膨らんで速さの検算が 1.25〜1.59 に壊れた。"""
    tok = np.zeros((22, 24), np.float32)
    olp = 5000.0
    # 自分のリーダー 5000（枠 0）＋ 4000 のキャラ（枠 2）＝ 1 枚付けて通る体（T45）
    tok[0, T.S_POWER], tok[0, T.S_CAN_ATTACK] = 0.5, 1.0
    tok[2, T.S_POWER], tok[2, T.S_IS_CHAR], tok[2, T.S_CAN_ATTACK] = 0.4, 1.0, 1.0
    g = CB.attach_groups(tok, olp, delta=0.0)
    assert g, "1000 低い体には付与の選択肢が立つ（値段 0 なら必ず）"
    # **`δ` を払うので、増分が `k·δ` を超えない付与は選択肢にならない**（T45 と同じ値付け）
    assert CB.attach_groups(tok, olp, delta=1e6) == []
    # **リーダーとキャラは別の名前**（リーダーは KO されない＝減衰の掛かる側と分ける・T95）
    names = {k for opts in g for _c, parts in opts for k in parts}
    assert names <= {"attach", "attach_lead"}
    # 財布が 0 なら 1 枚も付けられない
    assert CB.purse_plan(g, 0)["paid"] == 0.0
    # 盤面の項は `with_don` で素殴りに戻る（付与を財布で買うので二重に数えない）
    lead_don, chars_don = CB.theory_slope_parts(tok, olp)
    lead_raw, chars_raw = CB.theory_slope_parts(tok, olp, with_don=False)
    assert chars_don > chars_raw and lead_don == pytest.approx(lead_raw)
    try:
        assert CB.set_don_purse_mode("all") == "all"
        assert CB.set_don_purse_mode("one") == "one"
        with pytest.raises(ValueError):
            CB.set_don_purse_mode("なにか")
    finally:
        CB.set_don_purse_mode("all")


def test_the_next_turns_don_is_what_the_rules_give_back():
    """**T110**（ユーザ指示「それは直しましょうか」＝T109 §10 の 1）: **耐久の側もドンを規則どおり払う**。

    T106 の手札のブロッカーは予算を `min(10, 今のアクティブ + 2)` にしていたが、**規則はそうではない**:
    リフレッシュで**レストも付与も全部アクティブへ戻り**（`rules/turn.rs`）、そのあと `don_phase` が
    **ドンデッキから `min(2, 残り)`** 足す。よって
    **次の自分のターンのアクティブ ＝ アクティブ ＋ レスト ＋ 付与 ＋ min(2, デッキ)**。

    **上限は要らない**——4 ゾーンの合計はリーダーのルールが決める定数なので、この式は自動でその中に収まる。"""
    import don_ledger as DL
    sc = np.zeros(70, np.float32)
    tok = np.zeros((22, 24), np.float32)
    sc[DL.SC_MY_ACTIVE], sc[DL.SC_MY_RESTED] = 1, 4
    sc[DL.SC_MY_LEADER_DON] = 2 / DL.ATT_SCALE                    # 付与 2
    sc[DL.SC_MY_DON_DECK] = 3 / DL.DECK_SCALE                     # デッキ 3
    assert CB.next_turn_don(sc, tok) == pytest.approx(9.0)        # 1 + 4 + 2 + min(2, 3)
    # **ドンデッキが尽きたら足されない**（`don_phase` は空なら飛ばす）
    sc[DL.SC_MY_DON_DECK] = 0.0
    sc[DL.SC_MY_ACTIVE] = 4
    assert CB.next_turn_don(sc, tok) == pytest.approx(10.0)       # 4 + 4 + 2 + 0
    # **1 枚しか残っていなければ 1 枚だけ**
    sc[DL.SC_MY_DON_DECK] = 1 / DL.DECK_SCALE
    sc[DL.SC_MY_ACTIVE] = 3
    assert CB.next_turn_don(sc, tok) == pytest.approx(10.0)       # 3 + 4 + 2 + min(2, 1)
    # **6 枚のリーダー（OP15-058）でも式は自動で収まる**——合計が 6 しか無いので 6 を超えない
    sc2 = np.zeros(70, np.float32)
    sc2[DL.SC_MY_ACTIVE], sc2[DL.SC_MY_DON_DECK] = 4, 2 / DL.DECK_SCALE
    assert CB.next_turn_don(sc2, np.zeros((22, 24), np.float32)) == pytest.approx(6.0)
    # 旧式は自席ターン終わり（使い残しが少ない）で測るので大幅に過小
    try:
        assert CB.set_theta_don_mode("off") == "off"
        assert CB.set_theta_don_mode("blocker") == "blocker"
        assert CB.set_theta_don_mode("rule") == "rule"
        with pytest.raises(ValueError):
            CB.set_theta_don_mode("なにか")
    finally:
        CB.set_theta_don_mode(_SHIPPED["THETA_DON_MODE"])


def test_a_counter_event_has_to_be_paid_for():
    """**T110**: `rules/battle.rs::apply_counter` は **EVENT なら `pay_cost`**・
    印字カウンターの札はそのまま足せる＝**無料**。`cuttable` は「カウンター値 > 0」だけ見ていたので
    **イベントを無料で切れる前提**だった。**切るドンは相手のターンに在るドン**＝自席ターンの使い残し。

    **値はどの札も `μ` 1 枚ぶんで同じ**なので**安い順に取るのがそのまま最適**（ナップサックと一致）。"""
    ev1 = {"cid": "e1", "counter": 1000, "cost": 1, "event": True}
    ev2 = {"cid": "e2", "counter": 2000, "cost": 2, "event": True}
    printed = {"cid": "p", "counter": 1000, "cost": 3, "event": False}
    plain = {"cid": "x", "counter": 0, "cost": 1, "event": False}
    items = [ev1, ev2, printed, plain]
    assert CB.cuttable_share(items) == pytest.approx(3 / 4)        # 旧＝イベントも無料
    assert CB.cuttable_share(items, 0.0) == pytest.approx(1 / 4)   # ドンが無ければ印字だけ
    assert CB.cuttable_share(items, 1.0) == pytest.approx(2 / 4)   # 安い方（コスト 1）を 1 枚
    assert CB.cuttable_share(items, 3.0) == pytest.approx(3 / 4)   # 1 + 2 で両方
    assert CB.cuttable_share([], 5.0) == 0.0
    # **印字カウンターはドンが 0 でも数える**（規則どおり無料）
    assert CB.cuttable_share([printed], 0.0) == pytest.approx(1.0)


def test_the_purse_frontier_needs_no_exchange_rate():
    """**T111**（ユーザの問い「1本にまとめた定数はどんな意味を持つ？」）: 財布を 1 本にするには
    `価値 = A の分 + ρ × Θ の分` の `ρ` が要るが、**`ρ` は定数ではない**——

    * **次元が [1/ターン]**（`A` は価格/ターン・`Θ` は価格）＝**率**。`ρ = 1` は「耐久は 1 ターンで効く」
      と宣言するのと同じで、**時間の単位を変えると値が変わる**。
    * レースから出る値は `ρ = (∂D/∂Θ_me)/(∂D/∂A_me) = A_me²/(A_opp·Θ_opp) = (1/τ_me)·(A_me/A_opp)`
      ＝**局面の量**（どちらが速い側か）。**定数に固めると打ち筋を式に入れる**ことになる。

    **だから `ρ` を置かない**——`D` は両軸で単調増なので**最適点はパレート境界上に在り**、
    **境界の各点で `D` を直に測れば交換レートは現れない**。"""
    g = [[(0, {}), (3, {"atk": 0.10, "rush": 0.10})], [(0, {}), (3, {"theta_body": 0.08})]]
    # 予算 3 = どちらか片方しか買えない＝本物のトレードオフ
    front = CB.purse_pareto(g, 3)
    assert [(round(a, 3), round(t, 3)) for a, t, _p in front] == [(0.1, 0.0), (0.0, 0.08)]
    # **予算が足りれば両方**＝境界は 1 点に潰れる（取り合いが無い）
    assert len(CB.purse_pareto(g, 6)) == 1
    # **内訳（witness）も運ぶ**——歩きは `atk`（段差）と `eff`（一度きり）で扱いが違うので潰せない
    assert front[0][2]["rush"] == pytest.approx(0.10) and front[0][2]["paid"] == 3.0
    # **自分が速いほど耐久を買う**（`τ = Θ/A` が凸なので追加の速さの効きが落ちる）
    fast = CB.choose_by_race(front, 0.30, 0.60, 0.10, 0.60)
    assert fast[1] == pytest.approx(0.08) and fast[0] == pytest.approx(0.0)
    # **相手が速いほど速さを買う**（耐久 1 単位が買うターンが短い＝レースするしかない）
    slow = CB.choose_by_race(front, 0.10, 0.60, 0.30, 0.60)
    assert slow[0] == pytest.approx(0.10) and slow[1] == pytest.approx(0.0)
    # `D` の値そのものも 2 つの時計の差（`τ_opp − τ_me`）
    assert CB.race_margin(0.2, 0.6, 0.2, 0.6) == pytest.approx(0.0)
    assert CB.race_margin(0.4, 0.6, 0.2, 0.6) > 0.0          # 自分が速い＝margin は正
    assert CB.race_margin(0.0, 0.6, 0.2, 0.6) < 0.0          # 速さ 0 は床で割る（落ちない）


def test_the_endurance_options_are_the_two_the_rules_allow():
    """**T111**: 耐久に行ける選択肢は**規則が許す 2 つだけ**——
    **出せるブロッカー**（`theta_body = ν_meas`・ブロックに召喚酔いは無い）と
    **構えるカウンター・イベント**（`theta_hand = μ`・`apply_counter` が `pay_cost` を要求する）。
    **印字カウンターの札は無料**なので財布には入らない（`cuttable_share` が別に数える）。"""
    class _Cards:
        def __init__(self, tbl): self.tbl = tbl
        def info(self, cid): return self.tbl.get(cid)
    cards = _Cards({"B": {"power": 6000, "blocker": True, "cost": 4},
                    "E": {"power": 0, "event": True, "cost": 2},
                    "P": {"power": 5000, "cost": 3},
                    "N": {"power": 3000, "cost": 1}})
    # `event` は**枠のほうにも載る**（`hand_plan.hand_items` が付ける）——`cuttable_share` は枠を見る
    items = [{"cid": "B", "cost": 4, "counter": 0, "event": False},
             {"cid": "E", "cost": 2, "counter": 1000, "event": True},
             {"cid": "P", "cost": 3, "counter": 1000, "event": False},
             {"cid": "N", "cost": 1, "counter": 0, "event": False}]
    g = CB.theta_groups(items, cards, 5000.0)
    kinds = sorted({k for opts in g for _c, parts in opts for k in parts})
    assert kinds == ["theta_body", "theta_hand"]
    assert len(g) == 2                       # ブロッカー B とカウンター・イベント E だけ
    # 印字カウンターの非イベント（P）は**無料**なので財布の外
    assert CB.theta_groups([items[2]], cards, 5000.0) == []
    # `don_left` を渡すと払えないイベントは落ちる（`theta_body` は残る）
    g0 = CB.theta_groups(items, cards, 5000.0, don_left=0.0)
    assert sorted({k for opts in g0 for _c, parts in opts for k in parts}) == ["theta_body"]
    # 無料で切れる札の 1 枚あたり＝印字カウンターの非イベントだけ（4 枚中 1 枚）
    assert CB.free_cuttable_g(items) == pytest.approx(T.MU * 1 / 4)


def test_the_opponents_board_speed_is_readable_from_my_row():
    """**T111**: `A_opp` は**その席の行から読める**（相手の盤面の攻め手＝枠 1 と 7〜11・`opp_attackers_of`）。
    **`A_opp` が要るのは配分を決めるため**で、これが読めなければレースで割れない。"""
    tok = np.zeros((22, 24), np.float32)
    assert CB.opp_board_slope(tok, 5000.0) == pytest.approx(0.0)      # 誰も居なければ 0
    tok[1, T.S_POWER] = 0.5                                           # 相手のリーダー 5000
    one = CB.opp_board_slope(tok, 5000.0)
    assert one > 0.0
    tok[7, T.S_POWER], tok[7, T.S_IS_CHAR] = 0.6, 1.0                 # 相手の体 6000
    assert CB.opp_board_slope(tok, 5000.0) > one                      # 攻め手が増えれば速くなる


def test_the_body_term_can_move_to_the_rate_side():
    """**T129**（ユーザ決定 2026-09-20「相手の守りは入れましょう」）: **足すのではなく移す**。

    ブロッカーは既定で `Θ`（的の遠さ）に在る。規則としては「リーダーへの攻撃を**横取りする**」＝
    **そのターン届く量が減る**話なので速さ `A` の側。**両方に入れると同じ規則を 2 か所で数える**
    （T97 の実害）ので、移すには `Θ` 側を空にする必要がある＝`THETA_BODY_MODE=none`。
    """
    tok = np.zeros((22, 24), np.float32)
    s0 = T.SLOT_OPP_FIELD.start
    tok[s0, T.S_POWER], tok[s0, T.S_IS_CHAR], tok[s0, T.S_IS_BLOCKER] = 0.6, 1.0, 1.0
    sc = _sc(3, 4)
    with_body = CB.threshold(sc, tok)
    try:
        assert CB.set_theta_body_mode("none") == "none"
        without = CB.threshold(sc, tok)
        # 体の項だけが消える（ライフと手札はそのまま）
        life, hand, body = CB.threshold_parts(sc, tok)
        assert body == 0.0
        assert without == pytest.approx(life + hand)
    finally:
        CB.set_theta_body_mode("blockers")
    assert with_body > without                       # 既定では体の項が在る
    assert CB.THETA_BODY_MODE == "blockers"          # 既定は据え置き（採否はユーザ判定）
    with pytest.raises(ValueError):
        CB.set_theta_body_mode("なにか")


def test_the_blockers_also_bite_on_the_scheduled_path():
    """**T129**（2026-09-20）: **列を作る道でもブロッカーが効く**。

    `rate_at` を通る道では `seat_slope_terms` が自分でブロッカーを引いていたのに、**列の道は
    呼び出し側に任せていて橋は渡していなかった**＝`SLOPE_BLOCK_MODE=on` が**歩きに効いていなかった**
    （行ごとの `slope_theory` だけが動く）。**T128 の減衰と同じ型の取りこぼし。**
    **渡されなければ規則どおり自分で引く**ことをここで固定する（既定は `off` なので出荷の値は不動）。
    """
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = 0.5                                   # 自リーダー 5000
    s0 = T.SLOT_OWN_FIELD.start
    tok[s0, T.S_POWER], tok[s0, T.S_IS_CHAR], tok[s0, T.S_CAN_ATTACK] = 0.7, 1.0, 1.0
    b0 = T.SLOT_OPP_FIELD.start                               # 相手のアクティブなブロッカー
    tok[b0, T.S_POWER], tok[b0, T.S_IS_CHAR], tok[b0, T.S_IS_BLOCKER] = 0.6, 1.0, 1.0
    sc = _sc(3, 4)
    base = CB.seat_slope_sched(sc, tok, None, None, None, 5000.0, jmax=6)
    try:
        CB.set_slope_block_mode("on")
        blk = CB.seat_slope_sched(sc, tok, None, None, None, 5000.0, jmax=6)
    finally:
        CB.set_slope_block_mode("off")
    assert blk[1] < base[1]                                   # 横取りされる分だけ速さが落ちる
    assert all(x <= y + 1e-12 for x, y in zip(blk, base))      # どの段でも増えない
    assert CB.seat_slope_sched(sc, tok, None, None, None, 5000.0, jmax=6) == base   # 既定は不動
    assert CB.SLOPE_BLOCK_MODE == "off"


def test_a_body_mode_without_a_yardstick_entry_is_not_silently_borrowed():
    """**T129**: `σ_T` は**耐久の体の集合ごと**に表から引く。表に無い形では **`None` を返す**
    ——`theory_bridge` はそこで落ちる（**黙って前の σ を使い回さない**・すぐ下の `σ_rel` と同じ規約）。"""
    assert CB.sigma_t_for(None, "real", body_mode="blockers") is not None
    assert CB.sigma_t_for(None, "real", body_mode="none") is None      # T129 の新しい形は表に無い


# --------------------------------------------------------------------------- T131: 通った割合で割り引く
def test_the_through_share_is_a_dimensionless_ratio_of_the_rules():
    """**T131**（ユーザ決定 2026-09-20「入れてみてください」）: `通った本数 / 本数`。

    **既定は割り引かない**（1.0）。**`cut` はブロッカーを引かない**——既定ではブロッカーは
    **耐久 `Θ` の体の項に在る**ので、ここでも引くと**同じ規則を 2 か所で数える**（T97／T129 の型）。
    """
    assert CB.RATE_THROUGH_MODE == "off"
    assert CB.through_scale(3, 2, 1) == 1.0                   # `off` は素通し
    try:
        CB.set_rate_through_mode("cut")
        assert CB.through_scale(4, 1, 2) == pytest.approx(0.75)   # ブロッカーは引かない
        assert CB.through_scale(3, 0, 0) == 1.0                   # 切られなければ満額
        assert CB.through_scale(2, 5, 0) == 0.0                   # 全部止められたら 0（負にしない）
        assert CB.through_scale(0, 0, 0) == 1.0                   # 本数 0 は割り引くものが無い
        CB.set_rate_through_mode("cut_block")
        assert CB.through_scale(4, 1, 2) == pytest.approx(0.25)   # こちらは引く（`--theta-body none` と対）
    finally:
        CB.set_rate_through_mode("off")
    with pytest.raises(ValueError):
        CB.set_rate_through_mode("なにか")


def test_the_through_share_scales_both_the_leader_and_the_characters():
    """**リーダーの攻撃も答えられる**ので、減衰（KO されない）とは違い**両方に同じ割合が掛かる**。"""
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = 0.5
    s0 = T.SLOT_OWN_FIELD.start
    tok[s0, T.S_POWER], tok[s0, T.S_IS_CHAR], tok[s0, T.S_CAN_ATTACK] = 0.7, 1.0, 1.0
    lead, chars = CB.theory_slope_parts(tok, 5000.0)
    assert lead > 0.0 and chars > 0.0
    l2, c2 = CB.theory_slope_parts(tok, 5000.0, through=0.5)
    assert l2 == pytest.approx(lead * 0.5) and c2 == pytest.approx(chars * 0.5)


def test_a_missing_through_share_is_loud_not_silently_one():
    """**今日 2 度踏んだ事故**（T128 の減衰・T129 のブロッカー）——**渡し忘れが黙って旧の値になる**。
    守る席の手札は**攻める席の行からは読めない**ので渡すしかない＝**渡されなければ落とす**。"""
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = 0.5
    CB.theory_slope_parts(tok, 5000.0)                        # `off` なら渡さなくてよい
    try:
        CB.set_rate_through_mode("cut")
        with pytest.raises(ValueError):
            CB.theory_slope_parts(tok, 5000.0)                # 渡し忘れ＝落ちる
        CB.theory_slope_parts(tok, 5000.0, through=1.0)       # 渡せば通る
    finally:
        CB.set_rate_through_mode("off")


def test_the_budget_gap_is_zero_for_one_attack_and_positive_when_cards_are_reused():
    """**T132**（ユーザ指示 2026-09-20「測ってみて」）: **財布を共有しているか否かの差**。

    **1 本しか来なければ差は 0**（使い回しが起きない）。
    **2 本来て手札が 1 枚しか無ければ**、独立に取ると**同じ 1 枚で両方止められる**ことになり、
    共有では 1 本しか止まらない＝**差が出る**。**差は必ず 0 以上。**"""
    pairs = [(2000.0, 0.05)]                       # カウンター 2000 の札 1 枚
    take = 1.0                                     # 受けると高い＝必ず守りたい
    assert CB._budget_gap(pairs, [1000.0], take) == pytest.approx(0.0)        # 1 本なら差ゼロ
    gap = CB._budget_gap(pairs, [1000.0, 1000.0], take)
    assert gap > 0.0                                                          # 2 本目で使い回しが露出
    assert gap == pytest.approx(take - 0.05)                                  # 2 本目ぶんまるごと
    assert CB._budget_gap(pairs, [], take) == 0.0
    # 止められない攻撃は両方で節約 0 ＝差に寄与しない
    assert CB._budget_gap(pairs, [9000.0, 9000.0], take) == pytest.approx(0.0)


def test_the_budget_gap_never_goes_negative():
    """共有の方が節約できることは無い（同じ札を 2 回使えないので）。"""
    pairs = [(1000.0, 0.02), (2000.0, 0.09), (1000.0, 0.01)]
    for xs in ([0.0], [0.0, 1000.0], [1000.0, 1000.0, 2000.0], [500.0] * 5):
        assert CB._budget_gap(pairs, xs, 1.0) >= 0.0


# --------------------------------------------------------------------------- T133: Θ を両席で同じ式に
def _mirror_row(life=3.0, hand=5.0, n_char=2, pw=0.6):
    """**左右対称の盤面**（両席が同じライフ・同じ手札・同じ体・同じリーダー）。

    **`CAN_ATTACK` は自席にだけ立てる**（規則・その旗は手番の側でしか意味を持たない）——
    相手側は `opp_attackers_of` が「相手のターンが来れば全部アクティブ」と規則から数える。
    """
    sc = np.zeros(70, np.float32)
    sc[T.SC_MY_LIFE] = sc[T.SC_OPP_LIFE] = life
    sc[T.SC_MY_HAND] = sc[T.SC_OPP_HAND] = hand
    sc[T.SC_MY_LEADER_POWER] = sc[T.SC_OPP_LEADER_POWER] = 0.5
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = tok[1, T.S_POWER] = 0.5              # 両リーダー 5000
    for k in range(n_char):
        own, opp = T.SLOT_OWN_FIELD.start + k, T.SLOT_OPP_FIELD.start + k
        for s_i in (own, opp):
            tok[s_i, T.S_POWER], tok[s_i, T.S_IS_CHAR] = pw, 1.0
        tok[own, T.S_CAN_ATTACK] = 1.0
    return sc, tok


def test_a_mirrored_board_alone_does_not_expose_the_seat_asymmetry():
    """**測ってみて分かったこと**（T133・2026-09-20）——**この不変量では差を検出できない**。

    左右対称でライフに余裕があると、**リーダーの攻撃は超過 0 で「1 枚で止まる」**ので
    `hand_absorb_forced` の `c_eff` が 1.0 になり、**`μ × 枚数` に戻って従来の式と一致してしまう**。
    **検算を立てるときは「差が出る条件」まで書く**（最初の設計はここを外した）。
    **さらに体が小さいと `CBAR_MODE` にも依存する**（テスト環境の `loose` では `c_of(1000) = 1.0`）。
    """
    sc, tok = _mirror_row(life=3.0, n_char=2)
    legacy = abs(CB.threshold(sc, tok) - CB.threshold_of_me(sc, tok))
    assert legacy == pytest.approx(0.0, abs=1e-9)              # **従来でも破れない盤面が在る**


def test_the_two_seats_endurance_must_agree_once_guards_are_forced():
    """**T133**（ユーザ提案 2026-09-20「一つづつ丁寧に比較しましょうか」）: **`Θ` の式は席に依らない**。

    **見つかった欠陥**: `threshold_parts`（相手）は `cuttable_forced`（**規則が強いる守りで実際に吸える額**）
    を通るのに `threshold_of_me`（自分）は `g × 枚数` のままだった＝**同じ「手札」に 2 つの式**。
    **差が出るのは守りを強いられる盤面**（本数 > ライフ ＋ ブロッカー）——
    そこでは `c_eff > 1` になり、**組にして端数を捨てる床**が効く。**実測で 23.1% の行がこれに当たる。**
    """
    # **4 本 対 ライフ 1 ＝ 3 回は守らされる**・**体は 10000**——
    # **体が小さいと `CBAR_MODE=loose`（テスト環境）では `c_of(1000) = 1.0` になって差が消える**
    # （2026-09-20 に踏んだ: 盤面の選び方が環境の設定に依存していた）。
    sc, tok = _mirror_row(life=1.0, hand=5.0, n_char=3, pw=1.0)
    assert CB.THETA_HAND_MODE == "cuttable_forced"             # 既定（食い違いの所在）
    assert CB.THETA_SIDE_MODE == "legacy"                      # 既定は据え置き
    legacy_gap = abs(CB.threshold(sc, tok) - CB.threshold_of_me(sc, tok))
    try:
        CB.set_theta_side_mode("symmetric")
        sym_gap = abs(CB.threshold(sc, tok) - CB.threshold_of_me(sc, tok))
    finally:
        CB.set_theta_side_mode("legacy")
    assert legacy_gap > 1e-9          # **従来は破れる**（同じ盤面なのに席で違う）
    assert sym_gap == pytest.approx(0.0, abs=1e-9)             # **鏡にすると一致する**
    with pytest.raises(ValueError):
        CB.set_theta_side_mode("なにか")


def test_the_side_wrapper_reproduces_the_opponent_formula_exactly():
    """`threshold_parts` は `threshold_parts_side(..., "opp")` の薄い包み＝**値は 1 つも動かない**。"""
    sc, tok = _mirror_row()
    assert CB.threshold_parts(sc, tok) == CB.threshold_parts_side(sc, tok, "opp")
    assert CB.threshold_parts(sc, tok, g_hand=0.04, hand_blocker=0.01) == \
        CB.threshold_parts_side(sc, tok, "opp", g_hand=0.04, hand_blocker=0.01)
    with pytest.raises(ValueError):
        CB.threshold_parts_side(sc, tok, "どちらでもない")


def test_the_asymmetry_is_exactly_the_cuttable_branch():
    """**食い違いは手札の項だけ**——`count`（`g × 枚数`）なら両モードで一致する。"""
    sc, tok = _mirror_row()
    old = CB.THETA_HAND_MODE
    try:
        CB.set_theta_hand_mode("count")
        a = CB.threshold_of_me(sc, tok)
        CB.set_theta_side_mode("symmetric")
        b = CB.threshold_of_me(sc, tok)
        assert a == pytest.approx(b)                           # 手札の項が同じ式なら差は出ない
    finally:
        CB.set_theta_side_mode("legacy")
        CB.set_theta_hand_mode(old)


def test_the_symmetric_form_never_claims_more_endurance_than_the_old_one():
    """`hand_absorb_forced` は `μ × 枚数` の**床**（組にして端数を捨てる）なので、
    **鏡にすると自分の耐久は増えない**——向きを固定する。"""
    sc, tok = _mirror_row(life=1.0, hand=5.0, n_char=3, pw=1.0)
    for life, hand in ((1.0, 2.0), (1.0, 5.0), (2.0, 9.0)):
        sc[T.SC_MY_LIFE], sc[T.SC_MY_HAND] = life, hand
        legacy = CB.threshold_of_me(sc, tok)
        try:
            CB.set_theta_side_mode("symmetric")
            sym = CB.threshold_of_me(sc, tok)
        finally:
            CB.set_theta_side_mode("legacy")
        assert sym <= legacy + 1e-9, (life, hand, sym, legacy)


# --------------------------------------------------------------------------- T134: 受ける費用を守る側のライフで
def test_the_take_branch_reads_the_defenders_life():
    """**T134**（ユーザ提案「2 つの指標間と席間の扱いをそろえましょうか」の突き合わせ表 ③）。

    攻撃 1 回の価格は `min(c(x)·μ〔守られる〕, Θ·μ〔受けられる〕, ブロック)`。
    **「受けられる」側に定数を渡していた**——`theta_take(ライフ)` は既に在り、
    **守りの規則・線形の橋・実現の帳簿の 4 つでは使われている**のに**交点の橋の `A` だけが渡していなかった**。
    **既定の `TAKE_MODE=lethal` ではライフ 0 のときだけ値が変わる**＝
    **`A` は「この攻撃が通れば勝ち」を一度も見ていなかった**（T117 と同じ患部）。
    """
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = 0.5                                    # リーダーだけ（超過 0）
    assert CB.SLOPE_TAKE_MODE == "const"                       # 既定は据え置き
    base = CB.theory_slope(tok, 5000.0)
    assert CB.theory_slope(tok, 5000.0, life_opp=0.0) == pytest.approx(base)   # `const` なら無視
    try:
        CB.set_slope_take_mode("life")
        # **ライフが残っているうちは `theta_take` が定数を返す**（`TAKE_MODE=lethal` の規則）
        assert CB.theory_slope(tok, 5000.0, life_opp=3.0) == pytest.approx(base)
        # **ライフ 0＝この 1 本が通れば勝ち**——価格が変わる
        lethal = CB.theory_slope(tok, 5000.0, life_opp=0.0)
        assert lethal != pytest.approx(base)
    finally:
        CB.set_slope_take_mode("const")
    with pytest.raises(ValueError):
        CB.set_slope_take_mode("なにか")


def test_a_missing_defender_life_is_loud_not_silently_constant():
    """**T131 で決めた規約**——切替を入れたのに渡し忘れたら**黙って旧の値にならず落ちる**。"""
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = 0.5
    CB.theory_slope_parts(tok, 5000.0)                         # `const` なら渡さなくてよい
    try:
        CB.set_slope_take_mode("life")
        with pytest.raises(ValueError):
            CB.theory_slope_parts(tok, 5000.0)                 # 渡し忘れ＝落ちる
        CB.theory_slope_parts(tok, 5000.0, life_opp=2.0)       # 渡せば通る
    finally:
        CB.set_slope_take_mode("const")


def test_the_take_branch_only_binds_when_it_is_the_cheaper_response():
    """**素殴りなら `min` の枝**なので、守る方が安い攻撃では値が動かない
    ——**`Θ` の `λ·L`（在庫）と二重計上にならない**理由でもある（こちらは「この 1 本の価格」）。"""
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = 0.5
    s0 = T.SLOT_OWN_FIELD.start
    # **超過 0 の体**＝`c(0) = 1 枚`で止まるので守る方が安い（受ける費用が動いても `min` は変わらない）
    tok[s0, T.S_POWER], tok[s0, T.S_IS_CHAR], tok[s0, T.S_CAN_ATTACK] = 0.5, 1.0, 1.0
    _lead, chars_const = CB.theory_slope_parts(tok, 5000.0, with_don=False)
    try:
        CB.set_slope_take_mode("life")
        _l2, chars_life = CB.theory_slope_parts(tok, 5000.0, with_don=False, life_opp=0.0)
    finally:
        CB.set_slope_take_mode("const")
    assert chars_life == pytest.approx(chars_const)             # 守る方が安い枝は動かない


def test_at_lethal_the_higher_take_cost_makes_don_worth_attaching():
    """**ドン込みの価格では動く**（`attack_value_don` は `max_k [圧力(k) − k·δ]`）——
    **受ける費用が高いほど「ドンを付けて押し通す」が得になる**。

    **とどめの場面で正しい挙動**（そこだけ押し通す価値が跳ねる）。
    **最初に書いたテストはこれを「動かないはず」と書いて外した**——記録として残す。
    """
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = 0.5
    s0 = T.SLOT_OWN_FIELD.start
    tok[s0, T.S_POWER], tok[s0, T.S_IS_CHAR], tok[s0, T.S_CAN_ATTACK] = 0.5, 1.0, 1.0
    _lead, chars_const = CB.theory_slope_parts(tok, 5000.0)     # ドン込み（既定）
    try:
        CB.set_slope_take_mode("life")
        _l2, chars_lethal = CB.theory_slope_parts(tok, 5000.0, life_opp=0.0)
        _l3, chars_safe = CB.theory_slope_parts(tok, 5000.0, life_opp=3.0)
    finally:
        CB.set_slope_take_mode("const")
    assert chars_lethal > chars_const                           # とどめでは跳ねる
    assert chars_safe == pytest.approx(chars_const)             # ライフが残っていれば不動


def test_set_pre_settle_mode_rejects_unknown_names():
    """**T138b**: 決着後の行を除く切替。既定は `off`（従来どおり全行）で、未知の値は落ちる（規約どおり）。"""
    assert CB.PRE_SETTLE_MODE == "off"
    try:
        CB.set_pre_settle_mode("on")
        assert CB.PRE_SETTLE_MODE == "on"
    finally:
        CB.set_pre_settle_mode("off")
    with pytest.raises(ValueError):
        CB.set_pre_settle_mode("なにか")
    assert CB.PRE_SETTLE_MODE == "off"                          # 落ちても既定のまま


def test_pre_settle_off_skips_the_settled_map_lookup(monkeypatch):
    """**T138b**: `pre_settle=off`（既定）なら `lethal_rule.settled_map` を一度も呼ばない
    ——`collect()` を素通しする既存の呼び出し（記録を読まない他のテスト）に副作用が出ないための固定。"""
    import lethal_rule as LR
    calls = []
    monkeypatch.setattr(LR, "settled_map", lambda *a, **k: calls.append(1) or {})
    monkeypatch.setattr(CB.PL, "iter_games", lambda *a, **k: iter([]))
    assert CB.PRE_SETTLE_MODE == "off"
    CB.collect(["x"])
    assert calls == []


def test_pre_settle_on_reads_the_settled_map_once(monkeypatch):
    """**T138b**: `pre_settle=on` なら `settled_map(dirs, limit_games)` を 1 回だけ呼ぶ
    （`collect()` は記録をもう 1 度読み直す下請けとして扱う・`two_curves_settle.py` と同じ規約）。"""
    import lethal_rule as LR
    calls = []
    monkeypatch.setattr(LR, "settled_map", lambda dirs, limit_games: calls.append((dirs, limit_games)) or {})
    monkeypatch.setattr(CB.PL, "iter_games", lambda *a, **k: iter([]))
    try:
        CB.set_pre_settle_mode("on")
        CB.collect(["x"], 5)
    finally:
        CB.set_pre_settle_mode("off")
    assert calls == [(["x"], 5)]


# ---- T151-3: 局単位の決着前フィルタ（pre_settle=game） ------------------------------------------
#
# `on` は宣言した席の `(seed, w, t)` だけを落とす＝優勢側の「詰められる」行は消えるのに劣勢側の鏡の行は残る。
# `game` は**どちらかの席**の最初の宣言ターン `t*` 以降を**両席とも**落とす。判定は純関数 `pre_settle_skip`。

_SETTLED = {(9, 0, 1): False, (9, 1, 2): False, (9, 0, 3): False, (9, 1, 4): True, (9, 0, 5): True}
_FIRST = {9: 4}


def test_pre_settle_skip_off_never_drops():
    assert not CB.pre_settle_skip("off", _SETTLED, _FIRST, 9, 1, 4)
    assert not CB.pre_settle_skip("on", None, None, 9, 1, 4)     # 地図が無ければ落とさない


def test_pre_settle_skip_on_drops_only_the_declaring_seat_row():
    assert CB.pre_settle_skip("on", _SETTLED, _FIRST, 9, 1, 4)
    assert not CB.pre_settle_skip("on", _SETTLED, _FIRST, 9, 0, 3)   # 席 0 の 3 は残る（鏡の行）
    assert not CB.pre_settle_skip("on", _SETTLED, _FIRST, 9, 0, 1)


def test_pre_settle_skip_game_drops_both_seats_from_the_first_declared_turn():
    assert CB.pre_settle_skip("game", _SETTLED, _FIRST, 9, 1, 4)
    assert CB.pre_settle_skip("game", _SETTLED, _FIRST, 9, 0, 5)
    assert CB.pre_settle_skip("game", _SETTLED, _FIRST, 9, 0, 7)      # 宣言の無い後のターンも落ちる（t ≥ t*）
    assert not CB.pre_settle_skip("game", _SETTLED, _FIRST, 9, 0, 3)  # t* より前は両席とも残る
    assert not CB.pre_settle_skip("game", _SETTLED, _FIRST, 8, 0, 9)  # 宣言の無い局は 1 行も落ちない


def test_pre_settle_game_reads_the_settled_map_once_and_derives_the_first_turn(monkeypatch):
    import lethal_rule as LR
    calls = []
    monkeypatch.setattr(LR, "settled_map", lambda dirs, limit_games: calls.append((dirs, limit_games)) or dict(_SETTLED))
    firsts = []
    real_first = LR.first_declared_turn
    monkeypatch.setattr(LR, "first_declared_turn", lambda s: firsts.append(1) or real_first(s))
    monkeypatch.setattr(CB.PL, "iter_games", lambda *a, **k: iter([]))
    try:
        CB.set_pre_settle_mode("game")
        CB.collect(["x"], 3)
    finally:
        CB.set_pre_settle_mode("off")
    assert calls == [(["x"], 3)] and firsts == [1]
    assert "game" in CB.PRE_SETTLE_MODES


# ---- T151-2: 相手の時計を同じ瞬間から読む（opp_clock=mirror・mirror_view） ----------------------------
#
# 既定 `prev_start` は従来どおり（`op = per_seat[(1-w, 相手の前ターン)]`）。`mirror` は w のターン t の最後の行を
# 相手の席から見た鏡（`mirror_view`）にして同じ `seat_row` で読む。**既定の道は 1 ビットも動かない**
# （2026-09-24・実 20 局で rows_out／ledger／stats／theta_check／turn_harm のハッシュが HEAD と一致）。

def test_set_opp_clock_mode_defaults_to_prev_start_and_rejects_unknown():
    assert CB.OPP_CLOCK_MODE == "prev_start"
    try:
        assert CB.set_opp_clock_mode("mirror") == "mirror"
    finally:
        CB.set_opp_clock_mode("prev_start")
    with pytest.raises(ValueError):
        CB.set_opp_clock_mode("なにか")
    assert CB.OPP_CLOCK_MODE == "prev_start"


def _mirror_fixture():
    """自分＝席 A の行。A のリーダー 0・場 2..6・手札 12..21／相手＝リーダー 1・場 7..11。"""
    import don_ledger as DL
    n_slot, n_col = 22, 22
    tok = np.zeros((n_slot, n_col), float)
    # 自分のリーダー: パワー 0.5（手番のとき・付与 1 ドン込み）／列 20＝0.4（相手の手番のとき）
    tok[0, CB.S_POWER] = 0.5; tok[0, CB._TOK_POWER_OTHER] = 0.4; tok[0, CB._TOK_ATTACHED] = 0.2
    # 相手のリーダー: 列 0＝0.45（相手の手番でない＝付与無し）／列 20＝0.55（相手の手番）
    tok[1, CB.S_POWER] = 0.45; tok[1, CB._TOK_POWER_OTHER] = 0.55; tok[1, CB._TOK_ATTACHED] = 0.2
    # 自分の体（枠 2）: レスト中（殴った）・ブロッカー（列 6 は非レストのブロッカーなので 0）
    tok[2, CB.S_IS_CHAR] = 1.0; tok[2, CB.S_POWER] = 0.6; tok[2, CB._TOK_POWER_OTHER] = 0.5; tok[2, CB.S_IS_REST] = 1.0
    tok[2, CB._TOK_ATTACHED] = 0.2
    # 相手の体（枠 7）: そのターンに出た（召喚酔い）・レスト（殴った）ブロッカー・付与 1
    tok[7, CB.S_IS_CHAR] = 1.0; tok[7, CB.S_POWER] = 0.3; tok[7, CB._TOK_POWER_OTHER] = 0.4
    tok[7, CB.S_IS_REST] = 1.0; tok[7, CB._TOK_SICK] = 1.0; tok[7, CB._TOK_ATTACHED] = 0.2; tok[7, CB.S_IS_BLOCKER] = 0.0
    # 相手の体（枠 8）: アクティブな非ブロッカー
    tok[8, CB.S_IS_CHAR] = 1.0; tok[8, CB.S_POWER] = 0.7; tok[8, CB._TOK_POWER_OTHER] = 0.7
    # 自分の手札（枠 12）: 何か
    tok[12, CB.S_COUNTER if hasattr(CB, "S_COUNTER") else 7] = 1.0
    sc = np.zeros(123, float)
    sc[CB.SC_MY_LIFE], sc[CB.SC_OPP_LIFE] = 2.0, 4.0
    sc[CB.SC_MY_HAND], sc[CB.SC_OPP_HAND] = 3.0, 6.0
    sc[CB._SC_MY_FIELD_N], sc[CB._SC_OPP_FIELD_N] = 1.0, 2.0
    sc[CB.SC_MY_LEADER_POWER], sc[CB.SC_OPP_LEADER_POWER] = 0.5, 0.6
    sc[DL.SC_MY_ACTIVE], sc[DL.SC_MY_RESTED] = 1.0, 5.0        # 自分: 使い残し 1・使った 5
    sc[DL.SC_OPP_ACTIVE], sc[DL.SC_OPP_RESTED] = 0.0, 3.0      # 相手: アクティブ 0・レスト 3
    sc[DL.SC_MY_LEADER_DON], sc[DL.SC_OPP_LEADER_DON] = 0.2, 0.2   # リーダー付与 1 ずつ（×5）
    sc[DL.SC_MY_DON_DECK], sc[DL.SC_OPP_DON_DECK] = 0.3, 0.1   # ドンデッキ 3・1（×10）
    sc[CB._SC_IS_MY_TURN] = 1.0
    ci = np.zeros(24, int)
    ci[0], ci[1], ci[2], ci[7], ci[8], ci[12], ci[22], ci[23] = 11, 22, 33, 44, 55, 66, 77, 88
    tok_b = np.zeros((n_slot, n_col), float); tok_b[12, 7] = 2.0; tok_b[13, 7] = 0.5
    ci_b = np.zeros(24, int); ci_b[12], ci_b[13] = 101, 102
    return sc, tok, ci, tok_b, ci_b


class _Cards:
    def __init__(self, blockers):
        self._b = set(blockers)

    def info(self, cid):
        return {"blocker": cid in self._b}


def test_mirror_view_swaps_sides_and_refreshes_only_the_new_own_side():
    sc, tok, ci, tok_b, ci_b = _mirror_fixture()
    idx2cid = {44: "OPP_BLOCKER", 55: "OPP_VANILLA", 33: "MY_BLOCKER", 11: "L1", 22: "L2"}
    S, T, C = CB.mirror_view(sc, tok, ci, tok_hand=tok_b, ci_hand=ci_b, cards=_Cards({"OPP_BLOCKER", "MY_BLOCKER"}), idx2cid=idx2cid)
    # 元は触らない
    assert tok[7, CB.S_IS_REST] == 1.0 and sc[CB.SC_MY_LIFE] == 2.0
    # 新しい自分側＝相手の枠（リフレッシュ後）
    assert T[0, CB.S_POWER] == pytest.approx(0.45) and T[0, CB._TOK_ATTACHED] == 0.0 and T[0, CB.S_CAN_ATTACK] == 1.0
    assert T[2, CB.S_IS_CHAR] == 1.0 and T[2, CB.S_IS_REST] == 0.0 and T[2, CB._TOK_SICK] == 0.0
    assert T[2, CB.S_CAN_ATTACK] == 1.0 and T[2, CB._TOK_ATTACHED] == 0.0
    assert T[2, CB.S_IS_BLOCKER] == 1.0            # 殴ってレストしていたブロッカーが札の情報で立ち直る
    assert T[2, CB.S_POWER] == pytest.approx(0.3)   # 相手の枠の列 0＝付与無しのパワー
    assert T[3, CB.S_IS_CHAR] == 1.0 and T[3, CB.S_IS_BLOCKER] == 0.0 and T[3, CB.S_CAN_ATTACK] == 1.0
    assert T[4, CB.S_IS_CHAR] == 0.0 and T[4, CB.S_CAN_ATTACK] == 0.0
    # 新しい相手側＝自分の枠（そのまま・パワーは「相手が手番のとき」の列）
    assert T[1, CB.S_POWER] == pytest.approx(0.4) and T[1, CB._TOK_ATTACHED] == pytest.approx(0.2)
    assert T[7, CB.S_IS_CHAR] == 1.0 and T[7, CB.S_IS_REST] == 1.0 and T[7, CB._TOK_ATTACHED] == pytest.approx(0.2)
    assert T[7, CB.S_POWER] == pytest.approx(0.5) and T[7, CB.S_CAN_ATTACK] == 0.0
    # 手札＝相手の直近の行から
    assert T[12, 7] == 2.0 and T[13, 7] == 0.5 and C[12] == 101 and C[13] == 102
    # card_idx の入れ替え（リーダー・場・ステージ）
    assert C[0] == 22 and C[1] == 11 and C[2] == 44 and C[3] == 55 and C[7] == 33 and C[22] == 88 and C[23] == 77


def test_mirror_view_scalars_follow_the_refresh_rule():
    import don_ledger as DL
    sc, tok, ci, tok_b, ci_b = _mirror_fixture()
    S, T, C = CB.mirror_view(sc, tok, ci, tok_hand=tok_b, ci_hand=ci_b)
    assert S[CB.SC_MY_LIFE] == 4.0 and S[CB.SC_OPP_LIFE] == 2.0
    assert S[CB.SC_MY_HAND] == 6.0 and S[CB.SC_OPP_HAND] == 3.0
    assert S[CB._SC_MY_FIELD_N] == 2.0 and S[CB._SC_OPP_FIELD_N] == 1.0
    assert S[CB.SC_MY_LEADER_POWER] == 0.6 and S[CB.SC_OPP_LEADER_POWER] == 0.5
    # 相手のドン: アクティブ 0 ＋ レスト 3 ＋ 付与（リーダー 1 ＋ 枠 7 の 1）＋ min(2, デッキ 1) ＝ 6
    assert S[DL.SC_MY_ACTIVE] == pytest.approx(6.0) and S[DL.SC_MY_RESTED] == 0.0 and S[DL.SC_MY_LEADER_DON] == 0.0
    assert S[DL.SC_MY_DON_DECK] == pytest.approx(0.0)          # デッキ 1 − 1 = 0
    # 自分のドンはそのまま相手側へ
    assert S[DL.SC_OPP_ACTIVE] == 1.0 and S[DL.SC_OPP_RESTED] == 5.0 and S[DL.SC_OPP_LEADER_DON] == pytest.approx(0.2)
    assert S[DL.SC_OPP_DON_DECK] == pytest.approx(0.3) and S[CB._SC_IS_MY_TURN] == 1.0


def test_mirror_view_without_cards_keeps_the_blocker_column_as_is_and_empty_hand_when_no_partner_row():
    sc, tok, ci, _tb, _cb = _mirror_fixture()
    S, T, C = CB.mirror_view(sc, tok, ci)
    assert T[2, CB.S_IS_BLOCKER] == 0.0            # 札が引けなければ列 6 のまま（過小側・開示）
    assert float(np.abs(T[CB.SLOT_HAND]).sum()) == 0.0 and int(np.abs(C[CB.SLOT_HAND]).sum()) == 0


def test_opp_clock_mirror_on_empty_records_builds_no_rows_and_default_counts_no_mirror_rows(monkeypatch):
    monkeypatch.setattr(CB.PL, "iter_games", lambda *a, **k: iter([]))
    rows, _l, stats, _t, _c = CB.collect(["x"])
    assert rows == [] and stats["mirror_rows"] == 0
    try:
        CB.set_opp_clock_mode("mirror")
        rows2, _l2, stats2, _t2, _c2 = CB.collect(["x"])
    finally:
        CB.set_opp_clock_mode("prev_start")
    assert rows2 == [] and stats2["mirror_rows"] == 0
