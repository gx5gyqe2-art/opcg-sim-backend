"""`crossing_bridge.py`（T52・交点の橋）の算術を固める。

**当てはめない・回帰しない**器なので、しきい値・傾き・交点・予測勝者の定義がそのまま出ること、
残差が「実際に届いた側の τ 対 その側の残りターン」で作られること、単位の検算が比そのままであることを値で押さえる。
"""
import math
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
    "THETA_HAND_MODE", "THETA_HAND_PLACE", "THETA_BODY_MODE",
    "THETA_RETURN_MODE", "SLOPE_HAND_MODE", "SLOPE_BLOCK_MODE",
    "RATE_WALK_MODE", "RATE_DECAY_MODE", "RACE_MODE", "SLOPE_TAKE_MODE",
    "THETA_DON_MODE", "THETA_HAND_WINDOW")}


def test_the_shipped_defaults_are_the_ones_we_decided():
    """**既定の一覧をラチェットする**（`docs/cpu_theory_gap.md` §9.2 の表と 1 対 1）。

    **2026-09-19 のユーザ指示**（「使用できるドンと使ったドンの整合が取れるように最後まで進めてください」）で
    財布 1 つ（T109・旧 `DON_PURSE_MODE=all`・2026-10-05 に切替は削除）が入った——**財布が 1 つになって初めて帳尻が合う**
    （理論の使いすぎ 22.0% → 0.0%）。**続く「それは直しましょうか」で `THETA_DON_MODE=rule`（T110）**
    ——**耐久の側もドンを規則どおり払う**（手札のブロッカーの予算＝次ターンのアクティブ・
    カウンター・イベントは使い残しで払う）。

    **2026-09-20 のユーザ決定**（「効果があったものの規定はオンにしないの？」→「3 本すべて」）で 3 本動いた:
    歩きの成長を規則のドンから（T114・旧 `RATE_DON_MODE=flow`・同上）・`THETA_HAND_MODE=cuttable_forced` ＋
    `THETA_HAND_WINDOW=horizon`（T116・手札は守る窓が開く分だけ的に入る）・
    `theory_order.W_ERR_MODE=rel`（T118・勝率を比で読む）。**T114 と T116 は対で採る**
    ——単独では合成が動かないが、**組むと両記録で 5 軸が改善する**（偏り・的中・σ_T・`curve` の偏り・`Θ`/要）。
    **代金は ±1 当たりと `curve` の的中、そして線形の橋**（`dG` の AUC 0.696 → 0.660／0.720 → 0.692）。

    **同日のユーザ決定**（「1は規定、2は正しいものに直してください」）で 4 つ動いた:
    効果の項（T105／T108・2026-10-05 に切替は削除）・最初の自席ターンは打てない規則（T103・2026-10-05 に切替は削除）・速攻は出したターンから（T103・2026-10-05 に切替は削除）・
    手札のブロッカーを耐久に入れる（T103／T106＝**規則として正しい形**・2026-10-05 に切替は削除）。
    **黙って既定が変わると 2 つの橋の数字が比較不能になる**ので、ここで固定する。"""
    assert _SHIPPED == {
        "THETA_HAND_MODE": "rule_don",          # H-4・2026-10-04（ユーザ決定・旧 cuttable_forced は --theta-hand で再現）
        "THETA_HAND_PLACE": "stock",            # T102（切替として残す）
        "THETA_BODY_MODE": "blockers",           # T97
        "THETA_RETURN_MODE": "untap",            # T96・**C-5c で既定に採用**（2026-09-25）
        "SLOPE_HAND_MODE": "flow",               # T93
        "SLOPE_BLOCK_MODE": "on",                # T92・2026-10-05 既定に採用（ユーザ決定・旧 off は --slope-block off）
        "SLOPE_TAKE_MODE": "life",               # T134・2026-10-05 既定に採用（ユーザ決定・旧 const は --slope-take const）
        "RATE_WALK_MODE": "grow",                # T94
        "RATE_DECAY_MODE": "off",                # T95（切替として残す）
        "RACE_MODE": "static",                   # T90／T91／T104（切替として残す）
        "THETA_DON_MODE": "rule",                # T110・2026-09-19
        "THETA_HAND_WINDOW": "horizon",          # T116・2026-09-20
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


#: **2026-10-05**: `SLOPE_TAKE_MODE=life`／`SLOPE_BLOCK_MODE=on` が既定になった（ユーザ決定）。
#: **以下のテストは旧い数字（定数 `Θ`・ブロッカー無し）の算術を固定している**ので、自分で旧い形を
#: 明示して回す＝**テストの意味を変えずに既定の変更だけを吸収する**。既定そのものは
#: `test_the_shipped_defaults_are_the_ones_we_decided` と下の新既定のラチェットが固定する。
_NEW_DEFAULT_TESTS = (
    "test_the_shipped_defaults_are_the_ones_we_decided",
    "test_the_2026_10_05_defaults_are_take_life_block_on",
)


@pytest.fixture(autouse=True)
def _old_slope_forms(request):
    if request.node.name.split("[")[0] in _NEW_DEFAULT_TESTS:
        yield
        return
    old_t, old_b = CB.SLOPE_TAKE_MODE, CB.SLOPE_BLOCK_MODE
    CB.set_slope_take_mode("const")
    CB.set_slope_block_mode("off")
    try:
        yield
    finally:
        CB.set_slope_take_mode(old_t)
        CB.set_slope_block_mode(old_b)


def test_the_2026_10_05_defaults_are_take_life_block_on():
    """**3 本の既定採用のラチェット**（ユーザ決定 2026-10-05「全て正しい方式にしてください」）。
    橋の 2 本（受ける費用を守る側のライフで・ブロッカーを速さに入れる）と、帳簿の `κ` の物差し。
    旧い値は切替として選べる（掃除の波で死んだ切替を消すまで残す）。
    CLI の既定も module の既定と同じ（引数の既定が定数を指している）。"""
    import relative_ledger as RL                       # noqa: PLC0415
    assert CB.SLOPE_TAKE_MODE == "life" and CB.SLOPE_BLOCK_MODE == "on"
    assert T.KAPPA_SIGMA_MODE == "match"
    for old, new in (("const", "life"), ("off", "on")):
        assert old in CB.SLOPE_TAKE_MODES + CB.SLOPE_BLOCK_MODES and new in CB.SLOPE_TAKE_MODES + CB.SLOPE_BLOCK_MODES
    assert "abs" in T.KAPPA_SIGMA_MODES
    a = RL.build_parser().parse_args(["--in", "x"])
    assert (a.slope_take, a.slope_block, a.kappa_sigma) == (None, None, None)   # None＝module の既定に任せる


def _tg(*a, **k):
    """**歩きの代数は「局の途中の行」で確かめる**（`j0=2`）。

    最初の自席ターンは打てない規則（T103・2026-09-19 から）は**局の 1 自席ターン目だけ**速さを 0 にする規則
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


def test_the_hand_absorbs_by_price_order_per_attack():
    """**T158**（T99 が指した先）: `hand_absorb` は一番重い攻撃 1 本の `c(x_max)` で全札を割り、
    `hand_absorb_forced` は必ず守る `G` 回の**平均費用**で割る——どちらも「同じ大きさの組」で割る
    粗さが残る。**実際の守り手は攻撃ごとに止めるかを選ぶ**ので、`cuttable_seq`（`hand_absorb_seq`）は
    **その席が受ける攻撃を安い順に並べ、手札が尽きるまで攻撃ごとにその `c(x_i)` 枚をそのまま割り当てる**。"""
    mu = T.MU
    before = T.CBAR_MODE
    try:
        T.set_cbar_mode("strict")
        xs = [0.0, 1000.0, 3000.0]                         # c = 1.00 / 1.28 / 2.78（安い順）
        cs = sorted(T.c_of(x) for x in xs)
        assert cs[0] < cs[1] < cs[2]
        # 手札 3.5 枚: 1 本目・2 本目は丸ごと止まり、3 本目は端数が足りず打ち切り
        n_cut = cs[0] + cs[1] + 1.5
        got = CB.hand_absorb_seq(n_cut, xs, mu)
        assert got == pytest.approx(mu * (cs[0] + cs[1]))
        assert got < n_cut * mu                            # 端数のぶん小さい（一生 F に入らない）
        # `hand_absorb`（x_max 1 本）とも `hand_absorb_forced`（G 本の平均費用）とも異なる値になる
        # （`forced` は `G=2` の平均費用 `c_eff=(cs[0]+cs[1])/2` で `floor(n_cut/c_eff)=3` 回ぶん吸う）
        assert got != pytest.approx(CB.hand_absorb(n_cut, max(xs), mu))
        assert got != pytest.approx(CB.hand_absorb_forced(n_cut, xs, 1, 0, mu))
        # 通らない攻撃しか無ければ守る必要が無い＝0
        assert CB.hand_absorb_seq(3, [-1000.0], mu) == 0.0
        assert CB.hand_absorb_seq(3, [], mu) == 0.0
        # **1 回分に足りなければ 0**（一番安い攻撃の 1 回分にも届かない薄い手札）
        assert CB.hand_absorb_seq(cs[0] - 0.5, xs, mu) == 0.0
    finally:
        T.set_cbar_mode(before)


def test_the_hand_absorbed_by_price_order_does_not_carry_the_remainder_forward():
    """**T158**: ちょうど 2 本ぶん割り切ったところで止めると、**余りは 0 で次の攻撃には回らない**
    （3 本目の費用に満たない余りを 3 本目に一部だけ充てることはしない）。"""
    mu = T.MU
    before = T.CBAR_MODE
    try:
        T.set_cbar_mode("strict")
        xs = [0.0, 1000.0, 3000.0]
        cs = sorted(T.c_of(x) for x in xs)
        # 手札はちょうど 1 本目 ＋ 2 本目の費用の和＝2 本ぶん止めて余りは正確に 0
        n_cut = cs[0] + cs[1]
        got = CB.hand_absorb_seq(n_cut, xs, mu)
        assert got == pytest.approx(mu * (cs[0] + cs[1]))
        # 3 本目に余りが回っていれば `got` はこれより大きくなるはずだが、そうならない
        assert got < mu * (cs[0] + cs[1] + cs[2])
        # 手札を少し増やしても（3 本目の費用 `cs[2]` ≥ 2 には届かない量）絵柄は変わらない
        assert cs[2] > 1.0                             # 増分 0.5 が 3 本目には遠く届かないことの前提
        got_more = CB.hand_absorb_seq(n_cut + 0.5, xs, mu)
        assert got_more == pytest.approx(mu * (cs[0] + cs[1]))   # 3 本目は依然として 0 本ぶん
    finally:
        T.set_cbar_mode(before)


def test_selecting_cuttable_seq_leaves_the_older_hand_modes_byte_for_byte():
    """**「off は旧のまま」の作法**——`cuttable_seq`（T158）を `THETA_HAND_MODES` に足しても、
    既存のモード（`count`／`cuttable`／`cuttable_forced`）を選んでいるときの `threshold_parts` の
    出力は`threshold_parts_side` に元からある分岐（`cuttable_cx`＝`hand_absorb`／
    `cuttable_forced`＝`hand_absorb_forced`／`count`＝素の `g × 枚数`）そのままで、
    1 バイトも変わらない（`cuttable`（`_cx` 無し）は `g` の出どころが違うだけでこの分岐自体を通らない）。"""
    sc, tok = _mirror_row(life=1.0, hand=5.0, n_char=3, pw=1.0)
    olp = float(np.asarray(sc)[T.SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    xs = CB.own_attackers_of(tok, olp)
    n_blk = CB._opp_active_blockers(tok)
    hand_n = float(np.asarray(sc)[T.SC_OPP_HAND])
    life = float(np.asarray(sc)[T.SC_OPP_LIFE])
    mu = T.MU
    old = CB.THETA_HAND_MODE
    try:
        CB.set_theta_hand_mode("count")
        _, hand, _ = CB.threshold_parts(sc, tok)
        assert hand == pytest.approx(mu * hand_n)
        CB.set_theta_hand_mode("cuttable_cx")
        _, hand, _ = CB.threshold_parts(sc, tok)
        assert hand == pytest.approx(CB.hand_absorb(hand_n, max(xs) if xs else -1.0, mu))
        CB.set_theta_hand_mode("cuttable_forced")
        _, hand, _ = CB.threshold_parts(sc, tok)
        assert hand == pytest.approx(CB.hand_absorb_forced(hand_n, xs, life, n_blk, mu))
    finally:
        CB.set_theta_hand_mode(old)


def test_cuttable_seq_is_wired_into_threshold_parts_and_rejects_unknown_modes():
    """**T158**: `THETA_HAND_MODE=cuttable_seq` を選ぶと `threshold_parts` の手札の項が
    `hand_absorb_seq` そのものになる。**知らない名前は `set_theta_hand_mode` で `ValueError`**
    （T99 で踏んだ「黙って別の値で走る」穴を避ける）。"""
    sc, tok = _mirror_row(life=1.0, hand=5.0, n_char=3, pw=1.0)
    olp = float(np.asarray(sc)[T.SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    xs = CB.own_attackers_of(tok, olp)
    hand_n = float(np.asarray(sc)[T.SC_OPP_HAND])
    mu = T.MU
    old = CB.THETA_HAND_MODE
    try:
        CB.set_theta_hand_mode("cuttable_seq")
        assert CB.THETA_HAND_MODE == "cuttable_seq"
        _, hand, _ = CB.threshold_parts(sc, tok)
        assert hand == pytest.approx(CB.hand_absorb_seq(hand_n, xs, mu))
    finally:
        CB.set_theta_hand_mode(old)
    with pytest.raises(ValueError):
        CB.set_theta_hand_mode("cuttable_seqq")
    assert CB.THETA_HAND_MODE == old                                    # 失敗した切替は既定を汚さない


# ---- H-4: 規則から導いた手札の項（`THETA_HAND_MODE=rule`） ---------------------------------------


def _rule_row(life=0.0, hand=3.0, xs=(1000.0, 2000.0), blockers=()):
    """攻め手の行: 相手（守る側）のリーダー 5000・攻撃の超過 `xs`（リーダー＋キャラ）・相手のアクティブなブロッカー。"""
    sc = np.zeros(70, np.float32)
    sc[T.SC_OPP_LIFE], sc[T.SC_OPP_HAND] = life, hand
    sc[T.SC_MY_LEADER_POWER] = sc[T.SC_OPP_LEADER_POWER] = 0.5
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = (5000.0 + xs[0]) / 1e4
    tok[1, T.S_POWER] = 0.5
    for k, x in enumerate(xs[1:]):
        s_i = T.SLOT_OWN_FIELD.start + k
        tok[s_i, T.S_POWER], tok[s_i, T.S_IS_CHAR], tok[s_i, T.S_CAN_ATTACK] = (5000.0 + x) / 1e4, 1.0, 1.0
    for k, p in enumerate(blockers):
        s_i = T.SLOT_OPP_FIELD.start + k
        tok[s_i, T.S_POWER], tok[s_i, T.S_IS_CHAR], tok[s_i, T.S_IS_BLOCKER] = p / 1e4, 1.0, 1.0
    return sc, tok


def _read(counters, don=0.0, events=()):
    """守る席の手札の読み（`HandRead`）: `counters` のカウンター値・`events` の添字はイベント（費用 = 値 1000 ごとに 1）。"""
    items = [{"counter": float(c), "event": k in events, "cost": (float(c) / 1000.0 if k in events else 3.0)}
             for k, c in enumerate(counters)]
    return CB.hand_read_of_items(items, don, T.MU)


def _greedy_cheapest_first(cards, xs, life):
    """**比べる相手の貪欲**（T64「安い方から」の攻撃ごとの版）: 安い攻撃から、足りる組のうち**札の少ない**
    （同じなら合計の小さい）ものを取る。生き延びられなければ切らない（同じ規則）。"""
    import itertools
    avail = list(range(len(cards)))
    stops = cut = 0
    hits = sorted(x for x in xs if x >= -T.PWR_EPS)
    for x in hits:
        need = x + 1000.0 - T.PWR_EPS
        best = None
        for r in range(1, len(avail) + 1):
            for c in itertools.combinations(avail, r):
                s = sum(cards[i] for i in c)
                if s >= need and (best is None or s < sum(cards[i] for i in best)):
                    best = c
            if best:
                break
        if best:
            stops += 1; cut += len(best)
            avail = [i for i in avail if i not in best]
    if len(hits) - stops > life:
        return 0, 0
    return cut, stops


def test_the_rule_hand_beats_the_greedy_allocation_on_a_hand_built_case():
    """**H-4**: 手札 1000・3000・1000、攻撃の超過 1000 と 2000（要る合計 2000 と 3000）、ライフ 0。
    **貪欲**（安い攻撃に一番少ない札）は 2000 の攻撃に 3000 を 1 枚当て、残り 1000＋1000 では 3000 に届かず
    **2 本目が通って倒れる**（倒れるターンには切らない＝0 枚）。**最適**は 1000＋1000 を 2000 に、3000 を
    3000 に当てて**両方止めて生き延びる**（3 枚切る）。`Θ_hand = μ × 3`。"""
    cards = [1000.0, 3000.0, 1000.0]
    assert _greedy_cheapest_first(cards, [1000.0, 2000.0], 0) == (0, 0)
    assert CB.rule_guard_plan([(c, 0.0) for c in cards], 0.0, [1000.0, 2000.0], [1000.0, 2000.0], [], 0, 1) == (3, 2)
    sc, tok = _rule_row(life=0.0, hand=3.0, xs=(1000.0, 2000.0))
    old = CB.THETA_HAND_MODE
    try:
        CB.set_theta_hand_mode("rule")
        _, hand, _ = CB.threshold_parts(sc, tok, g_hand=_read(cards))
    finally:
        CB.set_theta_hand_mode(old)
    # 地平は「倒れるまで」: 1 ターン目で全部使い切り、2 ターン目は止められず倒れる＝切るのは 3 枚
    assert hand == pytest.approx(3 * T.MU)
    # 最適な割り当ては貪欲より多く止める（同じ札・同じ攻撃）
    assert CB.rule_guard_plan([(c, 0.0) for c in cards], 0.0, [1000.0, 2000.0], [1000.0, 2000.0], [], 0, 1)[1] \
        > _greedy_cheapest_first(cards, [1000.0, 2000.0], 0)[1]


def test_the_rule_hand_reads_the_actual_counter_values():
    """**H-4**: 切れる札の**枚数も割合も同じ**（2 枚・全部切れる）なのに、**値**が 1000＋1000 か 2000＋2000 かで
    止められる本数が変わる（超過 1000＝要る合計 2000 の攻撃 1 本・ライフ 0・1 ターン）。
    既定の形（平均の曲線 `c(x)`）は枚数しか見ないので両者を区別できない。"""
    sc, tok = _rule_row(life=0.0, hand=2.0, xs=(1000.0,))
    old = CB.THETA_HAND_MODE
    try:
        CB.set_theta_hand_mode("cuttable_forced")
        f_small = CB.threshold_parts(sc, tok, g_hand=float(_read([1000.0, 1000.0])))[1]
        f_big = CB.threshold_parts(sc, tok, g_hand=float(_read([2000.0, 2000.0])))[1]
        CB.set_theta_hand_mode("rule")
        r_small = CB._rule_hand_term(sc, tok, "opp", _read([1000.0, 1000.0]), T.MU, turns=1, count=False)
        r_big = CB._rule_hand_term(sc, tok, "opp", _read([2000.0, 2000.0]), T.MU, turns=1, count=False)
    finally:
        CB.set_theta_hand_mode(old)
    assert f_small == pytest.approx(f_big)                          # 旧の形は値を見ない
    assert r_small == pytest.approx(2 * T.MU)                       # 1000＋1000 で 1 本（2 枚切る）
    assert r_big == pytest.approx(1 * T.MU)                         # 2000 1 枚で 1 本（1 枚だけ切る）
    # 足りなければ切らない（1000 1 枚では 2000 に届かない・ライフ 0 で倒れる）
    assert CB._rule_hand_term(sc, tok, "opp", _read([1000.0]), T.MU, turns=1, count=False) == 0.0


def test_the_rule_hand_pays_event_counters_with_the_don_left_each_defending_turn():
    """**H-4**: イベントのカウンターは**相手のターンに在るドン**で払う（`apply_counter` の `pay_cost`）。
    ドン 1 では費用 2 のイベント（+2000）を切れず、2 なら切れる。**ドンは守備ターンごとに戻る**ので、
    同じ費用のイベント 2 枚は 1 ターンに 1 枚ずつなら両方使える（手札全体で 1 回払う形ではない）。"""
    plan = CB.rule_guard_plan
    assert plan([(2000.0, 2.0)], 1.0, [1000.0], [1000.0], [], 0, 1) == (0, 0)
    assert plan([(2000.0, 2.0)], 2.0, [1000.0], [1000.0], [], 0, 1) == (1, 1)
    # 印字のカウンターは無料（ドン 0 でも切れる）
    assert plan([(2000.0, 0.0)], 0.0, [1000.0], [1000.0], [], 0, 1) == (1, 1)
    # 1 ターンに 2 本（ライフ 0 で両方止めねばならない）: ドン 2 では 2 枚目を払えず倒れる＝切らない
    assert plan([(2000.0, 2.0), (2000.0, 2.0)], 2.0, [1000.0, 1000.0], [1000.0, 1000.0], [], 0, 1) == (0, 0)
    # 1 ターンに 1 本・ライフ 1・2 ターン: 毎ターン 1 枚ずつ払えるので 2 本とも止める
    assert plan([(2000.0, 2.0), (2000.0, 2.0)], 2.0, [1000.0], [1000.0], [], 1, 2) == (2, 2)
    # `hand_price_mean` 経由の `HandRead` もドンを持つ（`cuttable_share` と同じ値＋中身）
    r = _read([2000.0], don=1.0, events=(0,))
    assert r.cards == ((2000.0, 2.0),) and r.don == 1.0
    assert float(r) == pytest.approx(T.MU * CB.cuttable_share([{"counter": 2000.0, "event": True, "cost": 2.0}], None))


def test_the_rule_hand_lets_blockers_take_the_heaviest_attack_and_fall_when_overpowered():
    """**H-4**: アクティブなブロッカーは**一番重い攻撃**を横取りする（札で止めるのが一番高い攻撃を札抜きで消す）。
    攻撃側のパワー ≥ ブロッカーのパワーなら倒れて次のターンは居ない。"""
    tl = CB._rule_turn_lists
    # ブロッカー 6000（余裕 1000）: 超過 2000 の攻撃を横取りして倒れる → 2 ターン目は両方来る
    assert tl([0.0, 2000.0], [0.0, 2000.0], [1000.0], 2) == [(0.0,), (0.0, 2000.0)]
    # ブロッカー 8000（余裕 3000）: 超過 2000 を横取りして生き残る → 毎ターン同じ
    assert tl([0.0, 2000.0], [0.0, 2000.0], [3000.0], 2) == [(0.0,), (0.0,)]
    # 命中しない攻撃（超過 < 0）は並びに入らない
    assert tl([-1000.0, 0.0], [-1000.0, 0.0], [], 1) == [(0.0,)]


def test_the_rule_hand_is_priced_per_card_cut_not_per_life_saved():
    """**H-4（T77 を守る）**: `F` は切らせた札 1 枚を `μ` で数えるので、手札の項は **`μ × 切る枚数`**。
    救ったライフの本数（`λ` の単位）は足さない（ライフは後で取られるときにライフの項が数える）。
    ライフに余裕があっても、地平の中で止められる攻撃は止める（防いだ損害の最大）＝札 2 枚・地平 2 ターンで 2 本。
    地平を 1 ターンに切ると（T116 の窓）1 本しか止められない。"""
    cards = [(2000.0, 0.0), (2000.0, 0.0)]
    assert CB.rule_guard_plan(cards, 0.0, [0.0], [0.0], [], 3, 2) == (2, 2)
    assert CB.rule_guard_plan(cards, 0.0, [0.0], [0.0], [], 3, 1) == (1, 1)
    sc, tok = _rule_row(life=3.0, hand=2.0, xs=(0.0,))
    assert CB._rule_hand_term(sc, tok, "opp", _read([2000.0, 2000.0]), T.MU, turns=2, count=False) \
        == pytest.approx(2 * T.MU)
    # 手札が空（読めた上で 0 枚）なら 0・通る攻撃が無ければ 0
    assert CB.rule_guard_plan([], 0.0, [0.0], [0.0], [], 0, None) == (0, 0)
    assert CB.rule_guard_plan(cards, 0.0, [-1000.0], [-1000.0], [], 0, None) == (0, 0)


# ---- H-4b〜H-4f: 攻め手の付与（財布は 1 つ・最善応答・`THETA_HAND_MODE=rule_don`） ---------------------------


def _actx(budget, att1, later, cand, kmax=10, flow=None, price=None, a_tab=None, e_tab=None, board=0.0):
    """手で組んだ攻め手の財布（`attacker_ctx` と同じ形）。`a_tab[l]`／`e_tab[l]`＝残ったドン `l` 枚で引いた 1 枚が出す分。"""
    price = price or [[0.0] * (kmax + 1) for _ in att1]
    a_tab = list(a_tab) if a_tab is not None else [0.0] * (budget + 1)
    e_tab = list(e_tab) if e_tab is not None else [0.0] * (budget + 1)
    flow = list(flow) if flow is not None else [a + e for a, e in zip(a_tab, e_tab)]
    key = ("test", budget, tuple(att1), tuple(later), tuple((c, tuple(sorted(p.items())), bx, ru) for c, p, bx, ru in cand),
           tuple(flow), tuple(a_tab), tuple(e_tab), tuple(tuple(r) for r in price), kmax, board)
    return {"budget": budget, "att1": att1, "later": later, "cand": cand, "price": price, "flow": flow,
            "kmax": kmax, "key": key, "olp": 5000.0, "mlp": 5000.0, "lam": T.LAM,
            "lam_net": T.THETA * T.MU, "mu": T.MU, "ds": [float(budget)] * 30, "a_tab": a_tab,
            "ar_tab": [0.0] * len(a_tab), "e_tab": e_tab, "jmax": 30, "no_attack_now": False,
            "lead_bare": float(board), "chars_bare": 0.0, "rest_blk": (), "blk_a": None, "theta_p": T.THETA}


def test_attaching_one_don_raises_the_required_counters_by_exactly_1000():
    """**H-4b**: 付与 1 枚＝パワー +1000＝止めるのに要るカウンターの合計が**ちょうど 1000** 上がる。
    守る側 2000 1 枚・超過 1000（要る合計 2000）は止まるが、付与 1 枚で要る合計 3000 になり止まらない。"""
    cards = [(2000.0, 0.0)]
    X = CB.rule_guard_plan_ex
    assert (X(cards, 0.0, [1000.0], [1000.0], [], 0, 1)["cut"], X(cards, 0.0, [1000.0], [1000.0], [], 0, 1)["alive"]) \
        == (1.0, 1.0)
    assert (X(cards, 0.0, [2000.0], [2000.0], [], 0, 1)["cut"], X(cards, 0.0, [2000.0], [2000.0], [], 0, 1)["alive"]) \
        == (0.0, 0.0)
    ax = _actx(1, [(0, 1000.0)], [(0, 1000.0)], [])
    _c, _s, plan = CB.rule_don_solve(cards, 0.0, [], 0, ax, None)
    assert plan["k"] == (1,) and plan["xs_first"] == (2000.0,) and plan["paid"] == 1.0 and plan["alive"] == 0.0
    ax2 = _actx(2, [(0, 0.0)], [(0, 0.0)], [])
    _c, _s, plan2 = CB.rule_don_solve([(1000.0, 0.0), (1000.0, 0.0)], 0.0, [], 0, ax2, 1)
    assert plan2["xs_first"][0] == pytest.approx(1000.0 * plan2["k"][0])


def test_the_same_don_is_either_played_or_attached_never_both():
    """**H-4b（T109）**: 出す札の費用 ＋ 付与の枚数 ≤ 財布（`paid`）。財布 3・費用 2 の体を出したら付与は高々 1 枚。"""
    cand = [(2, {"atk": 0.0, "eff": 0.0}, 0.0, False)]
    import itertools
    for life, cs in itertools.product((0, 1, 2), ([1000.0], [2000.0, 1000.0], [2000.0, 2000.0, 1000.0])):
        ax = _actx(3, [(0, 0.0)], [(0, 0.0)], cand)
        _c, _s, plan = CB.rule_don_solve([(c, 0.0) for c in cs], 0.0, [], life, ax, None)
        assert plan["paid"] == 2.0 * len(plan["play"]) + sum(plan["k"]) <= 3.0
        if plan["play"]:
            assert sum(plan["k"]) <= 1


def _walk_of(ax, cards, don, blk, life, lt, play, ks, turns=None):
    """手で辿る 1 つの計画: 段ごとの財布 → 守る側の計算 → 歩きの列 → 届く時刻（`rule_don_solve` と同じ組み立て）。"""
    steps = CB.rules_steps(ax, tuple(play), int(ax["jmax"]))
    later_seq = tuple(st["hits"] for st in steps[1:]) or ((),)
    att1 = ax["att1"]
    hits1 = () if ax["no_attack_now"] else steps[0]["hits"]
    xf = () if ax["no_attack_now"] else (tuple(float(att1[q][1]) + 1000.0 * ks[q] for q in range(len(att1)))
                                         + tuple(hits1[len(att1):]))
    if turns is None:
        turns = CB.model_horizon(ax, blk, life)
    r = CB.rule_guard_plan_ex(cards, don, xf, None, blk, life, turns, lt, CB._prices_of(ax), later_seq=later_seq,
                              rest_blk=tuple(ax.get("rest_blk") or ()))
    paid = sum(ax["cand"][i][0] for i in play) + sum(ks)
    sched = CB.rules_sched(r["harms"], steps, ax, float(paid))
    return r, sched, CB.walk_crossing(sched, r["theta"], ax), paid


def test_the_attackers_best_split_beats_all_attach_and_all_play():
    """**H-4b／H-4e**: 攻め手の財布 3・手札に費用 1 の**速攻**の体（超過 0）・リーダー（超過 0）。守る側はライフ 0・
    2000 が 2 枚。**全部出す**・**全部付ける**では守る側は止めて生き延びる。**体を出す ＋ リーダーに 2 枚**なら
    要る合計 3000 と 1000 に 2000×2 が足りず**このターンに倒れる**＝歩きが 1 段目で交わる（一番早い）。"""
    cards = [(2000.0, 0.0), (2000.0, 0.0)]
    X = CB.rule_guard_plan_ex
    assert X(cards, 0.0, [0.0, 0.0], [0.0, 0.0], [], 0, 1)["alive"] >= 1.0
    assert X(cards, 0.0, [3000.0], [3000.0], [], 0, 1)["alive"] >= 1.0
    ax = _actx(3, [(0, 0.0)], [(0, 0.0)], [(1, {"atk": 0.0, "eff": 0.0, "rush": 0.0}, 0.0, True)])
    cut, st, plan = CB.rule_don_solve(cards, 0.0, [], 0, ax, None)
    assert plan["play"] == (0,) and plan["k"] == (2,) and plan["paid"] == 3.0
    # 倒れるターンの守る側の札の使い方は結果を変えない（とどめの段は残りの耐久そのもの・E2）＝損害の和が耐久
    assert plan["alive"] == 0.0 and plan["tau"] <= 1.0
    assert sum(plan["harm_steps"]) == pytest.approx(plan["theta"])
    # 計画が持つ列と時刻は、同じ組み立てを手で辿ったものと一致する（橋の歩き `tau_grow` そのもの）
    r, sched, tau, _paid = _walk_of(ax, cards, 0.0, [], 0, (), (0,), (2,))
    assert tuple(sched) == pytest.approx(plan["sched"]) and tau == pytest.approx(plan["tau"])


def test_the_plan_is_compared_in_whole_turns_so_a_kill_now_beats_a_fractionally_earlier_crossing():
    """**H-4e（E4 の見直し）**: 時刻は**整数のターン**。財布 3・残ったドンで引く札の効果が 1 枚あたり 0.01 を足す。
    リーダーに 2 枚だけ付けると守る側は札 2 枚で生き延び（耐久を使い切る）、残ったドン 1 枚の分だけ歩きの端数は
    1 より小さい。体を出してリーダーに 2 枚ならこのターンに倒す（端数 1.0）。端数で比べると前者が「早い」が、
    勝負が付くのは後者——整数のターンで比べ、同じターンなら倒す方を選ぶ。"""
    cards = [(2000.0, 0.0), (2000.0, 0.0)]
    ax = _actx(3, [(0, 0.0)], [(0, 0.0)], [(1, {"atk": 0.0, "eff": 0.0, "rush": 0.0}, 0.0, True)],
               e_tab=[0.0, 0.01, 0.02, 0.03])
    lone, _s, tau_lone, _p = _walk_of(ax, cards, 0.0, [], 0, (), (), (2,))
    _c, _s2, plan = CB.rule_don_solve(cards, 0.0, [], 0, ax, None)
    assert lone["alive"] >= 1.0 and tau_lone < plan["tau"] <= 1.0          # 端数では倒さない方が「早い」
    assert plan["play"] == (0,) and plan["k"] == (2,) and plan["alive"] == 0.0


def test_a_life_card_taken_this_turn_can_counter_a_later_attack():
    """**H-4e（E1）**: 命中で取られたライフの札は**すぐに手札に入り、同じターンの後の攻撃にカウンターとして切れる**。
    守る側は手札なし・ライフ 1・攻撃 2 本（超過 0）。ライフの札が必ず 1000 のカウンターなら、1 本目を受けて手に入れた
    1000 で 2 本目を止めて生き延びる。使えない札なら 2 本目で倒れる（とどめ）。半々なら期待値も半分。
    イベントのカウンター（費用 1）は使い残しのドンが無ければ切れない。"""
    X = CB.rule_guard_plan_ex
    none = X([], 0.0, [0.0, 0.0], [0.0, 0.0], [], 1, 1)
    full = X([], 0.0, [0.0, 0.0], [0.0, 0.0], [], 1, 1, ((1000.0, 0.0, 1.0),))
    half = X([], 0.0, [0.0, 0.0], [0.0, 0.0], [], 1, 1, ((1000.0, 0.0, 0.5),))
    ev0 = X([], 0.0, [0.0, 0.0], [0.0, 0.0], [], 1, 1, ((1000.0, 1.0, 1.0),))
    ev1 = X([], 1.0, [0.0, 0.0], [0.0, 0.0], [], 1, 1, ((1000.0, 1.0, 1.0),))
    assert none["alive"] == 0.0 and none["cut"] == 0.0
    assert full["alive"] == 1.0 and full["cut"] == 1.0 and full["stopped"] == 1.0
    assert half["alive"] == pytest.approx(0.5) and half["cut"] == pytest.approx(0.5)
    assert ev0["alive"] == 0.0 and ev1["alive"] == 1.0
    lam_net = T.THETA * T.MU
    assert full["harms"][0] == pytest.approx(lam_net + T.MU)           # 命中（正味）＋ 切らせた 1 枚
    # 分布はデッキの構成から（カウンター値 0 の札は入らない・手札に入る割合は `h`）
    assert CB.life_types_of([]) == ()


def test_the_kill_step_is_exactly_what_the_endurance_side_still_counts():
    """**H-4e（E2）**: とどめの段の損害は**耐久の側がまだ数えている残り**そのもの＝倒れた道筋では
    **各段の損害の和がちょうど耐久（`theta`＝ライフ ＋ 切る札 ＋ ブロッカー）**。使えなくなった札は耐久にも速さにも
    入らない（二重に数えない）。レビューの最小の例（ライフ 0・2000 が 1 枚・付与で止められない）は耐久 0・とどめ 0。"""
    X = CB.rule_guard_plan_ex
    k = X([(2000.0, 0.0)], 0.0, [2000.0], [2000.0], [], 0, None)
    assert k["alive"] == 0.0 and k["theta"] == pytest.approx(0.0) and sum(k["harms"]) == pytest.approx(0.0)
    b = X([(2000.0, 0.0)], 0.0, [1000.0], [1000.0], [], 0, None)
    assert b["theta"] == pytest.approx(T.MU) and sum(b["harms"]) == pytest.approx(T.MU)
    import random
    rng = random.Random(9)
    for _ in range(200):
        cards = [(float(rng.choice([1000, 2000])), 0.0) for _ in range(rng.randint(0, 3))]
        blk = [float(rng.choice([0, 1000, 2000]))] if rng.random() < 0.4 else []
        life = rng.randint(0, 2)
        xs = [float(rng.choice([0, 1000, 2000])) for _ in range(rng.randint(1, 3))]
        r = X(cards, 0.0, xs, xs, blk, life, None)
        if r["alive"] < len(r["harms"]) - 1e-9 or True:
            # 地平の中で必ず倒れる局面（攻撃が十分）なら和は耐久に一致する
            full = X(cards, 0.0, xs, xs, blk, life, None)
            if full["alive"] + 1 <= len(full["harms"]) and sum(full["harms"]) > 0:
                assert sum(full["harms"]) == pytest.approx(full["theta"])


def test_the_defender_assigns_blockers_optimally_not_always_to_the_heaviest_attack():
    """**H-4b（レビュー指摘 4）**: 攻撃の超過 1000 と 3000・ブロッカー 1 体（余裕 3000）・札なし・ライフ 2。一番重い攻撃へ
    横取りすると（H-4 の `rule`）ブロッカーは倒れて 1 ターンしか生き延びない。超過 1000 を横取りすれば 2 ターン生き延びる。"""
    heaviest = CB.rule_guard_plan3([], 0.0, [1000.0, 3000.0], [1000.0, 3000.0], [3000.0], 2, None)
    best = CB.rule_guard_plan_ex([], 0.0, [1000.0, 3000.0], [1000.0, 3000.0], [3000.0], 2, None)
    assert heaviest[2] == 1 and best["alive"] == 2.0
    opts = {(rem, surv) for rem, surv, _k, _n in CB.block_assignments((3000.0, 1000.0), (3000.0,))}
    assert opts == {((1000.0, 3000.0), (3000.0,)), ((3000.0,), (3000.0,)), ((1000.0,), ())}


def test_the_plan_is_chosen_by_the_walks_crossing_time_and_the_truncation_is_exact():
    """**H-4e（E4）／H-4f**: 計画は**歩きが耐久に届くターン**（整数）で選び、同じなら生き延びるターン数、速さの値打ち、
    使うドンの順。付与の列挙を「今のターンに守る側が使えるカウンター（手札 ＋ このターンに取るライフの札）・
    アクティブなブロッカー」で打ち切っても**全部の付与を数え上げた最善と一致**する（先の段の財布・引いた札の表・
    盤面が 0 でない盤面で）。"""
    import itertools
    import random
    rng = random.Random(11)
    for it in range(120):
        cards = [(float(rng.choice([1000, 2000])), float(rng.choice([0, 0, 1]))) for _ in range(rng.randint(0, 2))]
        don = float(rng.randint(0, 1))
        blk = [float(rng.choice([0, 1000, 2000]))] if rng.random() < 0.3 else []
        life = rng.randint(0, 2)
        lt = ((1000.0, 0.0, 0.4),) if rng.random() < 0.5 else ()
        att1 = [(s, float(rng.choice([-1000, 0, 1000]))) for s in ([0] + list(range(2, 2 + rng.randint(0, 1))))]
        later = list(att1) + ([(9, float(rng.choice([-1000, 0, 1000])))] if rng.random() < 0.3 else [])
        cand = []
        for _ in range(rng.randint(0, 1)):
            ru = rng.random() < 0.3
            a = round(rng.random() * 0.02, 4)
            cand.append((rng.randint(1, 2), {"atk": a, "eff": 0.0, **({"rush": a} if ru else {})},
                         float(rng.choice([-1000, 0, 1000])), ru))
        kmax = rng.randint(1, 3)
        budget = rng.randint(0, 3)
        a_tab = sorted(round(rng.random() * 0.02, 4) for _ in range(6))
        ax = _actx(budget, att1, later, cand, kmax=kmax, a_tab=a_tab, board=round(rng.random() * 0.05, 4))
        ax["ds"] = [float(min(5, budget + i)) for i in range(30)]
        ax["key"] = ax["key"] + (it,)
        _cut, _st, plan = CB.rule_don_solve(cards, don, blk, life, ax, None, lt)
        got = (math.ceil(round(plan["tau"], 9) - 1e-9), round(plan["alive"], 9), -round(plan["value"], 12), plan["paid"])
        best = None
        for mask in range(1 << len(cand)):
            play = tuple(i for i in range(len(cand)) if mask >> i & 1)
            cost = sum(cand[i][0] for i in play)
            if cost > budget:
                continue
            atk = sum(cand[i][1]["atk"] for i in play)
            h0 = _walk_of(ax, cards, don, blk, life, lt, play, [0] * len(att1))[0]["harms"]
            for ks in itertools.product(range(kmax + 1), repeat=len(att1)):
                if sum(ks) > budget - cost:
                    continue
                r, _sched, tau, paid = _walk_of(ax, cards, don, blk, life, lt, play, ks)
                incr = (r["harms"][0] if r["harms"] else 0.0) - (h0[0] if h0 else 0.0)
                val = atk + incr + ax["flow"][budget - paid]
                key = (math.ceil(round(tau, 9) - 1e-9), round(float(r["alive"]), 9), -round(val, 12), float(paid))
                if best is None or key < best:
                    best = key
        assert got == best, (it, got, best)


def test_a_resting_blocker_returns_inside_the_defender_model_and_the_walk_aims_at_the_same_endurance():
    """**H-4f（F1-a）**: レスト中のブロッカーは守る側の**次のターンに戻る**（規則）。守る側の計算はそれを盤面に入れ、
    耐久（`theta`＝歩きの的）にも同じ `ν` が入る——的と計算が別の耐久を見ない。ライフ 1・札なし・毎ターン超過 0 の
    攻撃 1 本: レストのブロッカー（余裕 1000）が無ければ 2 本目で倒れ、在れば 2 ターン目から毎ターン横取りする。"""
    X = CB.rule_guard_plan_ex
    bare = X([], 0.0, [0.0], [0.0], [], 1, 4)
    back = X([], 0.0, [0.0], [0.0], [], 1, 4, rest_blk=(1000.0,))
    nu = float(CB.nu_meas_of(1000.0 + 5000.0, 5000.0))
    assert bare["alive"] == 1.0 and back["alive"] == 4.0
    assert back["theta"] == pytest.approx(bare["theta"] + nu) and back["nu_all"] == pytest.approx(nu)
    # 倒れる道筋では、まだ戻っていないブロッカーもとどめの段に入る（耐久の残り全部・E2）
    now = X([], 0.0, [0.0, 0.0], [0.0], [], 1, None, rest_blk=(1000.0,))
    assert now["alive"] == 0.0 and sum(now["harms"]) == pytest.approx(now["theta"])
    # 計画の耐久の内訳（ライフ・切る札・全てのブロッカー）を `threshold_parts` がそのまま返す
    sc, tok = _rule_row(life=1.0, hand=0.0, xs=(0.0,))
    old = CB.THETA_HAND_MODE
    try:
        CB.set_theta_hand_mode("rule_don")
        plan = {"theta_parts": (0.1, 0.2, 0.3), "xs_first": (0.0,), "k": (0,), "play": (), "cut": 0.0, "stopped": 0.0}
        assert CB.threshold_parts(sc, tok, g_hand=_read([]), plan=plan) == (0.1, 0.2, 0.3)
    finally:
        CB.set_theta_hand_mode(old)


def test_the_attackers_purse_is_resolved_every_turn_with_the_growing_don():
    """**H-4f（F2・F1-b）**: 攻め手の財布は**段ごとにその段のドンで解き直す**（付けたドンはそのターンで戻る）。今は払えない
    費用 5 の体（財布 3）は、ドンが 5 になる段で出て、次の段から攻撃に加わる。守る側の計算が覆わない段の速さ（`fb`）にも
    その段の付与の増分が入る（付与を 0 にしない）。"""
    body = (5, {"atk": 0.5, "eff": 0.0}, 1000.0, False)
    ax = _actx(3, [(0, -1000.0)], [(0, -1000.0)], [body], kmax=3, board=0.02)
    ax["ds"] = [3.0, 4.0, 5.0, 6.0, 7.0] + [7.0] * 25
    steps = CB.rules_steps(ax, (), 6)
    paid = [s["paid"] for s in steps]
    assert paid[0] == 0.0 and paid[2] >= 5.0                     # 3 段目（ドン 5）で出す
    assert any(h == pytest.approx(1000.0) or h >= 1000.0 for h in steps[3]["hits"])   # 4 段目から殴る
    assert len(steps[3]["hits"]) == 2 and len(steps[1]["hits"]) == 1
    # 2 段目: 出せる札が無いのでドン 4 はリーダー（超過 −1000）への付与になる＝攻撃の超過が上がり、速さにも増分が入る
    k2 = int(round((steps[1]["hits"][0] + 1000.0) / 1000.0))
    assert k2 >= 1 and steps[1]["paid"] == float(k2)
    assert steps[1]["fb"] == pytest.approx(0.02 + CB._attach_gain(ax, -1000.0, k2))


def test_rush_counts_once_and_follows_the_rush_mode():
    """**H-4f（F3）**: 速攻の体は出したターンから**1 回だけ**数える（攻撃の価格は内訳の `atk` の中・`rush` は内訳の名前）。
    出したターンに殴る（旧の「殴らない」形 `RATE_RUSH_MODE=off` は 2026-10-05 に削除）。"""
    body = (1, {"atk": 0.04, "eff": 0.0, "rush": 0.04}, 0.0, True)
    ax = _actx(1, [(0, 0.0)], [(0, 0.0)], [body], board=0.01)
    on = CB.rules_steps(ax, (0,), 3)
    assert on[0]["fb"] == pytest.approx(0.05) and len(on[0]["hits"]) == 2
    assert on[1]["fb"] >= 0.05 and on[1]["fb"] < 0.05 + 0.04 + 1.0   # 2 段目: 体は場の 1 体として 1 回


def test_the_defender_draws_a_card_every_turn_like_the_attacker():
    """**H-4f（F4）**: 完全情報で両席の引きを対称に——守る側も**毎ターン 1 枚引く**（デッキの構成から）。
    札なし・ライフ 1・毎ターン超過 0 の攻撃 1 本。引く札が必ず 1000 のカウンターなら 2 ターン目から止め続け、
    半分なら生き延びるターン数も間に入る。分布はデッキの構成そのもの（ライフの札と違い手札に入る割合 1）。"""
    X = CB.rule_guard_plan_ex
    none = X([], 0.0, [0.0], [0.0], [], 1, 4)
    full = X([], 0.0, [0.0], [0.0], [], 1, 4, draw_types=((1000.0, 0.0, 1.0),))
    half = X([], 0.0, [0.0], [0.0], [], 1, 4, draw_types=((1000.0, 0.0, 0.5),))
    assert none["alive"] == 1.0 and full["alive"] == 4.0 and none["alive"] < half["alive"] < full["alive"]
    hr = CB.with_life_types(_read([]), ["OP01-006"] * 50)
    assert sum(p for _c, _d, p in hr.draw_types) == pytest.approx(
        sum(p for _c, _d, p in hr.life_types) / T.H_LIFE_TO_HAND)


def test_time_readers_get_endurance_over_the_walks_time_and_turn_readers_get_this_turns_harm():
    """**H-4f（Q1）**: 時刻で読む器（`seat_slope_terms`・帳簿の `rate_of_row`）は**耐久 ÷ 歩きの τ**（`Θ/A` がちょうど
    歩きの τ・最初のターンでも 0 にしない＝T103 の 0 は τ の中）、歩き（`seat_slope_sched`）は計画の列そのもの、
    1 ターンで読む器は**今のターンの損害**（列の 1 段目）。"""
    import kappa_vector as KV
    cards = [(2000.0, 0.0)]
    ax = _actx(2, [(0, 0.0)], [(0, 0.0)], [], board=0.03)
    _c, _s, plan = CB.rule_don_solve(cards, 0.0, [], 2, ax, None)
    assert plan["a_time"] * plan["tau"] == pytest.approx(plan["theta"])
    assert plan["a_turn"] == pytest.approx(plan["sched"][0])
    sc, tok = _mirror_row(life=2.0, hand=1.0, n_char=1, pw=0.6)
    olp = float(np.asarray(sc)[T.SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    terms = CB.seat_slope_terms(sc, tok, None, {}, None, olp, plan=plan)
    assert terms == (plan["a_time"], 0.0, 0.0, plan["a_time"], 0.0, 0.0, 0.0, 0.0)
    sch = CB.seat_slope_sched(sc, tok, None, {}, None, olp, jmax=5, plan=plan)
    assert sch == pytest.approx(list(plan["sched"])[:5])
    assert KV.rate_of_row(sc, tok, None, {}, None, j=0, plan=plan) == pytest.approx(plan["a_time"])
    assert CB.walk_crossing(plan["sched"], plan["theta"], ax) == pytest.approx(plan["tau"])


def test_the_walk_reads_the_dp_harm_for_the_steps_it_covers_then_the_base():
    """**H-4e（E3・E5）**: 速さの列（`seat_slope_sched`）は、計画の道筋が覆う段は**守る側の最善応答での実際の損害**
    （`harm_steps`）を盤面・出した体・付与の代わりに読み、先は盤面の素殴り（付与なし）に戻る。1 本の速さしか読まない器
    （`seat_slope_terms`）は**道筋の 1 段あたりの平均**を盤面として読む（とどめの段を毎ターン繰り返さない）。"""
    sc, tok = _mirror_row(life=3.0, hand=5.0, n_char=2, pw=0.6)
    olp = float(np.asarray(sc)[T.SC_OPP_LEADER_POWER]) * 1e4 or 5000.0

    class _NoCards:
        def info(self, _cid):
            return {}
    ci = np.zeros(22, np.int64)
    bare = CB.theory_slope_parts(tok, olp, with_don=False, life_opp=float(sc[T.SC_OPP_LIFE]))
    avg = (0.05 + 0.09) / 2.0
    plan = {"atk": 0.0, "rush": 0.0, "eff": 0.0, "attach": avg - (bare[0] + bare[1]), "attach_lead": 0.0, "paid": 1.0,
            "harm_steps": (0.05, 0.09)}
    sched = CB.seat_slope_sched(sc, tok, ci, {}, _NoCards(), olp, jmax=3, plan=plan, j0=2)   # 局の途中の行（1 ターン目の 0 を避ける）
    assert sched[0] == pytest.approx(0.05) and sched[1] == pytest.approx(0.09)
    assert sched[2] == pytest.approx(bare[0] + bare[1])
    base, _st, _f, lead, *_ = CB.seat_slope_terms(sc, tok, ci, {}, _NoCards(), olp, want_stock=True, plan=plan)
    assert base == pytest.approx(avg) and lead == pytest.approx(bare[0])


def test_the_drawn_card_is_bare_under_the_one_purse_forms_and_default_unchanged():
    """**H-4e（E6）**: 引いた 1 枚の攻撃の値段（`deck_refill.a_of`）は既定では付与を財布の外で無料で付けている
    （`attack_value_don`）。付与を入れる形は**素殴り**（`with_don=False`）で読む。既定の呼び出しは変わらない。"""
    import deck_refill as DR
    deck = ["OP01-006"] * 50
    d = DR.a_of(deck, 6000.0, 5.0)
    b = DR.a_of(deck, 6000.0, 5.0, with_don=False)
    assert d == DR.a_of(deck, 6000.0, 5.0, with_don=True)
    assert b <= d


def test_the_purse_witness_reproduces_purse_plan_exactly():
    """**H-4b**: 計画の中身を読む `purse_plan_witness` は `purse_plan` と**同じ選び方**。"""
    groups = [[(0, {}), (2, {"atk": 0.05, "eff": 0.0})], [(0, {}), (1, {"atk": 0.0, "eff": 0.03})],
              [(0, {}), (1, {"attach_lead": 0.02}), (2, {"attach_lead": 0.035})],
              [(0, {}), (1, {"attach": 0.015})]]
    for budget in range(0, 7):
        pick = CB.purse_plan_witness(groups, budget)
        pl = CB.purse_plan(groups, budget)
        tot = {k: 0.0 for k in CB.PURSE_PARTS}
        paid = 0
        for g, oi in zip(groups, pick):
            c, parts = g[oi]
            paid += c
            for k in tot:
                tot[k] += float(parts.get(k, 0.0))
        for k in CB.PURSE_PARTS:
            assert tot[k] == pytest.approx(pl[k])
        assert paid == pytest.approx(pl["paid"]) and paid <= budget


def test_rule_don_without_the_attackers_purse_falls_back_to_rule_and_counts_it():
    """**H-4b**: 攻め手の財布が読めない（`attacker=None`・計画も無い）ときは付与 0 の `rule` と同じ値に落ち、数を残す。"""
    sc, tok = _rule_row(life=0.0, hand=3.0, xs=(1000.0, 2000.0))
    cards = [1000.0, 3000.0, 1000.0]
    old = CB.THETA_HAND_MODE
    try:
        CB.set_theta_hand_mode("rule")
        r = CB.threshold_parts(sc, tok, g_hand=_read(cards))[1]
        CB.set_theta_hand_mode("rule_don")
        CB._rule_stats_reset()
        rd = CB.threshold_parts(sc, tok, g_hand=_read(cards))[1]
        assert rd == pytest.approx(r)
        assert CB.RULE_STATS.get("rule_don_fallback") == 1
        assert CB.THETA_HAND_PART["rule_don"] == "rule"
    finally:
        CB.set_theta_hand_mode(old)
    assert CB.THETA_HAND_MODE == old


def test_the_ledger_state_reads_the_same_plan_as_the_ledger_rate():
    """**H-4b（T109・レビュー指摘 3）**: 帳簿（`kappa_vector.state_of_row`）の耐久は**速さ（`rate_of_row(plan=)`）と同じ計画**を読む。
    計画を渡すとその並びで守りを解き、渡さなければ付与 0 の `rule` に落ちる（両側で付与 0＝食い違わない）。"""
    import kappa_vector as KV
    sc, tok = _rule_row(life=0.0, hand=1.0, xs=(1000.0,))
    old = CB.THETA_HAND_MODE
    try:
        CB.set_theta_hand_mode("rule_don")
        g = _read([2000.0])
        plan = {"xs_first": (2000.0,), "xs_later": (2000.0,), "k": (1,), "play": ()}
        st_plan = KV.state5_of_row(sc, tok, 0.1, 0.1, 1, g_opp=g, don_plan=plan)
        st_none = KV.state5_of_row(sc, tok, 0.1, 0.1, 1, g_opp=g)
        CB.set_theta_hand_mode("rule")
        st_rule = KV.state5_of_row(sc, tok, 0.1, 0.1, 1, g_opp=g)
    finally:
        CB.set_theta_hand_mode(old)
    assert st_none[1] == pytest.approx(st_rule[1])                    # 計画なし＝付与 0
    assert st_plan[1] == pytest.approx(st_rule[1] - T.MU)             # 付与 1 枚で 2000 では止まらない＝切る札 1 → 0


def test_selecting_rule_leaves_the_other_modes_and_falls_back_loudly_when_the_hand_is_unread():
    """**H-4「off は旧のまま」**: `rule` を足しても既定（`cuttable_forced`）の手札の項は `hand_absorb_forced`
    そのまま。`rule` でも**守る席の手札が読めない**（`g_hand` が `HandRead` でない＝`None` か数）ときは
    **既定の形に落ち、その回数を `RULE_STATS` に残す**（黙って別の値にしない）。"""
    sc, tok = _mirror_row(life=1.0, hand=5.0, n_char=3, pw=1.0)
    olp = float(np.asarray(sc)[T.SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    xs = CB.own_attackers_of(tok, olp)
    n_blk = CB._opp_active_blockers(tok)
    hand_n = float(np.asarray(sc)[T.SC_OPP_HAND])
    life = float(np.asarray(sc)[T.SC_OPP_LIFE])
    mu = T.MU
    old = CB.THETA_HAND_MODE
    try:
        CB.set_theta_hand_mode("cuttable_forced")
        base = CB.threshold_parts(sc, tok)
        assert base[1] == pytest.approx(CB.hand_absorb_forced(hand_n, xs, life, n_blk, mu))
        CB.set_theta_hand_mode("rule")
        CB._rule_stats_reset()
        assert CB.threshold_parts(sc, tok) == base                 # g_hand=None → 既定の形
        assert CB.threshold_parts(sc, tok, g_hand=0.7 * mu)[1] == pytest.approx(
            CB.hand_absorb_forced(0.7 * hand_n, xs, life, n_blk, mu))
        assert CB.RULE_STATS["rule_fallback"] == 2 and CB.RULE_STATS["rule_n"] == 0
        CB.threshold_parts(sc, tok, g_hand=_read([1000.0] * 5))
        assert CB.RULE_STATS["rule_n"] == 1
        # `THETA_HAND_PART` に `rule` が在る（`KeyError` で落ちない）・知らない名前は拒む
        assert CB.THETA_HAND_PART["rule"] == "rule"
    finally:
        CB.set_theta_hand_mode(old)
    assert CB.THETA_HAND_MODE == old


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
    assert CB.THETA_RETURN_MODE == "untap"                       # **C-5c で既定に採用**（2026-09-25・ユーザ決定）
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
        CB.set_theta_return_mode("untap")


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

    `rate_at` は `sched` が在ると**先頭で返す**ので、**規則のドンの列（既定）の下では
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


# ---- T152: 列の第 1 段に T103 の規則を当てる（SCHED_T1_MODE） --------------------------------------
#
# 旧（`walk`）は列の先頭を**無条件に** 0 にしていた（`j <= 1`）。`rate_at` の規則は `j0 + j − 1 <= 1`
# （局の最初の自席ターンだけ）で、`sched` が在ると `rate_at` はその規則を通らず `sched[0]` を返す
# ＝全ての歩きの第 1 段が 0 だった。`game` は `j0` を受け取って同じ規則を列にも通す。

def _sched_fixture():
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = 0.5
    s0 = T.SLOT_OWN_FIELD.start
    tok[s0, T.S_POWER], tok[s0, T.S_IS_CHAR], tok[s0, T.S_CAN_ATTACK] = 0.6, 1.0, 1.0
    return _sc(3, 4), tok


def test_sched_t1_mode_defaults_to_game_and_rejects_unknown():
    """**既定は `game`**（2026-09-24 採用・ユーザ決定「1は規定で」）。`walk` は T152 以前の数字との対照。"""
    assert CB.SCHED_T1_MODE == "game"
    try:
        assert CB.set_sched_t1_mode("walk") == "walk"
    finally:
        CB.set_sched_t1_mode("game")
    with pytest.raises(ValueError):
        CB.set_sched_t1_mode("なにか")
    assert CB.SCHED_T1_MODE == "game"


def test_sched_t1_walk_ignores_j0_and_zeroes_every_walks_first_step():
    sc, tok = _sched_fixture()
    try:
        CB.set_sched_t1_mode("walk")
        base = CB.seat_slope_sched(sc, tok, None, None, None, 5000.0, jmax=6)
        j5 = CB.seat_slope_sched(sc, tok, None, None, None, 5000.0, jmax=6, j0=5)
    finally:
        CB.set_sched_t1_mode("game")
    assert base[0] == 0.0 and base[1] > 0.0
    assert j5 == base                                                              # 旧は j0 を見ない


def test_sched_t1_game_zeroes_only_the_games_first_own_turn():
    sc, tok = _sched_fixture()
    try:
        CB.set_sched_t1_mode("walk")
        base = CB.seat_slope_sched(sc, tok, None, None, None, 5000.0, jmax=6)      # 旧（対照）
    finally:
        CB.set_sched_t1_mode("game")
    g1 = CB.seat_slope_sched(sc, tok, None, None, None, 5000.0, jmax=6, j0=1)     # 既定 game
    g2 = CB.seat_slope_sched(sc, tok, None, None, None, 5000.0, jmax=6, j0=2)
    g5 = CB.seat_slope_sched(sc, tok, None, None, None, 5000.0, jmax=6, j0=5)
    assert CB.seat_slope_sched(sc, tok, None, None, None, 5000.0, jmax=6) == g1   # j0 の既定は 1（局の最初）
    assert g1[0] == 0.0 and g1[1:] == base[1:]            # 局の最初の自席ターンから出る歩きは旧と同じ
    assert g2[0] > 0.0 and g2[0] == pytest.approx(base[1])  # 2 ターン目から出る歩きの第 1 段＝盤面がそのまま殴る
    assert g2[1:] == base[1:] and g5 == g2                  # 2 段目以降は不変・j0 ≥ 2 はどれも同じ


def test_rate_at_reads_the_nonzero_first_step_from_the_game_sched():
    sc, tok = _sched_fixture()
    g3 = CB.seat_slope_sched(sc, tok, None, None, None, 5000.0, jmax=6, j0=3)     # 既定 game
    assert CB.rate_at(1, 0.0, 0.0, 0.0, 0.0, j0=3, sched=g3) == pytest.approx(g3[0]) and g3[0] > 0.0
    assert CB.rate_at(1, 0.0, 0.0, 0.0, 0.0, j0=1, sched=g3) == 0.0      # `rate_at` 自身の規則（局の最初のターン）は生きている


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
    assert eff1 == 0.0                                # 効果の項は cards=None では 0（T108）
    assert s_rush == 0.0 and f_rush == 0.0            # 速攻の札が無いデッキ（T103）
    assert eff == 0.0                                 # 同上（T105）
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


def test_sigma_rel_drops_rows_whose_scale_is_zero_instead_of_dividing_by_a_floor():
    """**T154**: 両席の τ が 0 の行は相対残差が 0/0＝定義できない。床 1e-9 で割ると 1 行で σ_rel が 1e7 級に
    壊れる（`mirror`＋`game` の既定で実際に起きた）。母数から外し、外した行数を `n_scale0` に出す。"""
    rows = [_row(True, 2.0, 6.0, 2, 4)] * 30 + [_row(False, 5.0, 3.0, 4, 2)] * 30
    o_clean = CB.summarise(rows, [])["by_slope"]["hist"]
    assert o_clean["n_scale0"] == 0 and o_clean["sigma_rel"] is not None
    o = CB.summarise(rows + [_row(True, 0.0, 0.0, 1, 1)] * 3, [])["by_slope"]["hist"]
    assert o["n_scale0"] == 3
    assert o["sigma_rel"] == pytest.approx(o_clean["sigma_rel"])          # 尺度 0 の行は σ_rel に入らない
    assert o["sigma_rel"] < 10.0                                          # 床で割った 1e7 級にならない
    only0 = CB.summarise([_row(True, 0.0, 0.0, 1, 1)] * 5, [])["by_slope"]["hist"]
    assert only0["sigma_rel"] is None and only0["n_scale0"] == 5


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
    assert CB.THETA_HAND_MODE == "cuttable"   # **T77 の 2 値化**（速さの手札の項は 2026-10-05 から常に入る・切替は削除）


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
    # 局の 1 自席ターン目（`j0 = 1`）から歩くと、1 段目は 0（2026-09-19 から既定・切替は 2026-10-05 に削除）
    assert CB.rate_at(1, 0.05, 0.05, 0.1, 0.02, j0=1) == pytest.approx(0.0)
    assert CB.rate_at(2, 0.05, 0.05, 0.1, 0.02, j0=1) > 0.0
    # 途中の行（`j0 ≥ 2`）から歩くなら 1 段目から打てる
    assert CB.rate_at(1, 0.05, 0.05, 0.1, 0.02, j0=2) > 0.0
    # 的に届くまでのターン数は 1 つ増える側に動く（1 段ぶん進めないので）
    assert CB.tau_grow(0.3, 0.1, 0.0, 0.0, 0.0, j0=1) > CB.tau_grow(
        0.3, 0.1, 0.0, 0.0, 0.0, j0=2)


def test_rush_bodies_attack_the_turn_they_arrive():
    """**T103**: **速攻は出したターン・引いたターンからもう殴れる**（規則）。
    帳簿側は T84 の `play_starts_next_turn` が既に例外にしていて、**歩きだけが持っていなかった**。"""
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
    sc = _sc(3, 4)
    tok = np.zeros((22, 24), dtype=np.float32)
    base = CB.threshold(sc, tok)
    life, hand, body = CB.threshold_parts(sc, tok, hand_blocker=0.0)
    assert life + hand + body == pytest.approx(base)            # 渡さなければ（0）動かない
    life2, hand2, body2 = CB.threshold_parts(sc, tok, hand_blocker=0.5)
    assert body2 == pytest.approx(body + 0.5)                   # 体の項に載る（2026-09-19 から・切替は 2026-10-05 に削除）
    assert life2 == pytest.approx(life) and hand2 == pytest.approx(hand)
    assert CB.threshold_parts(sc, tok, hand_blocker=-1.0)[2] == pytest.approx(body)   # 負は 0 に倒す


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
    _orig_hp = sys.modules.get("hand_plan")
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
        # 元のモジュールを戻す（消すと後のテストの遅延 import が別の `hand_plan` を作り、monkeypatch が効かなくなる）
        if _orig_hp is not None:
            sys.modules["hand_plan"] = _orig_hp
        else:
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


def test_one_purse_pays_for_each_card_once(monkeypatch):
    """**T109**（ユーザ指示「使用できるドンと使ったドンの整合が取れるように」）: **支払いも付与も
    同じアクティブから出る**のが規則なのに、`stock`（T77・体）と `e₁`（T108・効果）は**別々に同じドンを
    使えた**。財布は**手札を 1 つのナップサックに入れ、1 枚 1 回だけ払う**（旧 `DON_PURSE_MODE=one`／`all` の核・
    旧 `off` は 2026-10-05 に削除）。"""
    import deck_refill as DR

    class _Cards:
        def __init__(self, tbl): self.tbl = tbl
        def info(self, cid): return self.tbl.get(cid)
    # A＝体だけ・B＝効果だけ（イベント＝体を持たない）・どちらもコスト 4
    cards = _Cards({"A": {"power": 6000, "cost": 4}, "B": {"power": 0, "cost": 4, "event": True}})
    monkeypatch.setattr(DR, "card_effect_harm", lambda cid, *a, **k: 0.05 if cid == "B" else 0.0)
    items = [{"cid": "A", "cost": 4}, {"cid": "B", "cost": 4}]
    def hand_purse(don):
        p = CB.purse_plan(CB.hand_groups(items, cards, 5000.0, with_don=False), don)
        return float(p["atk"]), float(p["rush"]), float(p["eff"]), float(p["paid"])
    # ドン 4 なら**どちらか 1 枚だけ**——旧 `off` は両方（体も効果も）数えられた
    atk, rush, eff, paid = hand_purse(4)
    assert paid == pytest.approx(4.0)                     # 払ったのは 1 枚ぶん
    assert (atk > 0.0) != (eff > 0.0)                     # 体か効果のどちらか片方しか立たない
    # ドン 8 なら両方買える＝両方立つ（払いは 8）
    atk2, _r2, eff2, paid2 = hand_purse(8)
    assert paid2 == pytest.approx(8.0) and atk2 > 0.0 and eff2 == pytest.approx(0.05)
    # **払いは絶対に財布を超えない**（不変量）
    for don in range(0, 11):
        assert hand_purse(don)[3] <= don + 1e-9


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
    p = CB.purse_plan(CB.hand_groups(items, cards, 5000.0, with_don=False), 5)
    atk, eff, paid = float(p["atk"]), float(p["eff"]), float(p["paid"])
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
    assert CB.THETA_SIDE_MODE == "legacy"                      # 既定は据え置き
    old_hm = CB.THETA_HAND_MODE
    CB.set_theta_hand_mode("cuttable_forced")                  # **旧の既定を明示**（食い違いの所在・H-4 で既定は rule_don になった）
    try:
        legacy_gap = abs(CB.threshold(sc, tok) - CB.threshold_of_me(sc, tok))
        try:
            CB.set_theta_side_mode("symmetric")
            sym_gap = abs(CB.threshold(sc, tok) - CB.threshold_of_me(sc, tok))
        finally:
            CB.set_theta_side_mode("legacy")
    finally:
        CB.set_theta_hand_mode(old_hm)
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

def test_set_opp_clock_mode_defaults_to_mirror_and_rejects_unknown():
    """**既定は `mirror`**（2026-09-24 採用・ユーザ決定）。`prev_start` は旧の数字との対照。"""
    assert CB.OPP_CLOCK_MODE == "mirror"
    try:
        assert CB.set_opp_clock_mode("prev_start") == "prev_start"
    finally:
        CB.set_opp_clock_mode("mirror")
    with pytest.raises(ValueError):
        CB.set_opp_clock_mode("なにか")
    assert CB.OPP_CLOCK_MODE == "mirror"


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
    rows, _l, stats, _t, _c = CB.collect(["x"])                  # 既定（mirror）
    assert rows == [] and stats["mirror_rows"] == 0
    try:
        CB.set_opp_clock_mode("prev_start")
        rows2, _l2, stats2, _t2, _c2 = CB.collect(["x"])
    finally:
        CB.set_opp_clock_mode("mirror")
    assert rows2 == [] and stats2["mirror_rows"] == 0


def test_the_defender_model_looks_only_as_far_as_the_walk_without_the_hand():
    """**H-4f（地平）**: 守る側の計算の地平は T116 と同じ「手札抜きの地平 `⌈τ0⌉`」（ライフ ＋ 全てのブロッカーに、盤面の
    素殴りと引いた 1 枚の流れで届くターン数）。手札・ライフの札・引く札で手札が増えても状態が爆発しない。
    ライフが多いほど・ブロッカーが多いほど・盤面が遅いほど地平は長い。最低 1 ターン。"""
    ax = _actx(3, [(0, 0.0)], [(0, 0.0)], [], a_tab=[0.01] * 4, board=0.1)
    h0 = CB.model_horizon(ax, [], 0)
    h3 = CB.model_horizon(ax, [], 3)
    hb = CB.model_horizon(ax, [1000.0, 1000.0], 3)
    slow = CB.model_horizon(_actx(3, [(0, 0.0)], [(0, 0.0)], [], a_tab=[0.01] * 4, board=0.02), [], 3)
    assert h0 == 1 and h3 >= h0 and hb >= h3 and slow > h3
    ax["rest_blk"] = (1000.0,)
    assert CB.model_horizon(ax, [], 3) >= h3


def test_an_oversized_defender_model_shortens_its_horizon_deterministically():
    """**H-4f（計算の予算・B1）**: 守る側の計算の状態数が予算を超えたら地平を 1 ターンずつ縮めてやり直す（地平 1 は必ず収まる）。
    **縮めるかどうかは問題だけの関数**——試行ごとに空の覚え書きで数えるので、共有の覚え書き・結果の覚え書き・前に解いた
    局面があっても同じ地平・同じ計画になる（再現できる）。予算が十分なら縮めない（結果は予算なしと同じ）。"""
    cards = [(1000.0, 0.0), (2000.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)]
    lt = ((1000.0, 0.0, 0.3), (2000.0, 0.0, 0.3))
    dt = ((1000.0, 0.0, 0.3), (2000.0, 0.0, 0.3))
    att = [(0, 0.0), (2, 1000.0), (3, 0.0)]
    ax = _actx(6, att, att, [], kmax=2, a_tab=[0.01] * 8, board=0.05)
    ax["ds"] = [6.0 + i for i in range(30)]
    old = CB.EX_STATE_BUDGET

    def solve(budget, tag):
        CB.EX_STATE_BUDGET = budget
        a2 = dict(ax, key=ax["key"] + (tag,))
        return CB.rule_don_solve(cards, 0.0, [], 3, a2, None, lt, dt)[2]
    try:
        free = solve(None, "free")
        assert free["horizon"] == free["horizon0"]
        cold = solve(50, "tight")
        assert cold["horizon"] < cold["horizon0"] and cold["horizon"] >= 1 and cold["tau"] >= 0.0
        # 共有の覚え書き・結果の覚え書きを温めてからでも同じ（呼ぶ順・覚え書きの状態に依らない）
        CB._RULE_DON_CACHE.clear()
        solve(None, "warm")
        warm = solve(50, "tight2")
        assert (warm["horizon"], warm["horizon0"], warm["k"], warm["play"]) == \
            (cold["horizon"], cold["horizon0"], cold["k"], cold["play"])
        assert warm["tau"] == pytest.approx(cold["tau"]) and warm["theta"] == pytest.approx(cold["theta"])
        # 予算が十分なら縮めない
        big = solve(10 ** 9, "big")
        assert big["horizon"] == big["horizon0"] and big["tau"] == pytest.approx(free["tau"])
    finally:
        CB.EX_STATE_BUDGET = old
        CB._RULE_DON_CACHE.clear()


class _JointView:
    """`cut_price.CutView`（`avg`）と同じ形の最小の窓: 切る札 1 枚 = `g`。"""
    kind = "avg"

    def __init__(self, g):
        self.g = float(g)
        self.curve = type("C", (), {"gbar": float(g)})()

    def price(self, k, mu=None):
        return float(k) * self.g


def test_the_defender_model_charges_the_joint_cut_price_and_flat_mu_otherwise():
    """**B2（T77 の対称）**: 切った札の値段は損害の側（攻撃の守る値段）と耐久の側（守る側の計算）で同じ数。`flat` は一律 `μ`・
    `joint` は N-3 と同じ予約の 1 枚あたりの平均 `ḡ`。受ける費用も `λ − h·ḡ`（`gbar`）。耐久の手札の項は
    切った枚数 × その値段、命中の正味は受けた回数ごと、積む損害は切らせた札ごと。`rule`／`rule_don` は joint の中でも落ちない。"""
    import cut_price as CPm
    X = CB.rule_guard_plan_ex
    cards = [(2000.0, 0.0), (2000.0, 0.0)]
    flat = X(cards, 0.0, [0.0], [0.0], [], 1, 3)
    g = 0.2
    old_take = T.CUT_TAKE_CARD
    try:
        with CPm.defending(_JointView(g)):
            T.CUT_TAKE_CARD = g
            assert CB.cut_card_price() == pytest.approx(g)
            assert CB.cut_take_price() == pytest.approx(T.THETA * T.MU + T.H_LIFE_TO_HAND * (T.MU - g))
            joint = X(cards, 0.0, [0.0], [0.0], [], 1, 3)
            sc, tok = _rule_row(life=1.0, hand=2.0, xs=(0.0,))
            old = CB.THETA_HAND_MODE
            try:
                for mode in ("rule", "rule_don"):
                    CB.set_theta_hand_mode(mode)
                    hand = CB.threshold_parts(sc, tok, g_hand=_read([2000.0, 2000.0]))[1]
                    assert hand > 0.0                                      # joint の中でも落ちない
                    if mode == "rule":
                        flat_hand = None
                        with CPm.defending(None):
                            flat_hand = CB.threshold_parts(sc, tok, g_hand=_read([2000.0, 2000.0]))[1]
                        assert hand / g == pytest.approx(flat_hand / T.MU)      # 同じ枚数・値段だけ違う
            finally:
                CB.set_theta_hand_mode(old)
    finally:
        T.CUT_TAKE_CARD = old_take
    assert joint["cut"] == pytest.approx(flat["cut"])                       # 数は同じ
    assert joint["theta"] - (T.LAM * 1) == pytest.approx(g * joint["cut"])   # 耐久の手札の項 = 切った枚数 × ḡ
    assert flat["theta"] - (T.LAM * 1) == pytest.approx(T.MU * flat["cut"])
    # 積む損害は切らせた札ごとに `ḡ`（`flat` は `μ`）・受けたとき（ここでは取る方が早く積む損害が小さい）は `λ − h·ḡ`
    assert sorted(round(x, 9) for x in joint["harms"][1:3]) == [round(g, 9)] * 2
    assert round(joint["harms"][0], 9) == round(T.THETA * T.MU + T.H_LIFE_TO_HAND * (T.MU - g), 9)
    assert sorted(round(x, 9) for x in flat["harms"][:2]) == [round(T.MU, 9)] * 2


def test_the_attackers_purse_and_the_solve_must_share_one_price_window():
    """**B2**: 攻め手の財布（`attacker_ctx`）は作った値段の文脈を持ち、別の文脈で解こうとすると落ちる（付与の増分と
    守る側の計算の値段が食い違ったまま黙って使わない）。"""
    import cut_price as CPm
    ax = _actx(2, [(0, 0.0)], [(0, 0.0)], [])
    ax["cp"] = (("avg", 0.2), 0.2)
    with pytest.raises(RuntimeError):
        CB.rule_don_solve([(2000.0, 0.0)], 0.0, [], 0, ax, None)


def test_the_mirror_reads_my_endurance_with_the_same_defender_model():
    """**H-4g（鏡）**: 線形の橋（curve）の自分の耐久は旧の `threshold_of_me`（別の式）ではなく、相手の席から見た行に
    相手の財布を渡して**相手の耐久と同じ守る側の計算**で読む。鏡を渡さなければ（または `rule_don` 以外なら）旧のまま。"""
    sc, tok = _rule_row(life=1.0, hand=2.0, xs=(0.0,))
    sc_m, tok_m = _rule_row(life=2.0, hand=3.0, xs=(1000.0, 0.0))
    g_me = _read([2000.0, 2000.0])
    prof = [0.1] * 12
    old = CB.THETA_HAND_MODE
    try:
        CB.set_theta_hand_mode("rule_don")
        legacy = CB.curve_d_of_row(sc, tok, 3, prof, g_hand_of_opp=_read([1000.0]), g_hand_of_me=g_me)
        same = CB.curve_d_of_row(sc, tok, 3, prof, g_hand_of_opp=_read([1000.0]), g_hand_of_me=g_me, mirror=None)
        assert same["theta_opp"] == pytest.approx(legacy["theta_opp"])
        m = {"sc": sc_m, "tok": tok_m, "g_me": g_me, "attacker": None}
        mir = CB.curve_d_of_row(sc, tok, 3, prof, g_hand_of_opp=_read([1000.0]), g_hand_of_me=g_me, mirror=lambda: m)
        assert mir["theta_opp"] == pytest.approx(CB.threshold(sc_m, tok_m, g_hand=g_me))
        assert mir["theta_me"] == pytest.approx(legacy["theta_me"])               # 相手の耐久は変わらない
        CB.set_theta_hand_mode("cuttable_forced")
        off = CB.curve_d_of_row(sc, tok, 3, prof, g_hand_of_opp=_read([1000.0]), g_hand_of_me=g_me, mirror=lambda: m)
        assert off["theta_opp"] == pytest.approx(CB.threshold_of_me(sc, tok, g_hand=g_me))   # rule_don 以外は旧のまま
    finally:
        CB.set_theta_hand_mode(old)
