"""**N-3（2026-09-26・ユーザ決定 判断6(a)／判断7(a) の第 2 段）**: 切らせた札の値段＝1 枚 1 役の手札の価値の減り
（`tests/scripts/cut_price.py`）を、損害の側（攻撃の守る値段・実現の損害の手札の部分）と耐久の側（`Θ` の手札の項）へ
**同時に**通す切替 `CUT_PRICE_MODE=joint`（既定は `flat`＝旧の 1 枚一律 `μ`）。

押さえること（期待値は手で計算した値か、独立の総当たり）:

* 手で計算した `L(k)`（切る k 枚の最安の値段）・端数の直線・手札を越えた分は `μ`。
* **一律の値段への還元**: 全部の札が `v = μ` で守りの役が効かないなら `L(k) = kμ`・`Θ` の手札の項・攻撃の値段が旧の値に戻る。
* **望遠鏡の不変量**: 切った組の値段を順に足すと `V(枠) − V(残り)`・最安の組を順に切れば**耐久の予約と損害が厳密に一致**。
* 単調（`L` は減らない・値段は 0 以上・窓の値段は足し算で閉じる）。
* 差し替え口は文脈の中だけ（外では `c(x)·μ` そのもの・相手の体の `ν` は旧の値段のまま）。
* `hand_cut_count × μ` は旧の `hand_absorb_forced` と一致（数を変えていない）・`cuttable_indices` は `cuttable_share` と一致。

**基盤健全性**（`cpu_infra`）。
"""
import itertools
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

import cut_price as CP  # noqa: E402

_SHIPPED_MODE = CP.CUT_PRICE_MODE                  # 収集時の出荷時の既定
_SHIPPED_TAKE = CP.CUT_TAKE_MODE
import crossing_bridge as CB  # noqa: E402
import hand_guard as HG  # noqa: E402
import hand_joint as HJ  # noqa: E402
import theory_order as T  # noqa: E402

MU = T.MU
TAKE = T.THETA * T.MU
S = 1.0 - T.KO_P
BIG = [10, 10, 10, 10]


def _curve(items, xs=(), take=TAKE, caps=BIG, start=1, cand=None, h0=None, cids=None):
    """`items` = [(コスト, v, カウンター)]。切れる札は既定でカウンター > 0 の札。"""
    val = HJ.valuer_of(items, caps, list(xs), take, start=start)
    if cand is None:
        cand = [i for i, it in enumerate(items) if it[2] > 0.0]
    n = len(items)
    return CP.CutCurve(val, cand, n if h0 is None else h0, (len(cand) / n) if n else 0.0, MU, cids=cids)


# ---- 出荷時の既定 ----

def test_shipped_default_is_joint_with_gbar():
    """**ラチェット**（2026-10-01・ユーザ決定「規則どおりの N-3 の読みへ今切り替える」）: 出荷の既定は `joint`（予約の平均の
    値段 `ḡ` を損害と耐久の両側に）＋受けたライフの札も `ḡ`（`gbar`・T77）。旧の `flat`＋`mu` は切替で再現する。
    文脈の外（器が守り手の窓に入る前）では値段の差し替え口は空。"""
    assert _SHIPPED_MODE == "joint"
    assert _SHIPPED_TAKE == "gbar"
    assert "flat" in CP.CUT_PRICE_MODES and "mu" in CP.CUT_TAKE_MODES
    assert T.CUT_PRICER is None
    assert CP.active() is None


# ---- 手で計算した値 ----

def test_L_by_hand_without_guard_role():
    """来る攻撃が無い＝守りの役は 0。V は出す計画だけ（枠は十分・t=0 は割引なし）。
    切れる札 A(v=0.03)・B(v=0.09)、切れない C(v=0.05)。L(1)=0.03・L(2)=0.12・端数・手札を越えた分は μ。"""
    cv = _curve([(1.0, 0.03, 1000.0), (1.0, 0.09, 2000.0), (2.0, 0.05, 0.0)])
    L = cv.L()
    assert L == pytest.approx([0.0, 0.03, 0.12], abs=1e-12)
    assert cv.Lx(1.5) == pytest.approx(0.03 + 0.5 * 0.09, abs=1e-12)
    assert cv.Lx(3.0) == pytest.approx(0.12 + MU, abs=1e-12)
    assert cv.view(kind="slice").price(1.28) == pytest.approx(0.03 + 0.28 * 0.09, abs=1e-12)
    # m = 1 枚ぶん消費した後の 1 枚＝2 枚目の値段
    assert CP.CutView(cv, 1.0, "slice").price(1.0) == pytest.approx(0.09, abs=1e-12)
    # 予約の平均（`joint`）: 予約 2 枚なら ḡ = 0.12 / 2・予約 1.5 枚なら Lx(1.5) / 1.5・予約が無ければ切れる札全部の平均
    cv.reserve = 2.0
    assert cv.gbar == pytest.approx(0.06, abs=1e-12)
    assert cv.view(kind="avg").price(1.28) == pytest.approx(1.28 * 0.06, abs=1e-12)
    cv.reserve = 1.5
    assert cv.gbar == pytest.approx((0.03 + 0.5 * 0.09) / 1.5, abs=1e-12)
    cv.reserve = 0.0
    assert cv.gbar == pytest.approx(0.06, abs=1e-12)
    assert _curve([(1.0, 0.05, 0.0)]).gbar == MU                  # 切れる札が無い＝旧の値段


def test_L_by_hand_with_guard_role():
    """カウンター 1000 の札（v=0.03）が 1 枚・超過 0 の攻撃が毎ターン来る（地平 2・最初の相手ターンは 1 ラウンド割引）。
    守りに回すと s·受ける損 = 0.711 × 0.0872 = 0.0620 > 出す 0.03 ⇒ V = 0.0620 = L(1)。
    同じ札 2 枚なら V = s·受ける損 + s²·受ける損 = 0.1060、1 枚残すと 0.0620 ⇒ L(1) = s²·受ける損 = 0.0441・L(2) = 0.1060
    （2 枚目を切る方が高い＝最後の守りほど価値がある）。"""
    assert HG.GUARD_TURNS == 2
    one = _curve([(1.0, 0.03, 1000.0)], xs=[0.0])
    assert one.L() == pytest.approx([0.0, S * TAKE], abs=1e-12)
    two = _curve([(1.0, 0.03, 1000.0), (1.0, 0.03, 1000.0)], xs=[0.0])
    assert two.L() == pytest.approx([0.0, S * S * TAKE, S * TAKE + S * S * TAKE], abs=1e-12)


# ---- 一律の値段への還元 ----

@pytest.mark.parametrize("n", [1, 3, 6])
def test_flat_cards_recover_mu(n):
    cv = _curve([(0.0, MU, 1000.0)] * n)
    assert cv.L() == pytest.approx([k * MU for k in range(n + 1)], abs=1e-12)
    for m in (0.0, 0.5, 1.0):
        for k in (0.3, 1.0, 1.28, 2.25, n + 1.5):
            assert CP.CutView(cv, m, "slice").price(k) == pytest.approx(k * MU, abs=1e-12)
    for res in (None, 0.0, 1.0, 1.28 * 2, float(n)):
        cv.reserve = res
        assert cv.gbar == pytest.approx(MU, abs=1e-15)
        assert CP.CutView(cv, 0.0, "avg").price(2.25) == pytest.approx(2.25 * MU, abs=1e-15)


def test_flat_view_leaves_attack_prices_unchanged():
    """一律の値段の窓の中では攻撃・ブロック・付与の値段が旧と同じ（差し替え口が数 `c(x)` を変えていない）。"""
    cases = [(6000.0, 5000.0, True), (5000.0, 5000.0, True), (8000.0, 5000.0, True), (4000.0, 5000.0, True),
             (7000.0, 4000.0, False)]
    for p, tp, lead in cases:
        base = T.attack_value(p, tp, lead, nu_target=0.07, blockers=[(5000.0, 0.05)])
        att = T.attach_value(p, tp, 2)
        with CP.defending(CP.FlatView(MU)):
            assert T.attack_value(p, tp, lead, nu_target=0.07, blockers=[(5000.0, 0.05)]) == base
            assert T.attach_value(p, tp, 2) == pytest.approx(att, abs=1e-15)
    assert T.CUT_PRICER is None


def test_theta_hand_count_is_unchanged():
    """`hand_cut_count × μ` は旧の手札の項（`hand_absorb_forced`）と一致＝N-3 は数を変えず値段だけを変える。
    （波C: 数え方は `cuttable_forced` の 1 つだけ——`cuttable`／`cuttable_cx`／`cuttable_seq` は削除。）"""
    rng = np.random.default_rng(3)
    old = CB.THETA_HAND_MODE
    try:
        for mode in ("cuttable_forced", "rule_don"):           # どちらの手札の形でも数は `cuttable_forced`
            CB.set_theta_hand_mode(mode)
            for _ in range(200):
                hand_n = float(rng.integers(0, 9))
                g = MU * float(rng.choice([0.0, 0.25, 0.5, 0.6, 1.0]))
                xs = [float(x) for x in rng.choice([-1000.0, 0.0, 1000.0, 2000.0, 4000.0], size=int(rng.integers(0, 5)))]
                life = float(rng.integers(0, 5)); blk = int(rng.integers(0, 3))
                n_cut = (g / MU) * hand_n
                want = CB.hand_absorb_forced(n_cut, xs, life, blk, MU)
                got = CB.hand_cut_count(g, hand_n, xs, life, blk, MU) * MU
                assert got == pytest.approx(want, abs=1e-12)
        xs = [0.0, 1000.0, 2000.0, 5000.0]
        for blk in (0, 1, 3):
            assert CB.shield_count_of(xs, blk) * MU == pytest.approx(CB.shield_rate_of(xs, blk), abs=1e-12)
    finally:
        CB.set_theta_hand_mode(old)


def test_cuttable_indices_match_cuttable_share():
    rng = np.random.default_rng(5)
    for _ in range(300):
        n = int(rng.integers(1, 8))
        items = [{"counter": float(rng.choice([0.0, 1000.0, 2000.0])), "event": bool(rng.random() < 0.3),
                  "cost": float(rng.integers(0, 4))} for _ in range(n)]
        for don in (None, 0.0, 1.0, 3.0):
            assert len(CP.cuttable_indices(items, don)) / n == pytest.approx(CB.cuttable_share(items, don), abs=1e-12)


# ---- 望遠鏡・単調 ----

def _rand_items(rng, n):
    return [(float(rng.integers(0, 6)), float(rng.choice([0.0, 0.01, 0.03, 0.05, 0.09])),
             float(rng.choice([0.0, 1000.0, 1000.0, 2000.0]))) for _ in range(n)]


def test_telescoping_and_endurance_equals_booked_harm():
    """**不変量**: 枠の札を順に切る列の値段の和は `V(枠) − V(残り)`（厳密）。耐久の予約 `L(k)` の最安の組 `T*` を
    どんな順・どんな分け方で切っても、切らせた損害の和は `L(k)` に厳密に一致する。高い組を切れば損害は予約以上。"""
    rng = np.random.default_rng(11)
    for _ in range(60):
        items = _rand_items(rng, int(rng.integers(2, 7)))
        xs = [float(x) for x in rng.choice([0.0, 1000.0, 2000.0], size=int(rng.integers(0, 3)))]
        cv = _curve(items, xs=xs)
        if cv.n0 == 0:
            continue
        full = cv.full()
        L = cv.L()
        for k in range(1, cv.n0 + 1):
            # 独立の総当たりで最安の組
            best, arg = math.inf, None
            for T_ in itertools.combinations(cv.cand, k):
                loss = cv.set_loss(full, T_)
                if loss < best:
                    best, arg = loss, T_
            assert L[k] == pytest.approx(best, abs=1e-12)
            # 最安の組を順に（1 枚ずつ／まとめて）切る
            for split in range(1, k + 1):
                keep, booked = full, 0.0
                parts = [arg[:split], arg[split:]]
                for P in parts:
                    booked += cv.set_loss(keep, P)
                    keep = keep - frozenset(P)
                assert booked == pytest.approx(L[k], abs=1e-12)
            # 別の組を切れば損害は予約以上
            for T_ in itertools.combinations(cv.cand, k):
                assert cv.set_loss(full, T_) >= L[k] - 1e-12
        assert all(L[i + 1] >= L[i] - 1e-15 for i in range(len(L) - 1))
        v = cv.view(kind="slice")
        for a, b in ((0.5, 0.7), (1.0, 1.28), (0.2, 3.0)):
            assert v.price(a) + CP.CutView(cv, a, "slice").price(b) == pytest.approx(v.price(a + b), abs=1e-12)
            assert v.price(a) >= 0.0
        # 予約の平均: 予約 N 枚ぶんの攻撃の値段の和（1 本 c 枚をどう分けても）＝耐久の予約 L(N)
        for N in range(1, cv.n0 + 1):
            cv.reserve = float(N)
            va = cv.view(kind="avg")
            cs = [0.5, 1.28, float(N) - 1.78] if N > 1.78 else [float(N)]
            assert sum(va.price(c) for c in cs) == pytest.approx(L[N], abs=1e-12)
            assert va.price(float(N)) == pytest.approx(L[N], abs=1e-12)


def test_realised_corrections_telescope_over_snapshots():
    """実現の直し: 枠の札が応答ごとに減る列（途中でライフの札が入って切られる）を読んで、
    直し ＋ μ × 減った枚数 の和 = 枠の札の `V(枠) − V(残り)` ＋ μ × 枠に無い札の減り。位置は減らした応答の行。"""
    items = [(1.0, 0.03, 1000.0), (1.0, 0.09, 2000.0), (2.0, 0.05, 0.0), (1.0, 0.03, 1000.0)]
    cids = ["A", "B", "C", "A"]
    cv = _curve(items, xs=[0.0, 1000.0], cids=cids)
    frame = ["A", "B", "C", "A"]
    snaps = [(10, ["A", "B", "C", "A"]),          # 最初の窓（まだ何も減っていない）
             (12, ["B", "C", "A"]),               # 10 の応答で A を 1 枚切った
             (15, ["C", "A", "L"]),               # 12 の応答で B を切り、ライフの札 L が入った
             (17, ["C", "A"])]                    # 15 の応答で L を切った（枠に無い札＝μ のまま）
    final = ["C", "D"]                            # 17 の応答で A を切った・D は次の自席ターンの引き
    corr = CP.realised_corrections(cv, frame, snaps, final, MU)
    assert [p for p, _c in corr] == [10, 12, 15, 17]
    full = cv.full()
    kA1 = frozenset([0])
    v1 = cv.set_loss(full, kA1)
    v2 = cv.set_loss(full - kA1, [1])
    v4 = cv.set_loss(full - kA1 - frozenset([1]), [3])
    assert [c for _p, c in corr] == pytest.approx([v1 - MU, v2 - MU, 0.0, v4 - MU], abs=1e-12)
    total = sum(c for _p, c in corr) + MU * 4
    want = (cv.valuer.value(full)[0] - cv.valuer.value(frozenset([2]))[0]) + MU * 1
    assert total == pytest.approx(want, abs=1e-12)
    assert CP.sum_in(corr, 9, 13) == pytest.approx(v1 - MU + v2 - MU, abs=1e-12)
    # 一律の札なら直しは 0
    flat = _curve([(0.0, MU, 1000.0)] * 3, cids=["A", "B", "C"])
    corr0 = CP.realised_corrections(flat, ["A", "B", "C"], [(3, ["A", "B", "C"]), (5, ["C"])], ["C", "X"], MU)
    assert all(abs(c) < 1e-12 for _p, c in corr0)
    # 次の自席ターンが無い（局の終わり）なら最後の応答の後は読まない
    corr1 = CP.realised_corrections(cv, frame, snaps, None, MU)
    assert [p for p, _c in corr1] == [10, 12, 15]


# ---- 文脈 ----

def test_context_is_scoped_and_nested():
    cv = _curve([(1.0, 0.03, 1000.0), (1.0, 0.09, 2000.0)])
    x = 1000.0                                      # c(x) = 1.28 枚
    flat = T.attack_value(5000.0 + x, 5000.0, True)
    with CP.defending(cv.view(kind="avg")):
        assert T.attack_value(5000.0 + x, 5000.0, True) == pytest.approx(min(cv.gbar * T.c_of(x), TAKE), abs=1e-12)
    with CP.defending(cv.view(kind="slice")):
        joint = T.attack_value(5000.0 + x, 5000.0, True)
        assert joint == pytest.approx(min(cv.Lx(T.c_of(x)), TAKE), abs=1e-12)
        with CP.defending(None):
            assert T.attack_value(5000.0 + x, 5000.0, True) == flat
        assert T.attack_value(5000.0 + x, 5000.0, True) == joint
        # 付与の値段は攻撃の値段の増分と同じ値段で
        assert T.attach_value(5000.0, 5000.0, 1) == pytest.approx(
            T.attack_value(6000.0, 5000.0, True) - T.attack_value(5000.0, 5000.0, True), abs=1e-12)
    assert T.CUT_PRICER is None and CP.active() is None
    assert T.attack_value(5000.0 + x, 5000.0, True) == flat


def test_other_side_body_value_stays_flat():
    """相手の体の `ν`（こちらを殴る攻撃の値）は守り手が逆の席なので旧の値段のまま。"""
    cv = _curve([(1.0, 0.0, 1000.0)])
    flat = T.nu_of(6000.0, 5000.0, 3.0)
    with CP.defending(cv.view()):
        assert T._nu_of_other_side(6000.0, 5000.0, 3.0) == flat
        assert T.CUT_PRICER is not None


def test_threshold_parts_use_count_times_price():
    """`Θ` の手札の項は文脈の中で `view.price(切る枚数)`・一律の窓なら旧の値に戻る。"""
    tok = np.zeros((24, 40), dtype=np.float32)
    sc = np.zeros(16, dtype=np.float32)
    sc[T.SC_OPP_LIFE] = 2.0; sc[T.SC_OPP_HAND] = 5.0
    sc[T.SC_MY_LEADER_POWER] = 0.5; sc[T.SC_OPP_LEADER_POWER] = 0.5
    tok[0, T.S_POWER] = 0.6                         # 自分のリーダー 6000 ＝ 超過 1000 の攻撃
    old = CB.THETA_HAND_MODE
    try:
        CB.set_theta_hand_mode("cuttable_forced")
        g = MU * 0.6
        base = CB.threshold_parts(sc, tok, g_hand=g)
        with CP.defending(CP.FlatView(MU)):
            flat = CB.threshold_parts(sc, tok, g_hand=g)
        assert flat == pytest.approx(base, abs=1e-15)
        cv = _curve([(1.0, 0.03, 1000.0), (1.0, 0.09, 2000.0), (1.0, 0.02, 1000.0)])
        n = CB.hand_cut_count(g, 5.0, CB.own_attackers_of(tok, 5000.0), 2.0, 0, MU)
        with CP.defending(cv.view(kind="slice")):
            got = CB.threshold_parts(sc, tok, g_hand=g)
        assert got[0] == base[0] and got[2] == base[2]
        assert got[1] == pytest.approx(cv.Lx(n), abs=1e-12)
        cv.reserve = n
        with CP.defending(cv.view(kind="avg")):
            got = CB.threshold_parts(sc, tok, g_hand=g)
        assert got[1] == pytest.approx(cv.Lx(n), abs=1e-12)          # 予約そのものなら ḡ × N = L(N)
    finally:
        CB.set_theta_hand_mode(old)


def test_value_caches_are_keyed_by_the_pricing_context():
    """覚えておく値（選択肢の価値・デッキの流入）は**値段の文脈を鍵に入れて**覚える——守り手ごとに値が違うので、
    鍵に入れないと別の守り手・旧の値段の読みへ漏れる（実測で守りの窓の判断が 2 行ずれた）。安い順の切れ目の窓は覚えない。"""
    import deck_refill as DR
    old_take = CP.CUT_TAKE_MODE
    CP.set_cut_take_mode("mu")                     # 旧の代数で鍵だけを見る（受けたライフの札の値段は別の切替）
    try:
        _check_value_caches(DR)
    finally:
        CP.set_cut_take_mode(old_take)


def _check_value_caches(DR):
    deck = ("OP01-013", "OP01-016", "OP01-025", "ST01-012")
    cheap = _curve([(1.0, 0.0, 1000.0), (1.0, 0.0, 1000.0)])       # 来る攻撃なし・出す価値なし＝切っても何も失わない
    cheap.reserve = 1.0
    assert cheap.gbar == 0.0
    dear = _curve([(1.0, 0.3, 1000.0)])
    dear.reserve = 1.0
    T._OPTION_CACHE.clear(); DR._FLOW.clear()
    base_o, base_f = T.option_value(6000.0, 5000.0, 3.0), DR.a_of(deck, 5000.0)
    with CP.defending(cheap.view(kind="avg")):
        o1, f1 = T.option_value(6000.0, 5000.0, 3.0), DR.a_of(deck, 5000.0)
    with CP.defending(dear.view(kind="avg")):
        o2, f2 = T.option_value(6000.0, 5000.0, 3.0), DR.a_of(deck, 5000.0)
    assert f1 < base_f < f2 or f1 < f2                     # 値段で流入の攻撃の値が動く
    # 覚えた値を読み直しても同じ（鍵が違う＝混ざらない）・外では旧の値のまま
    with CP.defending(cheap.view(kind="avg")):
        assert (T.option_value(6000.0, 5000.0, 3.0), DR.a_of(deck, 5000.0)) == (o1, f1)
    with CP.defending(dear.view(kind="avg")):
        assert (T.option_value(6000.0, 5000.0, 3.0), DR.a_of(deck, 5000.0)) == (o2, f2)
    assert (T.option_value(6000.0, 5000.0, 3.0), DR.a_of(deck, 5000.0)) == (base_o, base_f)
    # 切れ目の窓（1 枚あたり一定でない）は覚えない
    n_f = len(DR._FLOW)
    with CP.defending(dear.view(kind="slice")):
        DR.a_of(deck, 4000.0)
    assert len(DR._FLOW) == n_f


def test_search_gain_cache_is_keyed_by_the_pricing_context(monkeypatch):
    """探す値（`search_price.card_gain`）も同じ規約（鍵に値段の文脈・切れ目の窓では覚えない）。"""
    import search_price as SP
    calls = []
    monkeypatch.setattr(SP, "_card_body", lambda cid, cards: {"cost": 1.0, "counter": 1000.0})
    monkeypatch.setattr(SP, "use_value", lambda *a, **k: 0.05, raising=False)

    class _C:
        def info(self, cid):
            return {"cost": 1}

    def fake_deltas(items, card, caps, xs, take):
        calls.append(T.CUT_PRICER_KEY)
        return {"dtotal": 0.01 if T.CUT_PRICER is None else T.CUT_PRICER(1.0, MU)}
    monkeypatch.setattr(SP.HP, "card_deltas", fake_deltas)
    ctx = {"hand_items": [], "caps": [1, 1], "xs": [], "take": 0.1, "olp": 5000.0, "r": 3.0}
    SP._GAIN.clear()
    base = SP.card_gain("X", ctx, _C())
    dear = _curve([(1.0, 0.3, 1000.0)]); dear.reserve = 1.0
    with CP.defending(dear.view(kind="avg")):
        inside = SP.card_gain("X", ctx, _C())
    assert inside == pytest.approx(dear.gbar) and base == 0.01
    assert SP.card_gain("X", ctx, _C()) == 0.01            # 外へ漏れない
    n = len(SP._GAIN)
    with CP.defending(dear.view(kind="slice")):
        SP.card_gain("X", ctx, _C())
    assert len(SP._GAIN) == n


def test_other_side_turns_off_both_the_price_hook_and_the_hand_term_context():
    """**B3**: 相手の体の値（逆の席）を読む間は、攻撃の値段の差し替え口と手札の項の文脈（`active()`）が**両方**旧に戻る。"""
    cv = _curve([(1.0, 0.03, 1000.0)]); cv.reserve = 1.0
    seen = []
    orig = T.nu_of

    def spy(*a, **k):
        seen.append((T.CUT_PRICER, CP.active(), T.CUT_PRICER_KEY, T.CUT_TAKE_CARD))
        return orig(*a, **k)
    T.nu_of = spy
    try:
        with CP.defending(cv.view(kind="avg")):
            T._nu_of_other_side(6000.0, 5000.0, 3.0)
            assert CP.active() is not None and T.CUT_PRICER is not None
    finally:
        T.nu_of = orig
    assert seen and all(x == (None, None, None, None) for x in seen)     # 潜在価値（既定）の中の `nu_of` も全部
    # 攻撃の流れの中の相手の体の `ν`・`opp_bodies_of` も逆の席として旧の値段
    tok = np.zeros((24, 40), dtype=np.float32)
    tok[7, T.S_IS_CHAR] = 1.0; tok[7, T.S_POWER] = 0.6
    flat = T.opp_bodies_of(tok, 5000.0, 3.0)
    flat_s = T.attack_stream(7000.0, 5000.0, 3.0, opp_chars=[(6000.0, False)])
    with CP.defending(cv.view(kind="avg")):
        assert T.opp_bodies_of(tok, 5000.0, 3.0)[0]["nu"] == flat[0]["nu"]
        T.attack_stream(7000.0, 5000.0, 3.0, opp_chars=[(6000.0, False)])   # 走ること（値は守り手の値段で変わってよい）
    assert T.attack_stream(7000.0, 5000.0, 3.0, opp_chars=[(6000.0, False)]) == flat_s


def test_value_and_reserve_read_the_same_attackers_with_the_rules_power():
    """**B1**: 枠は守り手の自席ターンの行＝トークンの自分のリーダーには自分が付けたドン（自分のターンだけ +1000）が乗る。
    V（来る攻撃）と予約 `N_f` は同じ、相手のターンの規則どおりのパワー（付与ドン無し＝`SC_MY_LEADER_POWER`）で読む。"""
    sc = np.zeros(16, dtype=np.float32)
    sc[T.SC_MY_LEADER_POWER] = 0.5; sc[T.SC_OPP_LEADER_POWER] = 0.5; sc[T.SC_MY_LIFE] = 3.0; sc[T.SC_MY_HAND] = 0.0
    tok = np.zeros((24, 40), dtype=np.float32)
    tok[0, T.S_POWER] = 0.7                          # 自分のリーダー: 5000 ＋ 付与ドン 2 枚（自分のターンだけ）
    tok[1, T.S_POWER] = 0.5                          # 相手のリーダー 5000
    tok[7, T.S_IS_CHAR] = 1.0; tok[7, T.S_POWER] = 0.6
    xs = CP.defender_incoming(sc, tok)
    assert xs == pytest.approx([1000.0, 0.0], abs=1e-3)  # トークンのまま読むと [−1000, −2000]＝全部通らない（旧の誤り）
    assert sorted(x for x in CP.defender_attackers(sc, tok)) == pytest.approx([0.0, 1000.0], abs=1e-3)
    assert sorted(HG.incoming(tok)) == []            # 旧の読み（守りの窓の行では正しい・枠の行では誤り）


# ---- 残り 2a（2026-10-01）: 枠の守る側のパワーは次の相手ターンの規則どおりの値 ----

_REC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "f_identity", "rec")


def _vocab_maps():
    import guard_afford as GA
    v = GA._vocab()
    return v, {i: c for c, i in v.items()}


def test_leader_power_opp_turn_strips_don_and_swaps_turn_passives():
    """自席のターンの行: 付与ドン（列 0 にだけ載る）を外し、リーダー自身の【相手のターン中】の上昇を足す。
    ST09-001＝【ドン!!×1】【相手のターン中】自分のライフが 2 枚以下ならパワー +1000（印字 5000）。"""
    v, idx2cid = _vocab_maps()
    sc = np.zeros(127, dtype=np.float32)
    sc[T.SC_MY_LEADER_POWER] = 0.5; sc[T.SC_OPP_LEADER_POWER] = 0.5; sc[T.SC_MY_LIFE] = 2.0; sc[T.SC_OPP_LIFE] = 4.0
    sc[T.SC_IS_MY_TURN] = 1.0
    tok = np.zeros((22, 22), dtype=np.float32)
    tok[0, T.S_POWER] = 0.7; tok[0, T.S_POWER_OPP_TURN] = 0.5; tok[0, T.S_ATTACHED_DON] = 0.4   # 付与ドン 2 枚
    ci = np.zeros(24, dtype=np.int16); ci[0] = v["ST09-001"]; ci[1] = v["OP09-001"]
    assert T.leader_power_opp_turn(tok, sc, ci, idx2cid) == pytest.approx(6000.0)
    assert T.leader_power_opp_turn(tok, sc) == pytest.approx(5000.0)            # 札が読めなければ付与ドンだけ外す
    sc[T.SC_MY_LIFE] = 3.0
    assert T.leader_power_opp_turn(tok, sc, ci, idx2cid) == pytest.approx(5000.0)   # 条件が偽
    sc[T.SC_MY_LIFE] = 2.0; tok[0, T.S_ATTACHED_DON] = 0.0
    assert T.leader_power_opp_turn(tok, sc, ci, idx2cid) == pytest.approx(5000.0)   # 【ドン!!×1】が付いていない
    sc[T.SC_IS_MY_TURN] = 0.0; tok[0, T.S_POWER] = 0.6; tok[0, T.S_POWER_OPP_TURN] = 0.6
    assert T.leader_power_opp_turn(tok, sc, ci, idx2cid) == pytest.approx(6000.0)   # 相手のターンの行はそのまま
    # 旧の読み（列 0）は自席の行で付与ドンを載せる＝来る攻撃の超過が 2000 小さく見えていた
    tok[0, T.S_POWER] = 0.7; tok[0, T.S_POWER_OPP_TURN] = 0.5; sc[T.SC_IS_MY_TURN] = 1.0
    tok[1, T.S_POWER] = 0.6
    assert T.incoming_x(tok) == pytest.approx([-1000.0])
    assert T.incoming_x(tok, mine=T.leader_power_opp_turn(tok, sc)) == pytest.approx([1000.0])


def _fixture_frames():
    from opcg_sim.learned.train import plan_labels as PL
    import theory_bridge as TB
    from theory_bridge import POL_COLS, ROW_COLS, _extra
    _v, idx2cid = _vocab_maps()
    cards = PL.Cards()
    for rows, pol, ex, L, ptr, idx in PL.iter_games([_REC], row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        own_last = {}
        for i in idx:
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t >= 1 and PL.is_own_turn(w, t) and TB.is_decision_row(rows, pol, L, ptr, i):
                own_last[(w, t)] = i
        yield rows, ex, list(idx), idx2cid, cards, own_last


def test_end_of_turn_frames_read_the_next_opponent_turn_leader_power():
    """ターン末の枠（`end_of_turn=True`）は守る側のパワーを**次の相手ターンの最初の行**（攻め手の行の相手のリーダー＝
    付与ドン無し・このターンだけの増減が切れた後）から読む。V の来る攻撃と予約 `N_f` は同じ値で読む。次の行が無い枠と
    ターン頭の枠は行から規則で読む（`leader_power_opp_turn`）。実デッキの記録 2 局。"""
    n_next = n_rule = 0
    for rows, ex, order, idx2cid, cards, own_last in _fixture_frames():
        stats = {}
        cf = CP.CutFrames(order, rows, ex, idx2cid, cards, own_last, MU, don_rule=True, stats=stats, end_of_turn=True)
        cf0 = CP.CutFrames(order, rows, ex, idx2cid, cards, own_last, MU, don_rule=True, end_of_turn=False)
        for (d, t), i in own_last.items():
            cv = cf.curve(d, t)
            if cv is None:
                continue
            j = next((k for k in order if int(rows["who"][k]) == 1 - d and int(rows["turn"][k]) == t + 1), None)
            sc, tok = np.asarray(ex["sc"][i]), np.asarray(ex["tok"][i])
            rule = T.leader_power_opp_turn(tok, sc, ex["ci"][i], idx2cid)
            if j is not None:
                want = float(np.asarray(ex["sc"][j])[T.SC_OPP_LEADER_POWER]) * 1e4
                n_next += 1
            else:
                want = rule
                n_rule += 1
            assert cv.mlp == pytest.approx(want)
            assert cf0.curve(d, t).mlp == pytest.approx(rule)
            # V の来る攻撃と予約は同じ並び（同じパワー）で読む
            xs = sorted((float(x) for x in CB.opp_attackers_of(tok, cv.mlp) if float(x) >= -T.PWR_EPS), reverse=True)
            assert cv.xs_future == pytest.approx(xs)
            assert cv.reserve == pytest.approx(CP.reserve_of_row(sc, tok, cv.share, MU, mlp=cv.mlp))
        assert stats.get("cut_mlp_next", 0) + stats.get("cut_mlp_rule", 0) == stats.get("cut_frames", 0)
    assert n_next >= 10


# ---- 2026-10-01 の点検: 守りの窓で窓が先読みしない ----

def test_guard_rows_read_no_frame_after_the_row():
    """守りの窓（守り手 `w` の行・攻め手 `1 − w` のターン `t` の途中）で攻め手の値段の窓を引くとき、
    **攻め手のターン末の枠**（`end_of_turn=True`）はそのターンの最後の行と次のターンの最初の行＝この行より後ろを読む（先読み）。
    **今のターンの最初の行**（`theory_bridge` の `first_main`・T79 の相手の行と同じ）は普通この行より前で、後ろになる稀な窓
    （攻め手の最初の main の行より前に守りの窓が来る）は `frame_key(at_n=)` が前の枠へ落とす。実記録 2 局で全部の守りの窓を確かめる。"""
    from opcg_sim.learned.train import plan_labels as PL
    n_guard = n_flag_old = 0
    for rows, ex, order, idx2cid, cards, own_last in _fixture_frames():
        first_main = {}
        for i in order:
            w0, t0 = int(rows["who"][i]), int(rows["turn"][i])
            if t0 >= 1 and PL.is_own_turn(w0, t0) and int(rows["kind"][i]) == 0 and (w0, t0) not in first_main:
                first_main[(w0, t0)] = i
        old = CP.CutFrames(order, rows, ex, idx2cid, cards, own_last, MU, end_of_turn=True)
        new = CP.CutFrames(order, rows, ex, idx2cid, cards, first_main, MU)
        for n, i in enumerate(order):
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or PL.is_own_turn(w, t):
                continue                                  # 守りの窓＝相手のターンの自分の行
            n_guard += 1
            k_new = new.frame_key(1 - w, t, at_n=n)            # 因果の選び方（この行より後ろを読む枠は飛ばす）
            assert k_new is None or new._key_last_pos(k_new) <= n
            k_old = old.frame_key(1 - w, t, at_n=n)            # 旧の枠の束でも、因果の選び方なら先読みしない
            assert k_old is None or old._key_last_pos(k_old) <= n
            p_old = old.lookahead_pos(1 - w, t)
            if p_old is not None and p_old > n:
                n_flag_old += 1
    assert n_guard >= 20
    assert n_flag_old >= 1                                # 旧の引き方は先読みしていた（検出器が働く）


_TB_RUNNER = r"""
import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "tests")); sys.path.insert(0, os.path.join(os.getcwd(), "tests", "scripts"))
import _bootstrap  # noqa
import theory_bridge as TB
rc = TB.main(["--in", sys.argv[1], "--boot-reps", "10", "--out", sys.argv[2]])
sys.exit(rc)
"""


def test_linear_bridge_never_reads_a_frame_after_the_row(tmp_path):
    """線形の橋（出荷の既定 `joint`）の全部の窓の引き（攻めの行・守りの窓）が、呼んだ行より後ろの行を読まない
    （`CutFrames.view(at_n=)` の `cut_lookahead` が 0）。別プロセス・実記録 2 局。"""
    import json
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = str(tmp_path / "tb.json")
    r = subprocess.run([sys.executable, "-c", _TB_RUNNER, _REC, out], cwd=root,
                       env=dict(os.environ, OPCG_LOG_SILENT="1"), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    st = json.loads(open(out, encoding="utf-8").read())["stats"]
    assert st.get("cut_lookups", 0) >= 50
    assert st.get("cut_lookahead", 0) == 0
