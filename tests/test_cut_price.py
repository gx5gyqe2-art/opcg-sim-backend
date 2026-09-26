"""**N-3（2026-09-26・ユーザ決定 判断6(a)／判断7(a) の第 2 段）**: 切らせた札の値段＝1 枚 1 役の手札の価値の減り
（`tests/scripts/cut_price.py`）を、損害の側（攻撃の守る値段・実現の損害の手札の部分）と耐久の側（`Θ` の手札の項）へ
**同時に**通す切替 `CUT_PRICE_MODE=joint`（既定は `flat`＝旧の 1 枚一律 `μ`）。

押さえること（期待値は手で計算した値か、独立の総当たり）:

* 手で計算した `L(k)`（切る k 枚の最安の値段）・端数の直線・手札を越えた分は `μ`。
* **一律の値段への還元**: 全部の札が `v = μ` で守りの役が効かないなら `L(k) = kμ`・`Θ` の手札の項・攻撃の値段が旧の値に戻る。
* **望遠鏡の不変量**: 切った組の値段を順に足すと `V(枠) − V(残り)`・最安の組を順に切れば**耐久の予約と損害が厳密に一致**。
* 単調（`L` は減らない・値段は 0 以上・窓の値段は足し算で閉じる）。
* 差し替え口は文脈の中だけ（外では `c(x)·μ` そのもの・相手の体の `ν` は旧の値段のまま）。
* `hand_cut_count × μ` は旧の `hand_absorb*` と一致（数を変えていない）・`cuttable_indices` は `cuttable_share` と一致。

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

def test_shipped_default_is_flat():
    assert _SHIPPED_MODE == "flat"
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
    """`hand_cut_count × μ` は旧の手札の項（`hand_absorb*`）と一致＝N-3 は数を変えず値段だけを変える。"""
    rng = np.random.default_rng(3)
    old = CB.THETA_HAND_MODE
    try:
        for mode in ("cuttable", "cuttable_cx", "cuttable_forced", "cuttable_seq"):
            CB.set_theta_hand_mode(mode)
            for _ in range(200):
                hand_n = float(rng.integers(0, 9))
                g = MU * float(rng.choice([0.0, 0.25, 0.5, 0.6, 1.0]))
                xs = [float(x) for x in rng.choice([-1000.0, 0.0, 1000.0, 2000.0, 4000.0], size=int(rng.integers(0, 5)))]
                life = float(rng.integers(0, 5)); blk = int(rng.integers(0, 3))
                n_cut = (g / MU) * hand_n
                want = {"cuttable": g * hand_n,
                        "cuttable_cx": CB.hand_absorb(n_cut, max(xs) if xs else -1.0, MU),
                        "cuttable_forced": CB.hand_absorb_forced(n_cut, xs, life, blk, MU),
                        "cuttable_seq": CB.hand_absorb_seq(n_cut, xs, MU)}[mode]
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
