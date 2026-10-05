"""**Rust 化・第 1 段（2026-10-05）**: 守る側の計算（動的計画・段ごとの状態の数え方・地平の選び方）を Rust の核へ
移しても**値が 1 ビットも変わらない**ことの固定。

比べる相手は (1) 今の Python の解き方（`OPCG_RD_KERNEL=py`）、(2) 速くする前の原文（`tests/harness/rule_don_ref.py`）、
(3) 記録した解の**ビットの値**（`tests/fixtures/rd_kernel_golden.jsonl.gz`・実 w41 と合成 w39/w42 の記録の解）。

**古い wheel は skip ではなく fail**（`make test` は wheel を作り直さない・CLAUDE.md の golden 門と同じ作法）。
"""
import gzip
import json
import os
import random
import sys

import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "scripts"), os.path.join(_HERE, "harness")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import crossing_bridge as CB  # noqa: E402
import cut_price as CP  # noqa: E402
import plan_store as PS  # noqa: E402
import rd_kernel as RK  # noqa: E402
import rule_don_ref as REF  # noqa: E402
import test_rd_speed as TS  # noqa: E402  （問題の作り方・枠の読み込みを共有する）

GOLDEN = os.path.join(_HERE, "fixtures", "rd_kernel_golden.jsonl.gz")


@pytest.fixture(scope="module", autouse=True)
def _fresh_wheel():
    ok, why, _E = RK.status()
    if not ok:
        pytest.fail("Rust の核が使えない（skip ではなく fail）: " + why)
    assert sys.version_info < (3, 12), "Python 3.12 以降の sum() は補償和＝Python の基準の値が変わる（設計書 4.1）"


@pytest.fixture(autouse=True)
def _restore():
    old_budget, old_mode = CB.EX_STATE_BUDGET, RK.MODE
    yield
    CB.EX_STATE_BUDGET = old_budget
    RK.MODE = old_mode
    TS._clear_new()
    RK.reset_global()
    REF.clear()


def _E():
    return RK.engine()


# ---------------------------------------------------------------------------------------------
# L0: 数値の道具

def test_round_helpers_match_python_on_many_values():
    E = _E()
    rng = random.Random(5)
    vals = [rng.uniform(-3000, 3000) for _ in range(20000)]
    vals += [k / 1024 for k in range(-4096, 4096)]                     # 9 桁で厳密な同点になる値
    vals += [round(rng.random(), 3) + 5e-10 for _ in range(3000)]      # 同点すれすれ
    vals += [-1e-12, 1e-12, -0.0, 0.0, 2.5e-10, 7.5e-10, 1000.0000000004]
    for x in vals:
        for n in (9, 12, 3):
            assert E.rd_py_round(x, n).hex() == round(x, n).hex(), (x, n)
    for x in [i / 2 for i in range(-40, 41)] + [rng.uniform(-9, 9) for _ in range(2000)]:
        assert E.rd_bankers_round(x) == float(round(x)), x


def test_wheel_handshake_and_plan_store_tag():
    ok, why, E = RK.status()
    api, h = E.rd_kernel_version()
    assert ok and api == RK.RD_KERNEL_API
    assert h == RK.source_hash(), "wheel の原文のハッシュがソースと違う（`make rust-develop`）"
    RK.set_mode("py")
    assert RK.kernel_tag() == "py"
    RK.set_mode("rs")
    assert RK.kernel_tag() == "rs:%d:%s" % (api, h)
    RK.set_mode("both")
    assert RK.kernel_tag() == "py"            # 両方で解くが返すのは Python の結果


def test_a_stale_wheel_fails_loudly(monkeypatch):
    """原文のハッシュが違う wheel は `rs` で落ち、`auto` は Python へ戻る。"""
    monkeypatch.setattr(RK, "_STATUS", [])
    monkeypatch.setattr(RK, "source_hash", lambda root=None: "0" * 16)
    ok, why, _ = RK.status()
    assert not ok and "古い" in why
    monkeypatch.setattr(RK, "MODE", "rs")
    with pytest.raises(RuntimeError):
        RK.engine()
    monkeypatch.setattr(RK, "MODE", "auto")
    assert RK._mode() == "py"


# ---------------------------------------------------------------------------------------------
# L2: 動的計画（辞書のビット・状態の数・予算の判定）

def _rand_guard(rng):
    """動的計画だけの乱数の問題（手札・ブロッカー・ライフの札・引く札・攻撃の並びが多彩・分数のドンも）。"""
    kinds = [(1000.0, 0.0), (2000.0, 0.0), (1000.0, 1.0), (500.0, 0.0), (3000.0, 2.0), (2000.0, 1.0)]
    cards = [rng.choice(kinds) for _ in range(rng.randint(0, 5))]
    xs = (-2000.0, -1000.0, 0.0, 0.0, 1000.0, 2000.0, 3000.0, 4000.0)
    first = [rng.choice(xs) for _ in range(rng.randint(0, 4))]
    seq = tuple(tuple(sorted(rng.choice(xs) for _ in range(rng.randint(0, 3)))) for _ in range(rng.randint(0, 3))) or ((),)
    blk = [rng.choice((-1000.0, 0.0, 500.0, 1000.0, 2000.0)) for _ in range(rng.randint(0, 3))]
    rest = [rng.choice((0.0, 1000.0)) for _ in range(rng.randint(0, 2))]
    arr = [rng.choice((0.0, 1000.0, 3000.0)) for _ in range(rng.randint(0, 2))]
    lt = tuple((c, d, p) for (c, d), p in zip(rng.sample(kinds, rng.randint(0, 3)), (0.3, 0.25, 0.2)))
    dt = tuple((c, d, p) for (c, d), p in zip(rng.sample(kinds, rng.randint(0, 3)), (0.2, 0.2, 0.15)))
    don = rng.choice((0.0, 1.0, 2.0, 3.0, 0.3, 1.7, 2.0000000004))
    life = rng.choice((0.0, 1.0, 2.0, 3.0, 2.5, 0.5, 1.5))
    turns = rng.choice((None, 1, 2, 3, 4))
    mu = rng.choice((0.4, 0.55, 0.6))
    return dict(cards=cards, don=don, xs_first=first, seq=seq, blk=blk, life=life, turns=turns, lt=lt, dt=dt,
                lam=1.0, lam_net=rng.choice((0.35, 0.5)), mu=mu, olp=5000.0, mlp=rng.choice((5000.0, 6000.0)),
                rest=rest, arr=arr)


def _py_guard(p, memo, lim):
    orig = RK._ORIG[id(vars(CB))]["_rule_guard_plan_ex"]
    CB._EX_USED["memo"], CB._EX_USED["limit"] = memo, lim
    try:
        return orig(p["cards"], p["don"], p["xs_first"], p["seq"], p["blk"], p["life"], p["turns"], p["lt"], p["lam"],
                    p["lam_net"], p["mu"], p["olp"], p["mlp"], p["rest"], p["arr"], p["dt"])
    finally:
        CB._EX_USED["memo"], CB._EX_USED["limit"] = None, None


def _rs_guard(p, d):
    return RK.guard_rs(p["cards"], p["don"], p["xs_first"], p["seq"], p["blk"], p["life"], p["turns"], p["lt"],
                       p["lam"], p["lam_net"], p["mu"], p["olp"], p["mlp"], p["rest"], p["arr"], p["dt"],
                       defender=d)


def test_dp_random_problems_bit_identical_with_state_counts_and_budget_decisions():
    """乱数の動的計画 300 問: 辞書の `repr` が同じ・作った状態の数が同じ・予算ごとに「超えるか」が同じ。"""
    E = _E()
    rng = random.Random(20261005)
    n_over = n_ok = 0
    for i in range(300):
        p = _rand_guard(rng)
        for lim in (None, 30, 400):
            memo = {}
            try:
                py = _py_guard(p, memo, lim)
            except CB._ModelBudget:
                py = "budget"
            d = E.RdDefender(lim)
            rs = _rs_guard(p, d)
            if py == "budget":
                n_over += 1
                assert rs is None, (i, lim)
                assert d.n_states() == len(memo) == lim + 1
            else:
                n_ok += 1
                assert rs is not None and repr(py) == repr(rs), (i, lim, py, rs)
                assert d.n_states() == len(memo), (i, lim)
    assert n_over >= 50 and n_ok >= 400


def test_dp_memo_is_shared_across_roots_of_one_attempt():
    """同じ試行の覚え書きは複数の根（計画）で共有される＝2 問目以降は状態が少ししか増えない・数は Python と同じ。"""
    E = _E()
    rng = random.Random(3)
    for i in range(60):
        base = _rand_guard(rng)
        base["turns"] = rng.choice((2, 3, 4))
        memo = {}
        d = E.RdDefender(None)
        for xs in ([0.0], [1000.0, 2000.0], [0.0, 0.0, 1000.0], [2000.0]):
            p = dict(base, xs_first=xs)
            CB._EX_USED["memo"], CB._EX_USED["limit"] = memo, None
            try:
                orig = RK._ORIG[id(vars(CB))]["_rule_guard_plan_ex"]
                py = orig(p["cards"], p["don"], p["xs_first"], p["seq"], p["blk"], p["life"], p["turns"], p["lt"],
                          p["lam"], p["lam_net"], p["mu"], p["olp"], p["mlp"], p["rest"], p["arr"], p["dt"])
            finally:
                CB._EX_USED["memo"], CB._EX_USED["limit"] = None, None
            rs = _rs_guard(p, d)
            assert repr(py) == repr(rs)
            assert d.n_states() == len(memo)


def test_dp_ties_and_degenerate_inputs():
    """同点が多い問題（全部同じ値段・同じブロッカー）と退化した入力（空・ライフ 0・攻撃なし）。"""
    E = _E()
    cases = []
    for blk in ([], [1000.0], [1000.0, 1000.0], [1000.0, 1000.0, 1000.0]):
        for first in ([], [1000.0], [1000.0, 1000.0], [1000.0, 1000.0, 1000.0, 1000.0]):
            for life in (0.0, 1.0, 2.0):
                cases.append(dict(cards=[(1000.0, 0.0)] * 2, don=1.0, xs_first=first, seq=((1000.0, 1000.0),), blk=blk,
                                  life=life, turns=3, lt=((1000.0, 0.0, 0.5),), dt=((1000.0, 0.0, 0.5),), lam=1.0,
                                  lam_net=0.5, mu=0.5, olp=5000.0, mlp=5000.0, rest=[], arr=[]))
    cases.append(dict(cases[0], cards=[], lt=(), dt=(), turns=None))
    cases.append(dict(cases[1], turns=0))
    cases.append(dict(cases[2], xs_first=[-0.0, 0.0, -5.0], seq=((-0.0, 0.0),)))
    for p in cases:
        for lim in (None, 12):
            memo = {}
            try:
                py = _py_guard(p, memo, lim)
            except CB._ModelBudget:
                py = "budget"
            d = E.RdDefender(lim)
            rs = _rs_guard(p, d)
            if py == "budget":
                assert rs is None and d.n_states() == len(memo)
            else:
                assert repr(py) == repr(rs) and d.n_states() == len(memo)


# ---------------------------------------------------------------------------------------------
# 全体の解き方（`rule_don_solve`）: py と rs と原文のビットの一致

def _solve(p, mode, turns=None):
    RK.set_mode(mode)
    TS._clear_new()
    RK.reset_global()
    return TS._solve_new(p, turns)


@pytest.mark.parametrize("budget", [None, 40, 300, 25, 3000])
def test_random_problems_rs_equals_py_equals_reference(budget):
    CB.EX_STATE_BUDGET = budget
    cuts = 0
    for p in TS._PROBLEMS:
        py = _solve(p, "py")
        rs = _solve(p, "rs")
        assert repr(rs) == repr(py)
        if budget in (None, 300):
            REF.clear()
            assert repr(rs) == repr(TS._solve_ref(p))
        cuts += int(rs[2]["horizon"] < rs[2]["horizon0"])
    if budget in (25, 40):
        assert cuts >= 5


@pytest.mark.parametrize("budget", [300000, 150, 20])
def test_frames_rs_equals_py(budget):
    CB.EX_STATE_BUDGET = budget
    frames = TS._frames()
    assert len(frames) >= 8
    cuts = 0
    for _src, p in frames:
        py = _solve(p, "py")
        rs = _solve(p, "rs")
        assert repr(rs) == repr(py)
        cuts += int(rs[2]["horizon"] < rs[2]["horizon0"])
    if budget in (150, 20):
        assert cuts >= 3


def test_frames_rs_equals_reference_at_a_small_budget():
    CB.EX_STATE_BUDGET = 150
    for _src, p in TS._frames()[:4]:
        rs = _solve(p, "rs")
        REF.clear()
        assert repr(rs) == repr(TS._solve_ref(p))


def test_explicit_horizons_and_both_mode_agree():
    for p in TS._PROBLEMS[:15]:
        for h in (1, 2, 3):
            py = _solve(p, "py", h)
            assert repr(_solve(p, "rs", h)) == repr(py)
            assert repr(_solve(p, "both", h)) == repr(py)


def test_both_mode_passes_and_counts_checks_on_budgeted_solves():
    CB.EX_STATE_BUDGET = 60
    before = RK.STATS["both_checked"]
    for _src, p in TS._frames():
        py = _solve(p, "py")
        assert repr(_solve(p, "both")) == repr(py)
    for p in TS._PROBLEMS[:20]:
        py = _solve(p, "py")
        assert repr(_solve(p, "both")) == repr(py)
    assert RK.STATS["both_checked"] - before > 100


def test_both_mode_raises_on_a_wrong_kernel(monkeypatch):
    """`both` は食い違いを握りつぶさない（核が 1 ビットずれたと仮定して例外になる）。"""
    real = RK.guard_rs

    def bad(*a, **k):
        r = real(*a, **k)
        if r is not None:
            r = dict(r, theta=r["theta"] + 1e-15)
        return r
    monkeypatch.setattr(RK, "guard_rs", bad)
    with pytest.raises(AssertionError, match="rd_kernel"):
        _solve(TS._PROBLEMS[3], "both")


def test_cut_decisions_are_the_same_cold_warm_and_reordered_with_the_kernel():
    """縮めの判断は問題だけの関数（rs でも冷たい・温まった・逆順で同じ）。"""
    CB.EX_STATE_BUDGET = 60
    probs = TS._PROBLEMS[:20] + [p for _s, p in TS._frames()]
    cold = {}
    for i, p in enumerate(probs):
        cold[i] = repr(_solve(p, "rs"))
    RK.set_mode("rs")
    warm = {}
    for i in reversed(range(len(probs))):
        warm[i] = repr(TS._solve_new(probs[i]))        # 覚え書きを消さずに（逆順）
    assert cold == warm
    py = {i: repr(_solve(p, "py")) for i, p in enumerate(probs)}
    assert py == cold


# ---------------------------------------------------------------------------------------------
# L3: 段ごとの数え方・地平の選び方

def test_layer_counts_and_fit_horizon_match_python_and_the_real_dp():
    E = _E()
    checked = 0
    for p in TS._PROBLEMS[:30] + [p for _s, p in TS._frames()]:
        cards, don, blk, life, ax, _t, lt, dt, arr = p
        ax = dict(ax)
        with CP.defending(TS._view_of(ax)):
            masks = CB._rule_don_masks(cards, blk, life, ax, lt)
            roots = [(xf, m["later_seq"]) for m in masks for xf in CB._mask_firsts(m, ax)]
            rest = tuple(ax.get("rest_blk") or ())
            for lim in (10 ** 12, 5000, 400, 60, 7):
                py = CB._ex_count_layers(cards, don, blk, life, lt, rest, arr, dt, roots, 4, lim)
                rs = RK.count_layers(cards, don, blk, life, lt, rest, arr, dt, roots, 4, lim)
                assert list(py) == list(rs), (lim, py, rs)
                assert E.rd_fit_horizon(rs, lim, 4) == max(1, min(_fit(py, lim), 4))
                checked += 1
            # 段 0..h-1 の和 = 本物の動的計画の試行が作る状態の数（Rust の動的計画でも）
            sizes = RK.count_layers(cards, don, blk, life, lt, rest, arr, dt, roots, 4, 10 ** 12)
            for h in range(1, 5):
                RK.set_mode("rs")
                CB._EX_USED["memo"], CB._EX_USED["limit"] = {}, None
                try:
                    CB._rule_don_solve(cards, don, blk, life, ax, h, lt, dt, arr, masks=masks)
                    real = RK._ATT["d"].n_states()
                finally:
                    CB._EX_USED["memo"], CB._EX_USED["limit"] = None, None
                assert sum(sizes[:h]) == real
                checked += 1
    assert checked >= 200


def _fit(sizes, lim):
    tot, fit = 0, 1
    for t, n in enumerate(sizes):
        tot += n
        if tot > lim:
            break
        fit = t + 1
    return fit


def test_fit_horizon_wrapper_agrees_in_all_modes():
    CB.EX_STATE_BUDGET = 60
    for _src, p in TS._frames():
        cards, don, blk, life, ax, _t, lt, dt, arr = p
        ax = dict(ax)
        with CP.defending(TS._view_of(ax)):
            masks = CB._rule_don_masks(cards, blk, life, ax, lt)
            outs = []
            for mode in ("py", "rs", "both"):
                RK.set_mode(mode)
                outs.append(CB._ex_fit_horizon(cards, don, blk, life, ax, lt, dt, arr, masks, 4, 60))
            assert len(set(outs)) == 1


# ---------------------------------------------------------------------------------------------
# 記録した解（ビットの値）

def _golden():
    if not os.path.exists(GOLDEN):
        pytest.fail("記録した解が無い: " + GOLDEN)
    with gzip.open(GOLDEN, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_golden_solves_replay_bit_identically_with_the_kernel():
    recs = _golden()
    srcs = {r["src"].split("#")[0] for r in recs}
    n_real = sum(r["src"].startswith("real") for r in recs)
    n_syn = sum(r["src"].startswith("syn") for r in recs)
    assert n_real >= 60 and n_syn >= 60, (n_real, n_syn, srcs)
    cuts = 0
    for r in recs:
        cards, don, blk, life, ax, turns, lt, dt, arr = RK.dec(r["args"])
        ax = dict(ax)
        for k in ("key", "cp", "rest_blk"):
            ax[k] = TS._tup(ax[k])
        CB.EX_STATE_BUDGET = r["budget"]
        TS._clear_new()
        RK.reset_global()
        RK.set_mode("rs")
        with CP.defending(TS._view_of(ax)):
            out = CB.rule_don_solve(cards, don, blk, life, dict(ax), turns, lt, dt, arr)
        assert RK.enc(out) == r["expect"], r["src"]
        cuts += int(out[2]["horizon"] < out[2]["horizon0"])
    assert cuts >= 0


# ---------------------------------------------------------------------------------------------
# 計画の覚え書きの鍵

def test_the_plan_store_key_separates_the_kernel_and_keeps_python_keys_stable(tmp_path):
    p = TS._PROBLEMS[5]
    cards, don, blk, life, ax, _t, lt, dt, arr = p
    st = PS.PlanStore(str(tmp_path / "ps"), CB)
    RK.set_mode("py")
    k_py = st.key_of(cards, don, blk, life, ax, None, lt, dt, arr)
    RK.set_mode("both")
    assert st.key_of(cards, don, blk, life, ax, None, lt, dt, arr) == k_py
    RK.set_mode("rs")
    k_rs = st.key_of(cards, don, blk, life, ax, None, lt, dt, arr)
    assert k_rs != k_py
    assert "RD_KERNEL_TAG" in vars(CB)
    # 包んだ関数でも、解き方が読む大域の一覧（鍵に入る切替）が欠けない
    names = {n for _lab, n, _src in st.reads}
    assert {"RATE_T1_MODE", "RATE_DON_PAY", "EX_STATE_BUDGET", "CUT_PRICER_KEY", "NU_MEAS"} <= names
