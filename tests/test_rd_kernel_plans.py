"""**Rust 化・第 2a 段（2026-10-05）**: 計画の列挙と採点（`_rule_don_solve` の `visit`／`sig`／点数）・歩き（`rules_sched`・
`walk_crossing` の `sched` の経路）・地平の試行のループ（`rule_don_solve`）を Rust の核（`opcg_engine.rd_solve`）へ移しても
**値が 1 ビットも変わらない**ことの固定（設計書 L4）。

比べる相手は 3 つ: (1) 今の Python の解き方（`OPCG_RD_KERNEL=py`）、(2) 速くする前の原文（`tests/harness/rule_don_ref.py`）、
(3) 記録した解のビット（`rust/opcg_engine/tests/fixtures/rd_solve_golden.jsonl.gz` は `cargo test` が Python 無しで再生する・
`tests/fixtures/rd_kernel_golden.jsonl.gz` は `test_rd_kernel.py` が `rs` で再生する＝どちらもこの段の経路を通る）。
試行の開示（`EX_SPEED_STATS` の増分）と試行ごとの状態の数も Python と同じであることを見る。

**古い wheel は skip ではなく fail**（`make test` は wheel を作り直さない）。
"""
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
import test_rd_speed as TS  # noqa: E402

_KEYS = ("attempt_fail", "attempt_skipped", "count_calls", "count_fallback")


@pytest.fixture(scope="module", autouse=True)
def _fresh_wheel():
    ok, why, E = RK.status()
    if not ok:
        pytest.fail("Rust の核が使えない（skip ではなく fail）: " + why)
    if not hasattr(E, "rd_solve"):
        pytest.fail("wheel が第 2a 段より古い（rd_solve が無い）: `make rust-develop`")
    assert sys.version_info < (3, 12), "Python 3.12 以降の sum() は補償和＝Python の基準の値が変わる（設計書 4.1）"


@pytest.fixture(autouse=True)
def _restore():
    old_budget, old_mode, old_store = CB.EX_STATE_BUDGET, RK.MODE, CB.PLAN_STORE
    yield
    CB.EX_STATE_BUDGET = old_budget
    RK.MODE = old_mode
    CB.PLAN_STORE = old_store
    TS._clear_new()
    RK.reset_global()
    REF.clear()


def _solve(p, mode, turns=None):
    """冷たい解き（覚え書きを消して）と、その間の `EX_SPEED_STATS` の増分。"""
    RK.set_mode(mode)
    TS._clear_new()
    RK.reset_global()
    before = [CB.EX_SPEED_STATS[k] for k in _KEYS]
    out = TS._solve_new(p, turns)
    return out, tuple(CB.EX_SPEED_STATS[k] - b for k, b in zip(_KEYS, before))


def _more_problems():
    return [TS._rand_problem(random.Random(5000 + i), 1000 + i) for i in range(160)]


# ---------------------------------------------------------------------------------------------
# 歩き（E18・E19）

def _py_sched(harms, steps, actx, paid1, theta):
    st = [dict(s) for s in steps]
    sched = CB.rules_sched(harms, st, actx, paid1)
    return sched, CB.walk_crossing(sched, theta, actx)


def test_sched_and_walk_are_bit_identical_on_random_inputs():
    """`rules_sched`→`walk_crossing` を乱数の入力（表の端・払い過ぎ・損害が段より長い／短い・届かない・`-0.0`）で。"""
    E = RK.engine()
    rng = random.Random(77)
    cap = RK._race_cap(vars(CB))
    n_cap = n_floor = 0
    for i in range(3000):
        n = rng.randint(1, 30)
        nl = rng.randint(0, 6)
        tab = lambda: [rng.choice((0.0, rng.random() * 0.3, round(rng.random(), 3))) for _ in range(nl + 1)]  # noqa: E731
        a_tab = tab()
        ar_tab = [min(x, rng.random() * 0.2) for x in a_tab] if rng.random() < 0.7 else tab()
        e_tab = tab()
        ds = [float(rng.randint(0, nl + 2)) + rng.choice((0.0, 0.5, 0.3)) for _ in range(rng.randint(1, 35))]
        steps = [{"hits": (), "paid": float(rng.randint(0, 5)) + rng.choice((0.0, 0.5)),
                  "fb": rng.choice((0.0, -0.0, rng.random(), rng.random() * 1e-4)),
                  "eff": rng.choice((0, 0.0, rng.random() * 0.1))} for _ in range(n)]
        harms = [rng.choice((0.0, rng.random() * 0.5, 1e-5)) for _ in range(rng.randint(0, n + 3))]
        no_now = rng.random() < 0.2
        actx = {"ds": ds, "a_tab": a_tab, "ar_tab": ar_tab, "e_tab": e_tab, "no_attack_now": no_now}
        paid1 = float(rng.randint(0, 6)) + rng.choice((0.0, 0.5))
        theta = rng.choice((0.0, -0.0, rng.random() * 3, rng.random() * 50, 1e-9))
        py_s, py_t = _py_sched(harms, steps, actx, paid1, theta)
        rs_s, rs_t = E.rd_sched([float(h) for h in harms], [(s["paid"], s["eff"], s["fb"]) for s in steps], ds,
                                a_tab, ar_tab, e_tab, no_now, paid1, theta, float(CB.SLOPE_FLOOR), cap)
        assert [x.hex() for x in py_s] == [x.hex() for x in rs_s], i
        assert py_t.hex() == rs_t.hex(), (i, py_t, rs_t)
        n_cap += int(rs_t == cap)
        n_floor += int(any(0.0 < x <= CB.SLOPE_FLOOR for x in rs_s))
    assert n_cap >= 50


# ---------------------------------------------------------------------------------------------
# 全体（`rule_don_solve`）: rs ＝ py ＝ 原文・開示も同じ

@pytest.mark.parametrize("budget", [300000, 150])
def test_frames_rs_equals_py_equals_reference_with_the_same_disclosure(budget):
    CB.EX_STATE_BUDGET = budget
    cuts = 0
    for _src, p in TS._frames():
        py, st_py = _solve(p, "py")
        rs, st_rs = _solve(p, "rs")
        assert repr(rs) == repr(py), _src
        assert st_rs == st_py, (_src, st_py, st_rs)
        REF.clear()
        RK.set_mode("py")
        assert repr(TS._solve_ref(p)) == repr(rs), _src
        cuts += int(rs[2]["horizon"] < rs[2]["horizon0"])
    if budget == 150:
        assert cuts >= 3


@pytest.mark.parametrize("budget", [None, 25, 40, 300])
def test_random_problems_rs_equals_py_with_the_same_disclosure(budget):
    CB.EX_STATE_BUDGET = budget
    probs = TS._PROBLEMS + _more_problems()
    cuts = nonow = 0
    for i, p in enumerate(probs):
        py, st_py = _solve(p, "py")
        rs, st_rs = _solve(p, "rs")
        assert repr(rs) == repr(py), i
        assert st_rs == st_py, (i, st_py, st_rs)
        if budget in (None, 300) and i < 40:
            REF.clear()
            assert repr(TS._solve_ref(p)) == repr(rs), i
        cuts += int(rs[2]["horizon"] < rs[2]["horizon0"])
        nonow += int(bool(p[4].get("no_attack_now")))
    assert nonow >= 10
    if budget in (25, 40):
        assert cuts >= 10


def test_explicit_horizons_rs_equals_py_equals_reference():
    probs = TS._PROBLEMS[:20] + _more_problems()[:30] + [p for _s, p in TS._frames()]
    for i, p in enumerate(probs):
        for h in (1, 2, 3):
            py, _ = _solve(p, "py", h)
            rs, _ = _solve(p, "rs", h)
            assert repr(rs) == repr(py), (i, h)
            assert "horizon" not in rs[2]
            if i < 10:
                REF.clear()
                RK.set_mode("py")
                assert repr(TS._solve_ref(p, h)) == repr(rs), (i, h)


def test_states_per_attempt_equal_python(monkeypatch):
    """試行ごとに作った状態の数（予算超えの試行は超えた時点＝予算 + 1）が Python の `len(memo)` と同じ。"""
    seen = []
    real = CB._rule_don_solve

    def spy(*a, **k):
        try:
            return real(*a, **k)
        finally:
            memo = CB._EX_USED["memo"]
            seen.append(None if memo is None else len(memo))
    monkeypatch.setitem(vars(CB), "_rule_don_solve", spy)
    checked = 0
    for budget in (150, 40):
        CB.EX_STATE_BUDGET = budget
        for p in [p for _s, p in TS._frames()] + TS._PROBLEMS[:20]:
            cards, don, blk, life, ax, _t, lt, dt, arr = p
            seen.clear()
            _solve(p, "py")
            TS._clear_new()
            RK.reset_global()
            RK.set_mode("rs")
            with CP.defending(TS._view_of(ax)):
                _out, st = RK.solve_rs(cards, don, blk, life, dict(ax), None, lt, dt, arr, g=vars(CB), cached=False)
            assert list(st[4]) == seen, (budget, seen, st)
            checked += len(seen)
    assert checked >= 60


# ---------------------------------------------------------------------------------------------
# 覚え書き・切替

def test_unbudgeted_cache_aliasing_and_the_plan_store_behave_like_python(tmp_path):
    """予算なし（`EX_STATE_BUDGET=None`）では Python も `_rule_don_solve` の覚え書きを共有し、地平を渡した解と渡さない解が
    同じ計画の辞書を指す（後から地平の欄が足される）——同じ順に呼べば rs も同じ `repr` の列になる。計画の覚え書きも同じ。"""
    CB.EX_STATE_BUDGET = None
    for p in TS._PROBLEMS[:12]:
        cards, don, blk, life, ax, _t, lt, dt, arr = p
        seqs = []
        for mode in ("py", "rs"):
            RK.set_mode(mode)
            TS._clear_new()
            RK.reset_global()
            with CP.defending(TS._view_of(ax)):
                h0 = CB.model_horizon(dict(ax), blk, int(max(0, round(float(life)))), arr)
                a = CB.rule_don_solve(cards, don, blk, life, dict(ax), h0, lt, dt, arr)
                b = CB.rule_don_solve(cards, don, blk, life, dict(ax), None, lt, dt, arr)
                seqs.append((repr(a), repr(b), a is b or a[2] is b[2]))
        assert seqs[0] == seqs[1]
    CB.EX_STATE_BUDGET = 300
    for mode in ("py", "rs"):
        CB.PLAN_STORE = PS.PlanStore(str(tmp_path / ("ps_" + mode)), CB)
        RK.set_mode(mode)
        for p in TS._PROBLEMS[:8]:
            TS._clear_new()
            cold, _ = repr(TS._solve_new(p)), None
            TS._clear_new()
            warm = repr(TS._solve_new(p))               # ディスクから
            assert cold == warm
    CB.PLAN_STORE = None


def test_rs_mode_does_not_run_the_python_plan_loop(monkeypatch):
    """`rs` は計画の列挙・歩き・試行のループを Python で回さない（`_rule_don_solve`・`rules_sched` を呼ばない）。"""
    def boom(*a, **k):
        raise AssertionError("Python の経路が呼ばれた")
    monkeypatch.setitem(vars(CB), "_rule_don_solve", boom)
    monkeypatch.setitem(vars(CB), "rules_sched", boom)
    monkeypatch.setitem(vars(CB), "walk_crossing", boom)
    CB.EX_STATE_BUDGET = 40
    for p in TS._PROBLEMS[:10] + [p for _s, p in TS._frames()][:3]:
        _solve(p, "rs")
        _solve(p, "rs", 2)


def test_both_mode_checks_every_plan_and_raises_on_a_wrong_one(monkeypatch):
    CB.EX_STATE_BUDGET = 60
    before = RK.STATS["solve_checked"]
    for p in TS._PROBLEMS[:20] + [p for _s, p in TS._frames()]:
        py, _ = _solve(p, "py")
        both, _ = _solve(p, "both")
        assert repr(both) == repr(py)
    assert RK.STATS["solve_checked"] - before >= 28
    real = RK.run_rs

    def bad(*a, **k):
        out, h, st = real(*a, **k)
        out[2]["tau"] = out[2]["tau"] + 1e-15
        return out, h, st
    monkeypatch.setattr(RK, "run_rs", bad)
    with pytest.raises(AssertionError, match="rd_kernel"):
        _solve(TS._PROBLEMS[3], "both")
