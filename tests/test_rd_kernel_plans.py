"""**Rust 化・第 2a 段（2026-10-05）**: 計画の列挙と採点（`_rule_don_solve` の `visit`／`sig`／点数）・歩き（`rules_sched`・
`walk_crossing` の `sched` の経路）・地平の試行のループ（`rule_don_solve`）を Rust の核（`opcg_engine.rd_solve`）へ移しても
**値が 1 ビットも変わらない**ことの固定（設計書 L4）。

比べる相手は 2 つ: (1) 速くする前の原文（`tests/harness/rule_don_ref.py`）、
(2) 記録した解のビット（`rust/opcg_engine/tests/fixtures/rd_solve_golden.jsonl.gz` は `cargo test` が Python 無しで再生する・
`tests/fixtures/rd_kernel_golden.jsonl.gz` は `test_rd_kernel.py` が `rs` で再生する＝どちらもこの段の経路を通る）。
試行の開示（`EX_SPEED_STATS` の増分）は記録した値（速くした Python の解き方が出したもの）と、試行ごとの状態の数は
原文の同じ地平の試行と同じであることを見る。**第 3 段（2026-10-05）**で速くした Python の解き方（旧 `OPCG_RD_KERNEL=py`）を
消したので、それとの比べは原文・記録との比べに置き換えた。

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
    before = [CB.EX_SPEED_STATS[k] for k in _KEYS]
    out = TS._solve_new(p, turns)
    return out, tuple(CB.EX_SPEED_STATS[k] - b for k, b in zip(_KEYS, before))


def _more_problems():
    return [TS._rand_problem(random.Random(5000 + i), 1000 + i) for i in range(160)]


# ---------------------------------------------------------------------------------------------
# 歩き（E18・E19）

def _py_sched(harms, steps, actx, paid1, theta):
    """原文の歩き（`rule_don_ref.rules_sched`→`walk_crossing`）。"""
    st = [dict(s) for s in steps]
    sched = RK.ref_module().rules_sched(harms, st, actx, paid1)
    return sched, CB.walk_crossing(sched, theta, actx)


def test_sched_and_walk_are_bit_identical_on_random_inputs():
    """原文の `rules_sched`→`walk_crossing` を乱数の入力（表の端・払い過ぎ・損害が段より長い／短い・届かない・`-0.0`）で。"""
    E = RK.engine()
    rng = random.Random(77)
    cap = RK.race_cap(CB.tau_grow)
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
# 全体（`rule_don_solve`）: rs ＝ 原文・開示は記録と同じ

def _rec_stats():
    """記録した解（`rd_solve_golden`）の試行の開示（速くした Python の解き方が出した値・予算ごと）。"""
    import gzip
    import json
    path = os.path.join(_HERE, "..", "rust", "opcg_engine", "tests", "fixtures", "rd_solve_golden.jsonl.gz")
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            for r in d["runs"]:
                if r["turns"] is None:
                    out[(d["src"], r["limit"])] = tuple(r["expect"]["stats"])
    return out


@pytest.mark.parametrize("budget", [300000, 150])
def test_frames_rs_equals_reference_with_the_recorded_disclosure(budget):
    """局面 8 つ: rs ＝ 原文（`repr`）・試行の開示（`EX_SPEED_STATS` の増分）が記録した値（速くした Python のもの）と同じ。"""
    CB.EX_STATE_BUDGET = budget
    rec = _rec_stats()
    cuts = 0
    for _src, p in TS._frames():
        rs, st_rs = _solve(p, "rs")
        assert st_rs == rec[("frame:" + _src, budget)], (_src, st_rs)
        REF.clear()
        assert repr(TS._solve_ref(p)) == repr(rs), _src
        cuts += int(rs[2]["horizon"] < rs[2]["horizon0"])
    if budget == 150:
        assert cuts >= 3


def test_recorded_solves_have_the_recorded_disclosure():
    """記録した 152 解 × 予算 2: 試行の開示が記録（速くした Python の解き方の値）と同じ（解のビットは `test_rd_kernel` が見る）。"""
    import rd_kernel_golden as G
    import gzip
    import json
    rec = _rec_stats()
    with gzip.open(G.GOLDEN, "rt", encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    n = nz = 0
    for r in recs:
        cards, don, blk, life, ax, turns, lt, dt, arr = G.args_of(r)
        for budget in [r["budget"]] + [e["budget"] for e in r["extra"]]:
            CB.EX_STATE_BUDGET = budget
            _out, st = _solve((cards, don, blk, life, ax, None, lt, dt, arr), "rs", turns)
            assert st == rec[(r["src"], budget)], (r["src"], budget, st)
            n += 1
            nz += int(any(st))
    assert n >= 300 and nz >= 20


@pytest.mark.parametrize("budget", [None, 25, 40, 300])
def test_random_problems_rs_equals_reference(budget):
    """乱数の 200 問（既存 40＋新 160・`no_attack_now` を含む）× 予算 4: rs ＝ 原文。"""
    CB.EX_STATE_BUDGET = budget
    probs = TS._PROBLEMS + _more_problems()
    cuts = nonow = 0
    for i, p in enumerate(probs):
        rs, _st = _solve(p, "rs")
        REF.clear()
        assert repr(TS._solve_ref(p)) == repr(rs), i
        cuts += int(rs[2]["horizon"] < rs[2]["horizon0"])
        nonow += int(bool(p[4].get("no_attack_now")))
    assert nonow >= 10
    if budget in (25, 40):
        assert cuts >= 10


def test_explicit_horizons_rs_equals_reference():
    probs = TS._PROBLEMS[:20] + _more_problems()[:30] + [p for _s, p in TS._frames()]
    for i, p in enumerate(probs):
        for h in (1, 2, 3):
            rs, _ = _solve(p, "rs", h)
            assert "horizon" not in rs[2]
            REF.clear()
            assert repr(TS._solve_ref(p, h)) == repr(rs), (i, h)


def test_states_per_attempt_equal_the_reference(monkeypatch):
    """試行ごとに作った状態の数（予算超えの試行は超えた時点＝予算 + 1）が、同じ地平の原文の試行の `len(memo)` と同じ
    （原文は 1 ターンずつ縮めるので、Rust の試行〔最初の地平と、数えて選んだ地平〕は原文の試行の部分列）。"""
    R = RK.ref_module()
    seen = {}
    real = R._rule_don_solve

    def spy(cards_d, don_d, blk, life, actx, turns, *a, **k):
        try:
            return real(cards_d, don_d, blk, life, actx, turns, *a, **k)
        finally:
            memo = R._EX_USED["memo"]
            seen[turns] = None if memo is None else len(memo)
    monkeypatch.setattr(R, "_rule_don_solve", spy)
    checked = 0
    for budget in (150, 40):
        CB.EX_STATE_BUDGET = budget
        for p in [p for _s, p in TS._frames()] + TS._PROBLEMS[:20]:
            cards, don, blk, life, ax, _t, lt, dt, arr = p
            seen.clear()
            REF.clear()
            ref = TS._solve_ref(p)
            TS._clear_new()
            RK.set_mode("rs")
            with CP.defending(TS._view_of(ax)):
                masks = CB._rule_don_masks(cards, blk, life, dict(ax), lt)
                h0 = CB.model_horizon(dict(ax), blk, int(max(0, round(float(life)))), arr)
                (_c, _s, plan), h, st = CB._rd_run(cards, don, blk, life, dict(ax), None, lt, dt, arr, masks, h0)
            assert h == ref[2]["horizon"]
            states = list(st[4])
            assert states[0] == seen[h0] and states[-1] == seen[h], (budget, seen, st)
            checked += len(states)
    assert checked >= 60


# ---------------------------------------------------------------------------------------------
# 覚え書き・切替

def test_unbudgeted_cache_aliasing_and_the_plan_store(tmp_path):
    """予算なし（`EX_STATE_BUDGET=None`）では地平つきの鍵でも覚え、地平を渡した解と渡さない解が同じ計画の辞書を指す
    （後から地平の欄が足される・速くした Python と同じ振る舞い）。計画のディスクの覚え書きから読んでも同じ。"""
    CB.EX_STATE_BUDGET = None
    for p in TS._PROBLEMS[:12]:
        cards, don, blk, life, ax, _t, lt, dt, arr = p
        RK.set_mode("rs")
        TS._clear_new()
        with CP.defending(TS._view_of(ax)):
            h0 = CB.model_horizon(dict(ax), blk, int(max(0, round(float(life)))), arr)
            a = CB.rule_don_solve(cards, don, blk, life, dict(ax), h0, lt, dt, arr)
            b = CB.rule_don_solve(cards, don, blk, life, dict(ax), None, lt, dt, arr)
        assert a is b and b[2]["horizon"] == b[2]["horizon0"] == h0
        REF.clear()
        assert repr(b) == repr(TS._solve_ref(p))
    CB.EX_STATE_BUDGET = 300
    CB.PLAN_STORE = PS.PlanStore(str(tmp_path / "ps_rs"), CB)
    RK.set_mode("rs")
    for p in TS._PROBLEMS[:8]:
        TS._clear_new()
        cold = repr(TS._solve_new(p))
        TS._clear_new()
        warm = repr(TS._solve_new(p))               # ディスクから
        assert cold == warm
    assert CB.PLAN_STORE.hits == 8
    CB.PLAN_STORE = None


def test_both_mode_checks_every_plan_and_raises_on_a_wrong_one(monkeypatch):
    CB.EX_STATE_BUDGET = 60
    before = RK.STATS["solve_checked"]
    for p in TS._PROBLEMS[:20] + [p for _s, p in TS._frames()]:
        rs, _ = _solve(p, "rs")
        both, _ = _solve(p, "both")
        assert repr(both) == repr(rs)
    assert RK.STATS["solve_checked"] - before >= 28
    real = CB._rd_run

    def bad(*a, **k):
        out, h, st = real(*a, **k)
        out[2]["tau"] = out[2]["tau"] + 1e-15
        return out, h, st
    monkeypatch.setitem(vars(CB), "_rd_run", bad)
    with pytest.raises(AssertionError, match="rd_kernel"):
        _solve(TS._PROBLEMS[3], "both")
