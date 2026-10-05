"""**Rust 化・第 1 段（2026-10-05）**: 守る側の計算（動的計画・段ごとの状態の数え方・地平の選び方）を Rust の核へ
移しても**値が 1 ビットも変わらない**ことの固定。

比べる相手は (1) 速くする前の原文（`tests/harness/rule_don_ref.py`）、(2) 記録した解の**ビットの値**
（`tests/fixtures/rd_kernel_golden.jsonl.gz`・実 w41 と合成 w39/w42 の記録の解）。**第 3 段（2026-10-05）**で速くした
Python の解き方（旧 `OPCG_RD_KERNEL=py`）を消したので、それとの比べは原文との比べに置き換えた（状態の数も原文の覚え書きの
大きさと比べる）。

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

import theory_order as T  # noqa: E402

#: 器の既定（import 時＝`conftest` の固定の前）。記録した解は器の既定で解いたもの
_TOOL_DEFAULTS = (T.OPTION_MODE, T.W_MODE, T.SURV_MODE, T.CBAR_MODE, T.CLOCK_HAND_MODE)

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
    for m in ("rs", "both", "ref"):
        RK.set_mode(m)
        assert RK.kernel_tag() == "rs:%d:%s" % (api, h)


def test_the_default_is_rs_and_the_removed_modes_fail_loudly():
    """第 3 段: 既定は `rs`（環境変数なしの別プロセスで確かめる）。消した `py`／`auto` は理由つきで落ちる。"""
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "OPCG_RD_KERNEL"}
    code = "import sys; sys.path[:0] = %r; import rd_kernel as RK; print(RK.MODE)" % ([os.path.join(_HERE, "scripts")],)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip() == "rs", r.stderr
    for m in ("py", "auto"):
        with pytest.raises(ValueError, match="使えない"):
            RK.set_mode(m)
        r = subprocess.run([sys.executable, "-c", code], env=dict(env, OPCG_RD_KERNEL=m), capture_output=True,
                           text=True, timeout=60)
        assert r.returncode != 0 and "使えない" in r.stderr
    with pytest.raises(ValueError):
        RK.set_mode("nope")


def test_the_optimized_python_solver_is_gone():
    """第 3 段で消した速くした Python の解き方（設計 §9 の (B)）が戻っていない（写しを 3 つにしない）。"""
    for name in ("_ex_prep", "_norm_seq", "_seq_prep", "_ex_counter_sets", "_rule_guard_plan_ex", "_mask_firsts",
                 "_ex_fit_horizon", "_ex_count_layers", "_rule_don_solve", "rules_sched", "_tab", "EX_LAYER_COUNT",
                 "_SEQ_NORM", "_SEQ_PREP", "_RULE_EX_CACHE", "_RULE_EX_SETS", "_RULE_EX_MEMO", "_RULE_EX_CTX",
                 "_EX_USED", "_ModelBudget"):
        assert not hasattr(CB, name), name
    # 第 2b 段が移すもの（出す札の組）と、原文が借りるものは残る
    for name in ("rules_steps", "purse_plan_witness", "_attach_gain", "_rule_don_masks", "model_horizon",
                 "walk_crossing", "tau_grow", "rate_at", "_prices_of"):
        assert callable(getattr(CB, name)), name


def test_a_stale_wheel_fails_loudly(monkeypatch):
    """原文のハッシュが違う wheel は `rs`／`both` で落ちる（黙って他の解き方に戻らない）。"""
    monkeypatch.setattr(RK, "_STATUS", [])
    monkeypatch.setattr(RK, "source_hash", lambda root=None: "0" * 16)
    ok, why, _ = RK.status()
    assert not ok and "古い" in why
    with pytest.raises(RuntimeError):
        RK.engine()
    for m in ("rs", "both"):
        with pytest.raises(RuntimeError):
            RK.set_mode(m)


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


def _ref_guard(p, memo, lim):
    """原文（`rule_don_ref._rule_guard_plan_ex`）の動的計画を、渡した覚え書きと予算で解く。"""
    R = RK.ref_module()
    R._EX_USED["memo"], R._EX_USED["limit"] = memo, lim
    try:
        return R._rule_guard_plan_ex(p["cards"], p["don"], p["xs_first"], p["seq"], p["blk"], p["life"], p["turns"],
                                     p["lt"], p["lam"], p["lam_net"], p["mu"], p["olp"], p["mlp"], p["rest"],
                                     p["arr"], p["dt"])
    finally:
        R._EX_USED["memo"], R._EX_USED["limit"] = None, None


def _rs_guard(p, d):
    return RK.guard_rs(p["cards"], p["don"], p["xs_first"], p["seq"], p["blk"], p["life"], p["turns"], p["lt"],
                       p["lam"], p["lam_net"], p["mu"], p["olp"], p["mlp"], p["rest"], p["arr"], p["dt"],
                       defender=d)


def test_dp_random_problems_bit_identical_with_state_counts_and_budget_decisions():
    """乱数の動的計画 300 問: 原文と辞書の `repr` が同じ・作った状態の数が同じ・予算ごとに「超えるか」が同じ。"""
    E = _E()
    rng = random.Random(20261005)
    n_over = n_ok = 0
    for i in range(300):
        p = _rand_guard(rng)
        for lim in (None, 30, 400):
            memo = {}
            try:
                py = _ref_guard(p, memo, lim)
            except REF._ModelBudget:
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
        for xs, bk in (([0.0], base["blk"]), ([1000.0, 2000.0], [1000.0]), ([0.0, 0.0, 1000.0], []),
                       ([2000.0], base["blk"]), ([1000.0, 2000.0], [1000.0, 0.0])):
            p = dict(base, xs_first=xs, blk=bk)
            py = _ref_guard(p, memo, None)
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
                py = _ref_guard(p, memo, lim)
            except REF._ModelBudget:
                py = "budget"
            d = E.RdDefender(lim)
            rs = _rs_guard(p, d)
            if py == "budget":
                assert rs is None and d.n_states() == len(memo)
            else:
                assert repr(py) == repr(rs) and d.n_states() == len(memo)


# ---------------------------------------------------------------------------------------------
# 全体の解き方（`rule_don_solve`）: rs と原文のビットの一致

def _solve(p, mode, turns=None):
    RK.set_mode(mode)
    TS._clear_new()
    REF.clear()
    return TS._solve_new(p, turns)


@pytest.mark.parametrize("budget", [25, 3000])
def test_random_problems_rs_equals_reference_at_more_budgets(budget):
    """`test_rd_speed` の予算（無制限・40・300）に加えて、縮めの多い 25 と中くらいの 3000 でも rs ＝ 原文。"""
    CB.EX_STATE_BUDGET = budget
    cuts = 0
    for p in TS._PROBLEMS:
        rs = _solve(p, "rs")
        REF.clear()
        assert repr(rs) == repr(TS._solve_ref(p))
        cuts += int(rs[2]["horizon"] < rs[2]["horizon0"])
    if budget == 25:
        assert cuts >= 5


def test_frames_rs_equals_reference_at_a_tiny_budget():
    """局面 8 つを予算 20（何度も縮める）で: rs ＝ 原文（予算 300000／150 は `test_rd_speed` が見る）。"""
    CB.EX_STATE_BUDGET = 20
    frames = TS._frames()
    assert len(frames) >= 8
    cuts = 0
    for _src, p in frames:
        rs = _solve(p, "rs")
        REF.clear()
        assert repr(rs) == repr(TS._solve_ref(p))
        cuts += int(rs[2]["horizon"] < rs[2]["horizon0"])
    assert cuts >= 3


def test_explicit_horizons_and_both_mode_agree():
    for p in TS._PROBLEMS[:15]:
        for h in (1, 2, 3):
            rs = _solve(p, "rs", h)
            REF.clear()
            assert repr(rs) == repr(TS._solve_ref(p, h))
            assert repr(_solve(p, "both", h)) == repr(rs)


def test_both_mode_passes_and_counts_checks_on_budgeted_solves():
    CB.EX_STATE_BUDGET = 60
    before = RK.STATS["solve_checked"]
    for _src, p in TS._frames():
        rs = _solve(p, "rs")
        assert repr(_solve(p, "both")) == repr(rs)
    for p in TS._PROBLEMS[:20]:
        rs = _solve(p, "rs")
        assert repr(_solve(p, "both")) == repr(rs)
    assert RK.STATS["solve_checked"] - before >= 28


def test_both_mode_raises_on_a_wrong_kernel(monkeypatch):
    """`both` は食い違いを握りつぶさない（核が 1 ビットずれたと仮定して例外になる）——守る側の計算も計画も。"""
    real = RK.guard_rs

    def bad(*a, **k):
        r = real(*a, **k)
        if r is not None:
            r = dict(r, theta=r["theta"] + 1e-15)
        return r
    monkeypatch.setattr(RK, "guard_rs", bad)
    RK.set_mode("both")
    with pytest.raises(AssertionError, match="rd_kernel"):
        CB.rule_guard_plan_ex([(1000.0, 0.0)], 1.0, [1000.0], [1000.0], [0.0], 2, 2)
    monkeypatch.undo()
    real_solve = RK.rd_solve

    def bad_solve(args):
        out = list(real_solve(args))
        out[6] = out[6] + 1e-15                            # τ
        return tuple(out)
    monkeypatch.setattr(RK, "rd_solve", bad_solve)
    with pytest.raises(AssertionError, match="rd_kernel"):
        _solve(TS._PROBLEMS[3], "both")


def test_ref_mode_returns_the_reference_answers():
    """`ref` は原文そのもの（守る側の計算も計画も）。"""
    p = TS._PROBLEMS[7]
    rs = _solve(p, "rs")
    assert repr(_solve(p, "ref")) == repr(rs)
    args = ([(1000.0, 0.0), (2000.0, 0.0)], 1.0, [1000.0, 0.0], [1000.0], [0.0], 2, 3)
    RK.set_mode("rs")
    a = CB.rule_guard_plan_ex(*args)
    RK.set_mode("ref")
    assert repr(CB.rule_guard_plan_ex(*args)) == repr(a)


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


# ---------------------------------------------------------------------------------------------
# L3: 段ごとの数え方・地平の選び方

def test_layer_counts_and_fit_horizon_match_the_reference_dp():
    """段ごとの数（予算での打ち切りを含む）・地平の選び方・段の和＝原文の動的計画が作る状態の数＝Rust の動的計画の数。"""
    E = _E()
    checked = 0
    for p in TS._PROBLEMS[:30] + [p for _s, p in TS._frames()]:
        cards, don, blk, life, ax, _t, lt, dt, arr = p
        ax = dict(ax)
        with CP.defending(TS._view_of(ax)):
            masks = CB._rule_don_masks(cards, blk, life, ax, lt)
            roots = [(xf, m["later_seq"]) for m in masks for xf in TS._mask_firsts(m, ax)]
            rest = tuple(ax.get("rest_blk") or ())
            full = RK.count_layers(cards, don, blk, life, lt, rest, arr, dt, roots, 4, 10 ** 12)
            for lim in (10 ** 12, 5000, 400, 60, 7):
                rs = RK.count_layers(cards, don, blk, life, lt, rest, arr, dt, roots, 4, lim)
                assert list(rs) == _cut_at(full, lim), (lim, full, rs)
                assert E.rd_fit_horizon(rs, lim, 4) == max(1, min(_fit(full, lim), 4))
                checked += 1
            pr = CB._prices_of(ax)
            for h in range(1, 5):
                ref = TS._ref_states(cards, don, blk, life, ax, lt, dt, arr, roots, h)
                d = E.RdDefender(None)
                for xf, ls in roots:
                    RK.guard_rs(cards, don, xf, [tuple(sorted(float(x) for x in s)) for s in ls] or [()], blk, life,
                                h, lt, pr["lam"], pr["lam_net"], pr["mu"], pr["olp"], pr["mlp"], rest, arr, dt,
                                defender=d)
                assert sum(full[:h]) == ref == d.n_states()
                checked += 1
    assert checked >= 200


def _cut_at(sizes, lim):
    """数え方の打ち切り: 合計が `lim` を超えた段で止める（最後の段は超えた 1 つまで）。"""
    out, tot = [], 0
    for n in sizes:
        if tot + n > lim:
            out.append(lim + 1 - tot)
            return out
        tot += n
        out.append(n)
    return out


def _fit(sizes, lim):
    tot, fit = 0, 1
    for t, n in enumerate(sizes):
        tot += n
        if tot > lim:
            break
        fit = t + 1
    return fit


# ---------------------------------------------------------------------------------------------
# 記録した解（ビットの値）

def _golden():
    if not os.path.exists(GOLDEN):
        pytest.fail("記録した解が無い: " + GOLDEN)
    with gzip.open(GOLDEN, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_golden_solves_replay_bit_identically_with_the_kernel():
    cur = (T.OPTION_MODE, T.W_MODE, T.SURV_MODE, T.CBAR_MODE, T.CLOCK_HAND_MODE)
    setters = (T.set_option_mode, T.set_w_mode, T.set_surv_mode, T.set_cbar_mode, T.set_clock_hand_mode)
    for f, v in zip(setters, _TOOL_DEFAULTS):
        f(v)
    try:
        _golden_body()
    finally:
        for f, v in zip(setters, cur):
            f(v)


def _golden_body():
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
        for budget, expect in [(r["budget"], r["expect"])] + [(e["budget"], e["expect"]) for e in r["extra"]]:
            CB.EX_STATE_BUDGET = budget
            TS._clear_new()
            RK.reset_global()
            RK.set_mode("rs")
            with CP.defending(TS._view_of(ax)):
                out = CB.rule_don_solve(cards, don, blk, life, dict(ax), turns, lt, dt, arr)
            assert RK.enc(out) == expect, (r["src"], budget)
            cuts += int(out[2]["horizon"] < out[2]["horizon0"])
    assert cuts >= 5, "小さな予算で地平の縮めを通っている"


# ---------------------------------------------------------------------------------------------
# 計画の覚え書きの鍵

def test_the_plan_store_key_carries_the_kernel_version(tmp_path, monkeypatch):
    """第 3 段: 鍵には核の版（入口の版と `src/theory` の原文のハッシュ）が必ず入る＝核を直せば別の鍵。"""
    p = TS._PROBLEMS[5]
    cards, don, blk, life, ax, _t, lt, dt, arr = p
    st = PS.PlanStore(str(tmp_path / "ps"), CB)
    RK.set_mode("rs")
    k_rs = st.key_of(cards, don, blk, life, ax, None, lt, dt, arr)
    assert "RD_KERNEL_TAG" in vars(CB)
    monkeypatch.setitem(vars(CB), "RD_KERNEL_TAG", lambda: "rs:%d:%s" % (RK.RD_KERNEL_API, "f" * 16))
    assert st.key_of(cards, don, blk, life, ax, None, lt, dt, arr) != k_rs
    monkeypatch.undo()
    assert st.key_of(cards, don, blk, life, ax, None, lt, dt, arr) == k_rs
    # 解き方が読む大域の一覧（鍵に入る切替）が欠けない・数え先は入らない
    names = {n for _lab, n, _src in st.reads}
    assert {"RATE_DECAY_MODE", "EX_STATE_BUDGET", "CUT_PRICER_KEY", "NU_MEAS", "PWR_EPS", "SLOPE_FLOOR"} <= names
    assert not names & {"STATS", "RULE_STATS", "EX_SPEED_STATS"}
