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

#: 器の既定（import 時）。記録した解は器の既定で解いたもの
_TOOL_DEFAULTS = (T.OPTION_MODE, T.W_MODE, T.SURV_MODE, T.CBAR_MODE)

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


# ---------------------------------------------------------------------------------------------
# 全体の解き方（`rule_don_solve`）: rs と原文のビットの一致

def _solve(p, mode, turns=None):
    RK.set_mode(mode)
    TS._clear_new()
    REF.clear()
    return TS._solve_new(p, turns)


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

# ---------------------------------------------------------------------------------------------
# 記録した解（ビットの値）

def _golden():
    if not os.path.exists(GOLDEN):
        pytest.fail("記録した解が無い: " + GOLDEN)
    with gzip.open(GOLDEN, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_golden_solves_replay_bit_identically_with_the_kernel():
    cur = (T.OPTION_MODE, T.W_MODE, T.SURV_MODE, T.CBAR_MODE)
    setters = (T.set_option_mode, T.set_w_mode, T.set_surv_mode, T.set_cbar_mode)
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
