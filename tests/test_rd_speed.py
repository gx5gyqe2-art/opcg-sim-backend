"""**RD-speed（2026-10-04）**: `rule_don` の守る側の計算・攻め手の計画を速くした変更が**値を 1 ビットも変えない**ことを固める。

比べる相手は速くする前の解き方の原文（`tests/harness/rule_don_ref.py`）。同じ問題・同じ切替で `repr` が同じ
（浮動小数の最後の桁まで）・地平の縮め（計算の予算）も同じ地平を選ぶ・冷たい／温まった／順を入れ替えた実行で同じ・
ディスクの覚え書き（`plan_store.py`）から読んでも同じ、を確かめる。問題は乱数で作ったものと、実の記録（w41）と
合成の記録（w39/w42）から取った小さな局面（`tests/fixtures/rd_speed_frames.json`）。

**Rust 化・第 3 段（2026-10-05）**: 速くした Python の解き方は消した＝ここで比べる「新」は Rust の核（`OPCG_RD_KERNEL=rs`・
既定）。原文と比べる約束はそのまま。
"""
import hashlib
import inspect
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
import theory_order as T  # noqa: E402

_FRAMES = os.path.join(_HERE, "fixtures", "rd_speed_frames.json")


class _AvgView:
    """`cut_price.CutView`（`avg`）と同じ値段の窓: 切る札 1 枚 = `ḡ`（記録の局面の `actx["cp"]` から）。"""
    kind = "avg"

    def __init__(self, g):
        self.curve = type("C", (), {"gbar": float(g)})()

    def price(self, k, mu=None):
        k = float(k)
        return 0.0 if k <= 0.0 else float(k * self.curve.gbar)


def _tup(v):
    return tuple(_tup(x) for x in v) if isinstance(v, (list, tuple)) else v


def _frames():
    with open(_FRAMES, encoding="utf-8") as fh:
        raw = json.load(fh)
    out = []
    for f in raw:
        ax = dict(f["actx"])
        ax["key"] = _tup(ax["key"]); ax["cp"] = _tup(ax["cp"]); ax["rest_blk"] = _tup(ax["rest_blk"])
        out.append((f["src"], (f["cards"], f["don"], f["blk"], f["life"], ax, None, _tup(f["life_types"]),
                               _tup(f["draw_types"]), _tup(f["arrive"]))))
    return out


def _rand_problem(rng, i):
    """小さな乱数の問題（`attacker_ctx` と同じ形の財布・守る側の札・ブロッカー・ライフの札・引く札）。"""
    budget = rng.randint(0, 3)
    xs = (-1000.0, 0.0, 1000.0, 2000.0)
    n_att = rng.randint(1, 3)
    att1 = [(0, rng.choice(xs))] + [(5 + q, rng.choice(xs)) for q in range(n_att - 1)]
    later = list(att1) + ([(9, rng.choice(xs))] if rng.random() < 0.3 else [])
    cand = []
    for q in range(rng.randint(0, 3)):
        body = rng.random() < 0.6
        cand.append((rng.randint(0, 2), {"atk": round(rng.random() * 0.1, 3), "eff": round(rng.random() * 0.05, 3)},
                     rng.choice(xs) if body else None, body and rng.random() < 0.3))
    kmax = 2
    a_tab = [round(rng.random() * 0.02, 4) for _ in range(budget + 2)]
    e_tab = [round(rng.random() * 0.01, 4) for _ in range(budget + 2)]
    ax = {"budget": budget, "att1": att1, "later": later, "cand": cand, "price": [[0.0] * (kmax + 1) for _ in att1],
          "flow": [a + e for a, e in zip(a_tab, e_tab)], "kmax": kmax, "olp": 5000.0, "mlp": 5000.0,
          "lam": T.LAM, "lam_net": T.THETA * T.MU, "mu": T.MU, "cp": (None, None),
          "ds": [float(budget + (q % 2)) for q in range(30)], "a_tab": a_tab, "ar_tab": [0.0] * len(a_tab),
          "e_tab": e_tab, "jmax": 30, "no_attack_now": rng.random() < 0.15, "lead_bare": round(rng.random() * 0.1, 3),
          "chars_bare": round(rng.random() * 0.05, 3), "rest_blk": (500.0,) if rng.random() < 0.3 else (),
          "blk_a": None, "theta_p": T.THETA}
    if rng.random() < 0.15 and cand:
        play = (0,) if cand[0][0] <= budget else ()
        left = budget - (cand[0][0] if play else 0)
        rng.randint(0, min(1, left))        # 乱数の列を変えないための空引き（旧 `rule_don_purse` の `fixed` の枚数・2026-10-05 に形は削除）
    ax["key"] = ("rd-test", i, repr(sorted((k, v) for k, v in ax.items())))
    kinds = ((1000.0, 0.0), (2000.0, 0.0), (1000.0, 1.0))
    cards = [rng.choice(kinds) for _ in range(rng.randint(0, 3))]
    blk = [rng.choice((-1000.0, 0.0, 1000.0)) for _ in range(rng.randint(0, 2))]
    lt = ((1000.0, 0.0, 0.3), (2000.0, 0.0, 0.2)) if rng.random() < 0.7 else ()
    dt = ((1000.0, 0.0, 0.25), (2000.0, 1.0, 0.25)) if rng.random() < 0.6 else ()
    arrive = (1000.0,) if rng.random() < 0.2 else ()
    return (cards, float(rng.randint(0, 2)), blk, float(rng.randint(0, 2)), ax, None, lt, dt, arrive)


def _clear_new():
    CB._RULE_DON_CACHE.clear()
    RK.reset_global()


def _view_of(ax):
    return _AvgView(ax["cp"][1]) if ax["cp"][0] is not None else None


def _solve_new(p, turns=None):
    cards, don, blk, life, ax, _t, lt, dt, arr = p
    with CP.defending(_view_of(ax)):
        return CB.rule_don_solve(cards, don, blk, life, dict(ax), turns, lt, dt, arr)


def _solve_ref(p, turns=None):
    cards, don, blk, life, ax, _t, lt, dt, arr = p
    with CP.defending(_view_of(ax)):
        return REF.solve(cards, don, blk, life, dict(ax), turns, lt, dt, arr)


@pytest.fixture
def _budget():
    old = CB.EX_STATE_BUDGET
    yield
    CB.EX_STATE_BUDGET = old
    _clear_new(); REF.clear()


_PROBLEMS = [_rand_problem(random.Random(1000 + i), i) for i in range(40)]


def _mask_firsts(m, actx):
    """1 つの出す札の組で守る側の計算に渡る今のターンの攻撃の並びの全部（付与 0 の 1 本 ＋ 列挙する付与の全部）。
    （第 3 段で消した `crossing_bridge._mask_firsts` の写し・状態の数を比べるためだけ。）"""
    att1 = actx["att1"]
    no_now = bool(actx.get("no_attack_now"))
    hits1, caps, b = m["hits1"], m["caps"], m["b"]
    n = len(att1)

    def xf_of(ks):
        return () if no_now else (tuple(float(att1[q][1]) + 1000.0 * ks[q] for q in range(n)) + tuple(hits1[n:]))
    out = {xf_of([0] * n)}
    ks = [0] * n

    def visit(i, left):
        if i == n:
            out.add(xf_of(ks))
            return
        for k in range(0, min(caps[i], left) + 1):
            ks[i] = k
            visit(i + 1, left - k)
        ks[i] = 0
    visit(0, b)
    return out


def _ref_states(cards, don, blk, life, ax, lt, dt, arr, roots, h):
    """原文の守る側の計算が地平 `h` の 1 回の試行（全ての計画の根・覚え書きを共有）で作る状態の数。"""
    R = RK.ref_module()
    pr = CB._prices_of(ax)
    R._EX_USED["memo"], R._EX_USED["limit"] = {}, None
    try:
        for xf, ls in roots:
            R.rule_guard_plan_ex(cards, don, xf, None, blk, life, h, lt, pr, later_seq=ls,
                                 rest_blk=tuple(ax.get("rest_blk") or ()), arrive_blk=arr, draw_types=dt)
        return len(R._EX_USED["memo"])
    finally:
        R._EX_USED["memo"], R._EX_USED["limit"] = None, None


def test_layer_count_equals_the_states_each_horizon_actually_creates(_budget):
    """**数え方**（Rust の `rd_count_layers`）の段 `0..h−1` の和は、地平 `h` の試行が本当に作る状態の数（原文の守る側の
    計算の覚え書きの大きさ）とちょうど同じ——だから「超える地平を解いて捨てる」代わりに数えて選んでも、選ぶ地平は 1 つも
    変わらない（第 3 段: 比べる相手を速くした Python の数え方から原文の動的計画に替えた）。"""
    checked = 0
    for p in _PROBLEMS[:25] + [p for _s, p in _frames()]:
        cards, don, blk, life, ax, _t, lt, dt, arr = p
        ax = dict(ax)
        with CP.defending(_view_of(ax)):
            masks = CB._rule_don_masks(cards, blk, life, ax, lt)
            roots = [(xf, m["later_seq"]) for m in masks for xf in _mask_firsts(m, ax)]
            sizes = RK.count_layers(cards, don, blk, life, lt, tuple(ax.get("rest_blk") or ()), arr, dt,
                                    roots, 4, 10 ** 12)
            for h in range(1, 5):
                assert sum(sizes[:h]) == _ref_states(cards, don, blk, life, ax, lt, dt, arr, roots, h)
                checked += 1
    assert checked >= 100


def test_the_plan_store_returns_identical_plans_and_never_a_stale_one(_budget, tmp_path, monkeypatch):
    """ディスクの覚え書き: 2 回目は解かずに同じ `repr` を返す。鍵には解き方の版・`tests/scripts` の原文のハッシュ・
    切替の値が入る——どれかが違えば読まない（古い値を返さない）。"""
    p = _PROBLEMS[3]
    st = PS.PlanStore(str(tmp_path / "ps"), CB)
    monkeypatch.setattr(CB, "PLAN_STORE", st)
    _clear_new()
    a = _solve_new(p)
    _clear_new()
    b = _solve_new(p)
    assert repr(a) == repr(b) and st.hits == 1 and st.puts == 1
    cards, don, blk, life, ax, _t, lt, dt, arr = p
    k0 = st.key_of(cards, don, blk, life, ax, None, lt, dt, arr)
    monkeypatch.setattr(CB, "RATE_DECAY_MODE", "off" if CB.RATE_DECAY_MODE == "ko" else "ko")
    k1 = st.key_of(cards, don, blk, life, ax, None, lt, dt, arr)
    monkeypatch.undo()
    monkeypatch.setattr(CB, "SOLVER_VERSION", CB.SOLVER_VERSION + "-x")
    k2 = st.key_of(cards, don, blk, life, ax, None, lt, dt, arr)
    monkeypatch.undo()
    monkeypatch.setattr(st, "src", "changed-source")
    k3 = st.key_of(cards, don, blk, life, ax, None, lt, dt, arr)
    monkeypatch.undo()
    k4 = st.key_of(cards, don, blk, life, dict(ax, lead_bare=ax["lead_bare"] + 1e-12), None, lt, dt, arr)
    assert len({k0, k1, k2, k3, k4}) == 5
    assert st.key_of(cards, don, blk, life, ax, None, lt, dt, arr) == k0


def test_the_plan_store_key_is_shared_across_tools_and_module_names(_budget, tmp_path, monkeypatch):
    """器をまたいで共有できる鍵: 解き方が読まない切替（`PRE_SETTLE_MODE`・器ごとに既定が違う）では変わらず、
    `crossing_bridge` が `__main__` として走っても（別の名前で読み込んだ 2 つ目の写し）同じ鍵。解き方が読む切替は入る。"""
    import importlib.util
    p = _PROBLEMS[5]
    cards, don, blk, life, ax, _t, lt, dt, arr = p
    st = PS.PlanStore(str(tmp_path / "ps"), CB)
    names = {n for _lab, n, _src in st.reads}
    assert {"RATE_DECAY_MODE", "EX_STATE_BUDGET", "CUT_PRICER_KEY", "NU_MEAS"} <= names
    assert "PRE_SETTLE_MODE" not in names
    k0 = st.key_of(cards, don, blk, life, ax, None, lt, dt, arr)
    monkeypatch.setattr(CB, "PRE_SETTLE_MODE", "on" if CB.PRE_SETTLE_MODE == "off" else "off")
    assert st.key_of(cards, don, blk, life, ax, None, lt, dt, arr) == k0
    monkeypatch.undo()
    spec = importlib.util.spec_from_file_location("cb_as_main_copy", CB.__file__)
    cb2 = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "cb_as_main_copy", cb2)
    spec.loader.exec_module(cb2)
    st2 = PS.PlanStore(str(tmp_path / "ps2"), cb2)
    assert st2.key_of(cards, don, blk, life, ax, None, lt, dt, arr) == k0


#: 解き方の関数の原文の指紋（版ごと）。解き方を変えたら `crossing_bridge.SOLVER_VERSION` を上げ、ここに新しい指紋を足す
#: （ディスクの覚え書きの古い値を読まないため・値を変えたなら `rule_don_ref.py` も同じ変更で更新する）。
#: **rd-speed-3（Rust 化・第 3 段）から Rust の核の原文のハッシュ（`rd_kernel.source_hash()`）も指紋に入る**——
#: `src/theory` を直して版を上げ忘れると落ちる（設計 §6.4）。
SOLVER_FINGERPRINTS = {
    "rd-speed-1": "84e3b483d52b4b2c",
    "rd-speed-2": "45dc6aba0ff5c6b2",     # switch cleanup A（2026-10-05）: 死んだ切替の枝を削除（値は不変）
    "rd-speed-3": "9e21fec15160dcda",     # Rust 化・第 3 段（2026-10-05）: 速くした Python の解き方を消した（値は不変）
    "rd-speed-4": "1b96139ac93cffad",     # 全移植・段 1／2（2026-10-06）: `src/theory` に葉を足した（解き方の値は不変）
}
_SOLVER_FUNCS = ("rule_guard_plan_ex", "_prices_of", "_attach_gain", "rules_steps", "walk_crossing", "model_horizon",
                 "tau_grow", "rule_don_solve", "_rd_solve_args", "_rd_run", "_rule_don_masks")
#: Rust の核の継ぎ目（`rd_kernel`）のうち値に触れる関数
_KERNEL_FUNCS = ("guard_rs", "res_dict", "race_cap", "rd_solve")


def solver_fingerprint():
    h = hashlib.sha256()
    for name in _SOLVER_FUNCS:
        h.update(inspect.getsource(getattr(CB, name)).encode())
    for name in _KERNEL_FUNCS:
        h.update(inspect.getsource(getattr(RK, name)).encode())
    h.update((RK.source_hash() or "").encode())
    return h.hexdigest()[:16]


def test_the_solver_version_is_bumped_when_the_solver_changes():
    """解き方の関数の原文（と Rust の核の原文）が変わったのに `SOLVER_VERSION` を上げていなければ落ちる（ディスクの
    覚え書きの鍵にも `tests/scripts` 全部の原文のハッシュと核の版が入るので古い値は返らないが、版で変更を明示する約束）。"""
    assert CB.SOLVER_VERSION in SOLVER_FINGERPRINTS, "新しい SOLVER_VERSION の指紋を SOLVER_FINGERPRINTS に足す"
    assert SOLVER_FINGERPRINTS[CB.SOLVER_VERSION] == solver_fingerprint(), \
        "解き方が変わった: crossing_bridge.SOLVER_VERSION を上げて指紋 %s を足す" % solver_fingerprint()


@pytest.mark.parametrize("tool", ["crossing_bridge", "theory_bridge", "relative_ledger", "transition_ledger",
                                  "win_calib"])
def test_tool_help_does_not_crash(tool, capsys):
    """`--help` が落ちない（argparse の説明の `%` は `%%` に書く・`crossing_bridge` は落ちていた）。"""
    import importlib
    m = importlib.import_module(tool)
    with pytest.raises(SystemExit) as e:
        m.main(["--help"])
    assert e.value.code == 0
    assert "usage" in capsys.readouterr().out
