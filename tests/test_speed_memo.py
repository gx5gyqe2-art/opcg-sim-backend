"""**値を変えない高速化**（2026-10-01・N-3 採用の 3・`theory_order.SPEED_MEMO`）——覚え書きと厳密な打ち切りの有無で
値が 1 ビットも変わらないこと。

高速化の中身（どれも純関数の覚え書きか、最大を変えない打ち切り）:

* `attack_value_don` の覚え書き（鍵＝引数＋値段の文脈＋費用曲線の切替）と、上限（`min(受ける値, 各ブロッカーの ν)`）に
  届いた枚数での打ち切り（それより多い枚数は `上限 − k'δ` 未満）。
* `option_value` の分布の同じ盤面（体の並びは結果に効かない）を 1 度だけ読む（足す順はそのまま）。
* `attack_stream` の相手の体の `ν`（旧の値段で読む＝値段の文脈に依らない）の覚え書き・`turn_weights` の覚え書き。
* 手札の読み直し（`hand_plan.inflow_item`）の `use_value` と残りの手札の量の覚え書き（1 つの手札の読みの間だけ）。

押さえること: 乱数の標本（攻撃の値・選択肢の価値を色々な値段の文脈で）と実記録（`tests/fixtures/f_identity/rec`・
守り手の枠の 1 枚 1 役の価値 `V` と `L(k)`・交点の橋と線形の橋の出力）で、切替 on と off が厳密に等しい。

**基盤健全性**（`cpu_infra`）。
"""
import json
import os
import random
import subprocess
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
import theory_order as T  # noqa: E402

_REC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "f_identity", "rec")


def _clear():
    for d in (T._AVD_MEMO, T._BODIES_MEMO, T._TW_MEMO, T._OPTION_CACHE):
        d.clear()


class _Curve:
    def __init__(self, g):
        self.gbar = g


class _View:
    """1 枚あたり一定の値段の窓（`CutView(kind="avg")` と同じ形）。"""
    kind = "avg"

    def __init__(self, g):
        self.curve = _Curve(g)
        self.m = 0.0

    def price(self, k, mu=None):
        k = float(k)
        return 0.0 if k <= 0.0 else float(k * self.curve.gbar)


def _both(fn):
    before = T.SPEED_MEMO
    try:
        T.SPEED_MEMO = False
        _clear()
        off = fn()
        T.SPEED_MEMO = True
        _clear()
        on = fn()
        on2 = fn()                     # 覚えた値を読んでも同じ
    finally:
        T.SPEED_MEMO = before
        _clear()
    return off, on, on2


@pytest.mark.parametrize("take", ["mu", "gbar"])
def test_attack_value_don_is_bit_identical_with_and_without_the_memo(take):
    rng = random.Random(20261001)
    cases = []
    for _ in range(400):
        p = 1000.0 * rng.randint(0, 12)
        tp = 1000.0 * rng.randint(2, 10)
        lead = rng.random() < 0.5
        nu = None if lead or rng.random() < 0.2 else rng.uniform(0.0, 0.3)
        blk = [(1000.0 * rng.randint(1, 9), rng.uniform(0.0, 0.25)) for _ in range(rng.randint(0, 2))]
        g = None if rng.random() < 0.3 else rng.uniform(0.0, 0.12)
        cases.append((p, tp, lead, nu, blk, g))
    old_take = CP.CUT_TAKE_MODE
    CP.set_cut_take_mode(take)
    try:
        def run():
            out = []
            for p, tp, lead, nu, blk, g in cases:
                with CP.defending(None if g is None else _View(g)):
                    out.append(T.attack_value_don(p, tp, lead, T.THETA, T.MU, nu_target=nu, blockers=blk))
            return out
        off, on, on2 = _both(run)
    finally:
        CP.set_cut_take_mode(old_take)
    assert off == on == on2
    assert len(set(off)) > 50                                   # 標本が値の幅を持つ


def test_option_value_and_nu_are_bit_identical_with_and_without_the_memo():
    rng = random.Random(7)
    cases = [(1000.0 * rng.randint(2, 12), float(rng.choice([4000, 5000, 6000])), rng.uniform(1.0, 5.0),
              None if rng.random() < 0.3 else rng.uniform(0.01, 0.1)) for _ in range(40)]

    def run():
        out = []
        for p, olp, r, g in cases:
            with CP.defending(None if g is None else _View(g)):
                out.append(T.option_value(p, olp, r))
                out.append(T.nu_of(p, olp, r, is_blocker=(p > 5000)))
                out.append(T.attack_stream(p, olp, r, opp_chars=[(5000.0, True), (3000.0, False)]))
        return out
    off, on, on2 = _both(run)
    assert off == on == on2


def _frames():
    from opcg_sim.learned.train import plan_labels as PL
    import guard_afford as GA
    import theory_bridge as TB
    from theory_bridge import POL_COLS, ROW_COLS, _extra
    idx2cid = {i: c for c, i in GA._vocab().items()}
    cards = PL.Cards()
    out = []
    for rows, pol, ex, L, ptr, idx in PL.iter_games([_REC], row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        for i in idx:
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t >= 1 and PL.is_own_turn(w, t) and TB.is_decision_row(rows, pol, L, ptr, i):
                out.append((ex["sc"][i], ex["tok"][i], ex["ci"][i]))
    return out, idx2cid, cards


def test_joint_hand_value_on_real_frames_is_bit_identical_with_and_without_the_memo():
    """実記録の守り手の枠（自席ターンの判断の行・読み直しのある手札を含む）で、1 枚 1 役の価値 `V`（全部の部分集合）・
    `L(k)`・`ḡ` が切替 on と off で厳密に等しい。"""
    import theory_bridge as TB
    frames, idx2cid, cards = _frames()
    assert len(frames) >= 40
    from opcg_sim.learned.train import plan_labels as PL  # noqa: F401
    n_reread = 0

    def run():
        nonlocal n_reread
        n_reread = 0
        out = []
        for sc, tok, ci in frames:
            cv = CP.curve_of_row(sc, tok, ci, idx2cid, cards, don=float(np.asarray(sc)[T.SC_MY_DON]))
            if cv is None:
                out.append(None)
                continue
            n_reread += int(not cv.valuer.monotone)
            vals = [cv.valuer.value(frozenset(k for k in range(cv.valuer.n) if m >> k & 1))[0]
                    for m in range(1 << cv.valuer.n)]
            out.append((vals, cv.L(), cv.gbar, cv.reserve, tuple(cv.xs_future)))
        _ = TB
        return out
    off, on, on2 = _both(run)
    assert off == on == on2
    assert n_reread >= 3                                        # 読み直しのある手札（覚え書きが効く手札）を含む


_RUNNER = r"""
import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "tests")); sys.path.insert(0, os.path.join(os.getcwd(), "tests", "scripts"))
import _bootstrap  # noqa
import theory_order as T, cut_price as CP, crossing_bridge as CB, theory_bridge as TB
T.SPEED_MEMO = (sys.argv[3] == "on")
CP.set_cut_price_mode("joint"); CP.set_cut_take_mode("gbar")
rec, out, _sw, which = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
if which == "cb":
    rc = CB.main(["--in", rec, "--out", out])
else:
    rc = TB.main(["--in", rec, "--boot-reps", "10", "--out", out])
sys.exit(rc)
"""


@pytest.mark.parametrize("which", ["cb", "tb"])
def test_bridges_on_real_records_are_identical_with_and_without_the_memo(tmp_path, which):
    """交点の橋と線形の橋（`joint`・`gbar`）の出力が、切替 on と off で秒数を除いて一致する（別プロセス）。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, OPCG_LOG_SILENT="1")
    got = {}
    for sw in ("off", "on"):
        out = str(tmp_path / ("%s_%s.json" % (which, sw)))
        r = subprocess.run([sys.executable, "-c", _RUNNER, _REC, out, sw, which], cwd=root, env=env,
                           capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr[-2000:]
        d = json.loads(open(out, encoding="utf-8").read())
        d.pop("seconds", None)
        got[sw] = d
    assert got["off"] == got["on"]
