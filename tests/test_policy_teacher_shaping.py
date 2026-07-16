"""v7 教師整形（案D/E/F）の機構健全性: 生成prior平坦化・教師ラベル平滑化・Q補正教師。

docs/cpu_v7_plan.md。確定した原因＝「policy 教師が prior のエコー（実測相関 0.93）で、
value 無差別な決定点の prior が訓練ノイズで独立に酔歩する」（docs/reports/seesaw_probe_20260716.md）
への対策3点。すべて **0 で恒等（従来と bit 一致）** が後方互換の要（既存 run に影響しない）。
"""
import numpy as np
import pytest

import conftest  # noqa: F401
import p3_loop as P
from opcg_sim.src.learned.policy import smooth_target

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（学習パイプライン内部機構）


# --- 案D: priors_fn_of(flatten) ---
class _FakePolicy:
    def __init__(self, p):
        self._p = np.asarray(p, dtype=np.float64)

    def priors(self, ctx, am):
        return self._p.copy()


def _priors_with(flatten, base=(0.9, 0.1, 0.0)):
    # state/vocab は使わない fake で priors_fn の変換だけを固定する
    fn = P.priors_fn_of(_FakePolicy(base), vocab=None, enc_version=1, flatten=flatten)

    class _S:   # pending_actor_action と encode を迂回する最小スタブ
        def pending_actor_action(self):
            return ("p1", "MAIN")

    import p3_loop
    orig_ctx, orig_am = p3_loop.state_context, p3_loop.legal_action_matrix
    p3_loop.state_context = lambda *a, **k: None
    p3_loop.legal_action_matrix = lambda *a, **k: None
    try:
        return fn(_S(), [object()] * len(base))
    finally:
        p3_loop.state_context, p3_loop.legal_action_matrix = orig_ctx, orig_am


def test_flatten_zero_is_identity():
    p = _priors_with(0.0)
    assert np.allclose(p, [0.9, 0.1, 0.0])


def test_flatten_mixes_uniform_and_keeps_normalization():
    p = _priors_with(0.3)
    expect = 0.7 * np.array([0.9, 0.1, 0.0]) + 0.3 / 3
    assert np.allclose(p, expect)
    assert abs(p.sum() - 1.0) < 1e-9
    assert p.min() >= 0.3 / 3 - 1e-12   # 平坦化は prior の床＝どの手も探索から消えない


# --- 案F: q_reweight ---
def test_q_reweight_beta_zero_is_visit_normalization():
    N = np.array([30, 10, 0]); Q = np.array([0.1, 0.5, 0.9])
    assert np.allclose(P.q_reweight(N, Q, 0.0), N / N.sum())


def test_q_reweight_boosts_higher_q_and_keeps_unvisited_zero():
    N = np.array([30, 10, 0]); Q = np.array([0.0, 0.5, 0.9])
    t = P.q_reweight(N, Q, 2.0)
    base = N / N.sum()
    assert t[1] > base[1], "Q が高い手の教師質量が増えていない"
    assert t[0] < base[0], "Q が低い手の教師質量が減っていない"
    assert t[2] == 0.0, "読んでいない手（N=0）を持ち上げてはいけない"
    assert abs(t.sum() - 1.0) < 1e-9


# --- 案E: smooth_target ---
def test_smooth_zero_is_identity():
    tg = np.array([1.0, 0.0, 0.0])
    assert np.allclose(smooth_target(tg, 0.0), tg)


def test_smooth_floors_and_preserves_normalization():
    tg = np.array([1.0, 0.0, 0.0])
    s = smooth_target(tg, 0.06)
    assert abs(float(np.sum(s)) - 1.0) < 1e-9
    assert s.min() >= 0.06 / 3 - 1e-12, "床が敷かれていない（盲点の不可逆化を防げない）"
    assert s[0] > s[1] == s[2], "順位が保存されていない"
