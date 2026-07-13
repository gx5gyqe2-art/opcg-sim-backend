"""aux 粘り項（cpu_learned._aux_tie_scale・config.SERVE_AUX_TIEBREAK・v5 §4-1）の単体検証。

value が敗勢で −1 付近に飽和すると「本当に延命する守り」と「無意味な守り」が無差別になる
（v4 実測マーク C4・@115／v3 @63 と同根）。v4 学習済みの残りターン補助ヘッド t̂ で飽和域のみ
振幅を減衰 v' = v·max(TERM_FLOOR, 1 − AUX_TIE_DECAY·t̂·sat) することで、敗勢では t̂ が伸びる手
（延命）を、優勢では t̂ が短い手（速い勝ち）を選好する。非飽和域（|v| < AUX_SAT_START）は
恒等＝中間域の較正に影響しない。ゲート OFF（SERVE_AUX_TIEBREAK=False）で従来（v4）に一致。
"""
import numpy as np
import pytest

import conftest  # noqa: F401
from opcg_sim.src.core.cpu_learned import _aux_tie_scale, _value_fn
from opcg_sim.src.learned import config as CFG
from opcg_sim.src.learned.config import AUX_SAT_START, AUX_TIE_DECAY, TERM_FLOOR
from opcg_sim.src.learned.value_net import ValueNet

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（葉評価の性質テスト）


# --- 純関数 _aux_tie_scale の性質 ---

def test_midrange_is_identity():
    """非飽和域（|v| < SAT_START）は t̂ に依らず完全恒等＝中間域の較正は不変。"""
    for v in (0.0, 0.3, -0.5, 0.79, -0.79, AUX_SAT_START):
        for t in (0.0, 5.0, 15.0, 100.0):
            assert _aux_tie_scale(v, t) == v


def test_saturated_loss_prefers_longer_survival():
    """飽和した負け側（v≈−1）は t̂ が大きい（長く粘れる）ほど v' が浅くなる＝延命手を選好。"""
    v = -0.95
    scaled = [_aux_tie_scale(v, t) for t in (0.0, 2.0, 5.0, 10.0)]
    assert scaled[0] == v                       # t̂=0 は減衰なし
    for a, b in zip(scaled, scaled[1:]):
        assert b > a, f"t̂ 増加で v' が浅くならない: {scaled}"
    assert all(s < 0 for s in scaled), "符号（敗勢）は保存される"


def test_saturated_win_prefers_faster_finish():
    """飽和した勝ち側（v≈+1）は t̂ が小さい（早く終わる）ほど v' が高い＝速い勝ちを選好。"""
    v = 0.95
    fast, slow = _aux_tie_scale(v, 1.0), _aux_tie_scale(v, 8.0)
    assert fast > slow > 0


def test_decay_is_floored():
    """減衰は TERM_FLOOR で下げ止まる（巨大 t̂ でも符号情報が消えない）。"""
    v = -1.0
    assert _aux_tie_scale(v, 1000.0) == pytest.approx(v * TERM_FLOOR)
    assert _aux_tie_scale(-v, 1000.0) == pytest.approx(-v * TERM_FLOOR)


def test_sat_ramp_is_continuous():
    """SAT_START 境界で連続（段差なし）・|v|=1 で sat=1（フル減衰）。"""
    t = 5.0
    eps = 1e-9
    below = _aux_tie_scale(AUX_SAT_START - eps, t)
    above = _aux_tie_scale(AUX_SAT_START + eps, t)
    assert above == pytest.approx(below, abs=1e-6)
    assert _aux_tie_scale(1.0, t) == pytest.approx(max(TERM_FLOOR, 1 - AUX_TIE_DECAY * t))


def test_negative_t_hat_clamped():
    """t̂<0（学習初期・外挿の異常値）は 0 に切り上げ＝増幅（|v'|>|v|）は起きない。"""
    assert _aux_tie_scale(-0.95, -3.0) == -0.95
    for v in (-1.0, -0.9, 0.9, 1.0):
        for t in (-5.0, 0.0, 3.0, 50.0):
            assert abs(_aux_tie_scale(v, t)) <= abs(v) + 1e-12


# --- serve 配線（_value_fn のゲートと forward 共有） ---

def _tiny_net_and_batchable():
    """W2t を非ゼロにした極小 ValueNet と、encode 済みの疑似バッチを返す。"""
    net = ValueNet(vocab_size=8, d_emb=4, hidden=8, feat_dim=6, seed=1)
    net.W2t = np.abs(np.random.default_rng(0).standard_normal((8, 1))) * 0.5
    batch = {
        "scalars": np.zeros((1, 3)),
        "field": np.ones((1, 1, 3)),
        "card_idx": np.array([[1, 2, 3]]),
    }
    return net, batch


def test_predict_with_aux_matches_separate_calls():
    """predict_with_aux は predict / predict_aux と bit-identical（forward 1回の共有のみ）。"""
    net, batch = _tiny_net_and_batchable()
    pred, aux = net.predict_with_aux(batch)
    assert pred == pytest.approx(net.predict(batch))
    assert aux == pytest.approx(net.predict_aux(batch))


def test_value_fn_gate_off_is_identity(monkeypatch):
    """SERVE_AUX_TIEBREAK=False（および明示 OFF）では従来の predict と完全一致。"""
    net, batch = _tiny_net_and_batchable()

    class _St:
        winner = None
    st = _St()

    def fake_encode(state, to_move, vocab, version=1):
        return {k: batch[k][0] for k in batch}
    monkeypatch.setattr("opcg_sim.src.core.cpu_learned.E.encode", fake_encode)

    raw = float(net.predict(batch)[0])
    off = _value_fn(net, vocab=None, aux_tiebreak=False)(st, "p1")
    assert off == raw
    monkeypatch.setattr(CFG, "SERVE_AUX_TIEBREAK", False)
    assert _value_fn(net, vocab=None)(st, "p1") == raw
    monkeypatch.setattr(CFG, "SERVE_AUX_TIEBREAK", True)
    on = _value_fn(net, vocab=None)(st, "p1")
    t_hat = float(net.predict_aux(batch)[0]) * CFG.V4_TURNS_SCALE
    assert on == pytest.approx(_aux_tie_scale(raw, t_hat))


def test_value_fn_terminal_unaffected():
    """終局（winner 確定）は ±1 のまま＝粘り項は非終局の葉のみに作用する。"""
    net, _ = _tiny_net_and_batchable()

    class _St:
        winner = "p1"
    fn = _value_fn(net, vocab=None)
    assert fn(_St(), "p1") == 1.0
    assert fn(_St(), "p2") == -1.0
