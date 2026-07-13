"""v4 自己対戦データ生成（p3_loop.selfplay_game の拡張・docs/cpu_v4_plan.md §4-1/4-2）の単体検証。

sticky 世界線（ターン内で決定化 seed 固定）・防御応答の温度延長・q_root/turns_left の記録が
仕様どおりか、決定論（同一 seed → 同一データ）と併せて固定する。value_fn は定数（ネット非依存）
＝生成機構そのものの性質テスト。
"""
import numpy as np
import pytest

import conftest  # noqa: F401
import p3_loop as P
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
import rl_encoder as E

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（学習データ生成機構）

_DB = None


def _setup():
    global _DB
    if _DB is None:
        _DB = _load_db()
    return _DB, E.build_vocab(_DB)


def _zero_vf(state, to_move):
    return 0.0


def _run(seed, game=None, sims=6, **kw):
    db, vocab = _setup()
    game = game or OPCGGame()
    rng = np.random.default_rng(seed)
    return P.selfplay_game(game, _zero_vf, None, vocab, sims=sims, c_puct=1.5, rng=rng,
                           enc_version=1, db=db, **kw)


def test_val_recs_carry_q_root_and_turns_left():
    """val_recs は (enc, who, q_root, turns_left)。q_root∈[-1,1]・turns_left は非負で終局に向かい減少。"""
    vr, pr, w = _run(7, dirichlet_eps=0.1)
    assert w is not None and len(vr) == len(pr) > 0
    q = np.array([r[2] for r in vr]); t = np.array([r[3] for r in vr])
    assert (np.abs(q) <= 1.0 + 1e-9).all()
    assert (t >= 0).all() and t[0] >= t[-1] and t[-1] == 0, "turns_left が終局で0に収束しない"


def test_pack_vdata_schema_v2():
    """pack_vdata は batch スキーマ v2（value/q_root/turns_left が同形）を返す。"""
    vr, pr, w = _run(7, dirichlet_eps=0.1)
    sinks = {"S": [], "F": [], "I": [], "Y": [], "Q": [], "T": []}
    P.merge_val_recs(vr, w, sinks)
    vd = P.pack_vdata(sinks)
    assert set(vd) == {"scalars", "field", "card_idx", "value", "q_root", "turns_left"}
    assert vd["value"].shape == vd["q_root"].shape == vd["turns_left"].shape
    assert set(np.unique(vd["value"])) <= {-1.0, 1.0}


def test_deterministic_from_seed():
    """同一 seed → 同一対局・同一ラベル（sticky/温度延長を入れても決定論は保たれる）。"""
    a = _run(11, dirichlet_eps=0.1)
    b = _run(11, dirichlet_eps=0.1)
    assert a[2] == b[2] and len(a[0]) == len(b[0])
    assert all(abs(x[2] - y[2]) < 1e-12 and x[3] == y[3] for x, y in zip(a[0], b[0]))


class _DetProbe(OPCGGame):
    """determinize に渡る rng の先頭乱数を (turn, name) 付きで記録する（sticky 検証用プローブ）。"""

    def __init__(self):
        super().__init__()
        self.calls = []

    def determinize(self, state, me_name, rng):
        probe = int(np.random.default_rng(int(rng.integers(2 ** 63 - 1))).integers(2 ** 31))
        # ↑ rng から1つ引いた値を「世界の指紋」として記録（同じ seed の Generator なら同じ値）。
        self.calls.append((int(getattr(state, "turn_count", 0) or 0), me_name, probe))
        return super().determinize(state, me_name, np.random.default_rng(probe))


def test_sticky_world_fixed_within_turn():
    """同一 (turn, 手番) 内の全 decide は同じ世界 seed・ターンが変われば引き直す。"""
    game = _DetProbe()
    _run(13, game=game, dirichlet_eps=0.1)
    assert len(game.calls) > 5
    by_key = {}
    for turn, name, probe in game.calls:
        by_key.setdefault((turn, name), set()).add(probe)
    assert all(len(v) == 1 for v in by_key.values()), \
        f"ターン内で世界が引き直されている: { {k: len(v) for k, v in by_key.items() if len(v) > 1} }"
    fingerprints = [next(iter(v)) for v in by_key.values()]
    assert len(set(fingerprints)) > 1, "全ターンが同一世界＝sticky が過剰（seed 更新が効いていない）"


def test_l1_mixed_opponent_seat():
    """(d) 対戦相手の混合: L1 席の決定は policy 教師に入らず、value は q_root=NaN で記録→
    merge で勝敗ラベルへ退化（全有限）。決定論も維持。"""
    import math
    vr, pr, w = _run(21, dirichlet_eps=0.1, l1_seat="p2")
    assert w is not None
    assert all(who == "p1" for _, _, _, who in pr), "L1席の決定が policy 教師に混入"
    p2_vals = [r for r in vr if r[1] == "p2"]
    assert p2_vals and all(math.isnan(r[2]) for r in p2_vals), "L1席の q_root が NaN でない"
    assert any(r[1] == "p1" and math.isfinite(r[2]) for r in vr), "net席の q_root が欠落"
    sinks = {"S": [], "F": [], "I": [], "Y": [], "Q": [], "T": []}
    P.merge_val_recs(vr, w, sinks)
    vd = P.pack_vdata(sinks)
    assert np.isfinite(vd["q_root"]).all(), "merge 後に NaN が残っている（勝敗退化が効いていない）"
    b = _run(21, dirichlet_eps=0.1, l1_seat="p2")
    assert w == b[2] and len(vr) == len(b[0])


def test_battle_response_sampled_with_temperature():
    """steps>=temp_moves でも戦闘応答（SELECT_BLOCKER/SELECT_COUNTER）は温度1でサンプリングされる。"""
    temps = []
    orig = P._sample

    def probe(counts, rng, temp):
        temps.append(temp)
        return orig(counts, rng, temp)

    P._sample = probe
    try:
        _run(7, dirichlet_eps=0.1, temp_moves=0)   # 温度手数0＝メイン行動は全て argmax(temp=0)
    finally:
        P._sample = orig
    assert 0.0 in temps, "argmax サンプリングが観測されない（temp_moves=0 の前提が破れ）"
    assert 1.0 in temps, "戦闘応答の温度延長が効いていない（temp=1.0 の decide が無い）"
