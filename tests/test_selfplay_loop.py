"""self-play ループ（selfplay_loop）の配管スモーク（CI・高速）。

v4c 本走ループの各段が例外なく通り、期待する形のデータ／判定を返すことを最小コストで保証する:
selfplay_game が value/policy サンプルを返す・value 学習が通る・pair_gate が Wilson CI を返す。
"""
import random

import numpy as np

import conftest  # noqa: F401
import rl_fingerprint as FP
import rl_encoder as E
from rl_effective_state import encode_v3, DIM_V3, make_value_fn_for
from mini_set_trial import MLP
from pre_flight4_mcts import mask_fps, COLOR
from deck_generator import DeckGenerator
import selfplay_loop as SL
from cpu_selfplay import _load_db
from opcg_sim.src.learned.adapter import OPCGGame


def test_ckpt_roundtrip(tmp_path):
    """チェックポイントの save→load で nets/buffer/世代/RNG が復元される（再起動耐性）。"""
    vnet = MLP(DIM_V3, seed=1)
    vnet.fit_norm(np.zeros((2, DIM_V3), np.float32), np.array([-1., 1.], np.float32))
    st = dict(vnet=vnet, pnet=None, v_buf=[[("x", 1)]], p_buf=[],
              next_gen=2, best_avg=0.55, best=1,
              py_rng=random.Random(3).getstate(),
              np_rng=np.random.default_rng(3).bit_generator.state)
    path = str(tmp_path / "ck.pkl")
    SL.save_ckpt(path, st)
    ck = SL.load_ckpt(path)
    assert ck["next_gen"] == 2 and ck["best_avg"] == 0.55 and ck["v_buf"] == [[("x", 1)]]
    assert np.array_equal(ck["vnet"].W1, vnet.W1) and ck["vnet"].xmu is not None
    assert SL.load_ckpt(str(tmp_path / "missing.pkl")) is None


def test_wilson_bounds():
    p, lo, hi = SL.wilson(6, 10)
    assert 0.0 <= lo <= p <= hi <= 1.0
    assert SL.wilson(0, 0) == (0.0, 0.0, 1.0)


def test_selfplay_game_and_train_and_gate_smoke():
    db = _load_db()
    vocab = E.build_vocab(db)
    fps = mask_fps(FP.build_fingerprints(db), [COLOR])
    gen = DeckGenerator(db, seed=0)
    game = OPCGGame(fair_determinize=True)
    rng = random.Random(0); nrng = np.random.default_rng(0)
    vnet = MLP(DIM_V3, seed=0)
    vnet.fit_norm(np.zeros((4, DIM_V3), np.float32), np.array([-1., 1., -1., 1.], np.float32))  # 実ループの warm-start 相当（norm初期化）
    # 極小 sims で1局
    vr, pr = SL.selfplay_game(game, make_value_fn_for(vnet, vocab, fps, encode_v3), None,
                              gen, db, vocab, fps, rng, nrng, fast_sims=2, full_sims=4)
    assert isinstance(vr, list) and isinstance(pr, list)
    if vr:  # 決着した場合
        enc, z = vr[0]
        assert enc.shape[0] == DIM_V3 and z in (-1.0, 1.0)
        SL.train_value(vnet, vr, epochs=2, nrng=nrng)   # value 学習が通る
    for ctx, am, tgt in pr:
        assert am.shape[0] == tgt.shape[0]              # policy 教師の整合
        assert abs(float(tgt.sum()) - 1.0) < 1e-6

    # pair_gate（1ペア・極小sims）が Wilson CI 付きで返る
    vf = make_value_fn_for(vnet, vocab, fps, encode_v3)
    lm = lambda m, me: SL._run_mcts(game, vf, None, m, me, 2, 0.0, nrng)[0]
    res = SL.pair_gate(lm, db, vocab, fps, n_pairs=1, ply_cap=400, rng=rng)
    assert set(res.keys()) == set(__import__("heldout_decks").deck_ids())
    for did, (p, lo, n) in res.items():
        assert 0.0 <= lo <= p <= 1.0 and n >= 0
