"""マーク局面シード（mark_seeds.load_mark_boards・p3_loop.selfplay_game の seed_boards/seed_frac・
v5 §4-2）の検証。

失敗局面（実対局の人間マーク）を自己対戦の開始局面に混ぜる機構が、(a) プレイ可能な中盤盤面を
復元し、(b) そこから自己対戦が最後まで進んでラベルを採れ、(c) seed_frac=0 では従来（turn1 開始）と
**完全に挙動不変**（rng 消費順も同一）であることを固定する。value_fn は定数＝生成機構の性質テスト。
"""
import numpy as np
import pytest

import conftest  # noqa: F401
import p3_loop as P
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
from mark_seeds import load_mark_boards
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


@pytest.fixture(scope="module")
def boards():
    db, _ = _setup()
    return load_mark_boards(db)


def test_boards_are_midgame_and_playable(boards):
    """復元プールは非空・各盤面が非終局・合法手あり・中盤（turn>=2）＝turn1 分布に無い局面。"""
    assert len(boards) >= 10, f"シードプールが小さすぎる: {len(boards)}"
    for m in boards:
        assert m.winner is None
        actor = m.turn_player
        assert len(m.get_legal_actions(actor)) > 0
        assert int(getattr(m, "turn_count", 0) or 0) >= 2


def test_boards_are_deterministic():
    """load_mark_boards は決定論（同一プール・同数・同ターン列）＝ワーカー間で同一分布。"""
    db, _ = _setup()
    a = load_mark_boards(db)
    b = load_mark_boards(db)
    assert len(a) == len(b)
    assert [int(x.turn_count) for x in a] == [int(x.turn_count) for x in b]


def test_selfplay_from_seed_board_terminates(boards):
    """seed_frac=1.0 では必ずシード盤面から開始し、最後まで進んで value/policy を採れる。
    turns_left は非負（開始が中盤でも終局ターンからの逆算で正しい）。"""
    db, vocab = _setup()
    game = OPCGGame()
    rng = np.random.default_rng(3)
    vr, pr, w = P.selfplay_game(game, _zero_vf, None, vocab, sims=6, c_puct=1.5, rng=rng,
                                enc_version=1, db=db, seed_boards=boards, seed_frac=1.0)
    assert w is not None and len(vr) == len(pr) > 0
    assert (np.array([r[3] for r in vr]) >= 0).all()


def test_seed_frac_zero_is_behavior_identical(boards):
    """seed_frac=0（既定）は seed_boards を渡しても turn1 開始と完全一致（rng 消費順も不変）＝
    シードを OFF にした本走が従来 v4 生成と bit 単位で同じデータを出す保証。"""
    db, vocab = _setup()
    game = OPCGGame()
    # 同一 seed で「seed_boards 無し」と「seed_boards 有り・frac=0」を比較。
    r1 = P.selfplay_game(game, _zero_vf, None, vocab, sims=6, c_puct=1.5,
                         rng=np.random.default_rng(11), enc_version=1, db=db)
    r2 = P.selfplay_game(game, _zero_vf, None, vocab, sims=6, c_puct=1.5,
                         rng=np.random.default_rng(11), enc_version=1, db=db,
                         seed_boards=boards, seed_frac=0.0)
    assert r1[2] == r2[2]                       # 同一勝者
    assert len(r1[0]) == len(r2[0])             # 同一レコード数
    for a, b in zip(r1[0], r2[0]):
        assert (a[0]["scalars"] == b[0]["scalars"]).all()   # 同一軌跡（先頭局面から一致）
        assert a[3] == b[3]                     # 同一 turns_left
