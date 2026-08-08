"""層別アンカー（v35・`option_pair_finetune.load_anchor(own_turn_only=True)`）の検証。

背景: 蒸留アンカー（v33）は dense 一般盤面で候補ネットを base 予測へ MSE で引き戻し
「触っていない挙動の忘却」を防ぐ錘。ただし防御較正（v34/v35）は**ライフ↔手札の交換
レートという評価尺度そのもの**を動かす学習であり、dense 盤面の約4割を占める相手ターン
（防御判断側）の盤面まで釘付けにすると、順位教師の押しが探索木の深部で打ち消される
（実測: 1手先は正解カウンターが +0.117 上なのに探索後 root Q は PASS が上へ逆転）。
層別アンカーは相手ターン盤面をアンカーから除外し、自ターン盤面の挙動だけを固定する。

固定する性質:
  - `IDX_IS_MY_TURN` が encode() の実出力と一致する（append-only 契約の列位置の正）
  - own_turn_only=True は自ターン行だけを残し、False は従来と同一（後方互換）
  - フィルタ後の y はフィルタ後の盤面に対する base 予測（行の対応がズレない）
"""
import os

import conftest  # noqa: F401
import numpy as np
import pytest

import rl_encoder as E
from option_pair_finetune import load_anchor
from opcg_sim.src.learned.value_net import ValueNet

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（学習機構）

N_IDX = 24


def _net(feat_dim):
    return ValueNet(vocab_size=8, d_emb=4, hidden=12, feat_dim=feat_dim, seed=5)


def _write_dense(dirpath, n, my_turn_flags, rng):
    """v1 幅（scalars 14）の合成 dense シャードを書く。field は幅0（feat=14 の小ネット用）。"""
    scalars = rng.normal(size=(n, E.SCALARS_V1)).astype(np.float32)
    scalars[:, E.IDX_IS_MY_TURN] = np.asarray(my_turn_flags, np.float32)
    np.savez_compressed(
        os.path.join(dirpath, "dense_00000.npz"),
        scalars=scalars, field=np.zeros((n, 0), np.float32),
        card_idx=rng.integers(0, 8, size=(n, N_IDX)).astype(np.int64))
    return scalars


def test_idx_is_my_turn_matches_encoder(tmp_path):
    """IDX_IS_MY_TURN の列が encode() の手番フラグと一致する（両手番で反転を確認）。"""
    from cpu_selfplay import build_deck, _load_db
    from opcg_sim.src.core.gamestate import GameManager, Player
    import random
    random.seed(0)
    db = _load_db()
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    vocab = E.build_vocab(db)
    turn_name = gm.turn_player.name
    other = "p2" if turn_name == "p1" else "p1"
    assert E.encode(gm, turn_name, vocab)["scalars"][E.IDX_IS_MY_TURN] == 1.0
    assert E.encode(gm, other, vocab)["scalars"][E.IDX_IS_MY_TURN] == 0.0


def test_own_turn_only_filters_and_y_stays_aligned(tmp_path):
    rng = np.random.default_rng(0)
    flags = [1, 0, 1, 0, 0, 1, 1, 0] * 4          # 32行・自ターン16
    _write_dense(str(tmp_path), 32, flags, rng)
    base = _net(E.SCALARS_V1)

    anchor, y = load_anchor([str(tmp_path)], 1, base, rows=32, own_turn_only=True)
    assert len(anchor["scalars"]) == 16, "自ターン行だけが残っていない"
    assert (anchor["scalars"][:, E.IDX_IS_MY_TURN] > 0.5).all(), "相手ターン行が混入"
    # y はフィルタ後の盤面への base 予測（行対応がズレたら錘が別盤面を引く）
    assert np.allclose(y, base.predict(anchor), atol=1e-6)


def test_default_keeps_all_rows(tmp_path):
    """own_turn_only 未指定は従来挙動（全行）＝後方互換。"""
    rng = np.random.default_rng(1)
    _write_dense(str(tmp_path), 32, [1, 0] * 16, rng)
    anchor, _ = load_anchor([str(tmp_path)], 1, _net(E.SCALARS_V1), rows=32)
    assert len(anchor["scalars"]) == 32
