"""ネット付属 vocab（`value_net.vocab_ids`）の回帰テスト（2026-07-15 索引ズレ事故の恒久対策）。

カードDB更新（+32枚）で `build_vocab`（card_id ソート）が**途中挿入**され、既存371枚の idx が
+2 ズレて学習済み Emb/EffF 行との対応が破壊（PRB/ST デッキの無言の品質破壊）＋新カードが
範囲外参照でクラッシュした（docs/reports/net_vocab_pinning_20260715.md）。対策＝訓練時の
card_id→idx 対応を**ネット自身が持ち**（vocab_ids）、serve/生成はそれで符号化する。
無ければ実プレイのクラッシュ/誤評価を見逃すため**必須テスト**（cpu_infra ではない）。
"""
import os

import numpy as np
import pytest

import conftest  # noqa: F401
import rl_net as RN
import rl_encoder as E
from cpu_selfplay import _load_db

_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "opcg_sim", "data", "learned")


@pytest.fixture(scope="module")
def db():
    return _load_db()


def test_bundled_nets_carry_vocab_ids():
    """同梱 gen2〜gen5 の value npz は vocab_ids を持ち、Emb 行数と一致する。"""
    for g in (2, 3, 4, 5):
        v = RN.ValueNet.load(os.path.join(_MODELS, f"gen{g}_value.npz"))
        assert v.vocab_ids, f"gen{g} に vocab_ids が無い"
        assert len(v.vocab_ids) == v.Emb.shape[0] - 1, f"gen{g} の vocab_ids と Emb 行数が不一致"


def test_engine_pins_trained_indices():
    """既定エンジンの符号化はネット付属 vocab＝訓練時 idx を維持する（DB増加でズレない）。

    PRB01-001 は訓練時 idx=2282（現行DBソートだと 2284 にズレる）。ネットが知らない新カード
    （ST31-001）は vocab に**含めない**＝encode の `_vidx` が UNK=0 に落とす＝範囲外参照なし。"""
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    eng = LearnedEngine()
    assert eng.vocab.get("PRB01-001") == 2282, "訓練時 idx が復元されていない（ズレ再発）"
    assert "ST31-001" not in eng.vocab, "訓練後に追加されたカードは UNK 扱いのはず"
    assert max(eng.vocab.values()) == eng.vnet.Emb.shape[0] - 1, "idx が Emb 範囲を超えうる"


def test_extend_appends_and_preserves_output(db):
    """学習側の拡張（RN.extend_to_vocab）は既存 idx 不変・新カード末尾追記・既存盤面の出力恒等。"""
    net = RN.ValueNet.load(os.path.join(_MODELS, "gen5_value.npz"))
    old_ids = list(net.vocab_ids)
    # 実盤面ベースの batch（符号化は旧 vocab＝既存カードのみ）で予測を固定する。
    import cpu_arena
    from opcg_sim.src.core.gamestate import GameManager, Player
    import random
    random.seed(9)
    l1, c1 = cpu_arena.build_deck(db, "p1"); l2, c2 = cpu_arena.build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2)); m.start_game()
    vocab_old = E.vocab_from_ids(old_ids)
    enc = E.encode(m, "p1", vocab_old, version=4)
    batch = {k: enc[k][None, ...] for k in ("scalars", "field", "card_idx")}
    before = float(net.predict(batch)[0])

    vocab_new = RN.extend_to_vocab(net, db)
    assert net.vocab_ids[:len(old_ids)] == old_ids, "既存 idx がズレた（append-only 破れ）"
    added = net.vocab_ids[len(old_ids):]
    assert added == sorted(added) and "ST31-001" in added, "新カードが末尾ソート追記されていない"
    assert net.Emb.shape[0] == len(net.vocab_ids) + 1
    if net.EffF is not None:
        assert net.EffF.shape[0] == len(net.vocab_ids) + 1
    assert vocab_new["PRB01-001"] == 2282 and vocab_new["ST31-001"] > len(old_ids)
    after = float(net.predict(batch)[0])
    assert abs(before - after) < 1e-12, "拡張で既存盤面の評価が変わった（恒等破れ）"


def test_extend_without_ids_and_mismatched_rows_errors(db):
    """vocab_ids 無し＋行数不一致（訓練時DB世代が不明）は黙って進まず明示エラー。"""
    tiny = RN.ValueNet(vocab_size=10, d_emb=4, hidden=8, feat_dim=E.feature_dim(1))
    tiny.vocab_ids = None
    with pytest.raises(ValueError):
        RN.extend_to_vocab(tiny, db)


def test_save_load_roundtrip_preserves_ids(tmp_path):
    """save→load で vocab_ids が保存される（npz 追加キー・旧 npz は None のまま）。"""
    net = RN.ValueNet(vocab_size=3, d_emb=4, hidden=8, feat_dim=E.feature_dim(1))
    net.vocab_ids = ["OP01-001", "OP01-002", "OP01-003"]
    p = str(tmp_path / "v.npz")
    net.save(p)
    r = RN.ValueNet.load(p)
    assert r.vocab_ids == net.vocab_ids
