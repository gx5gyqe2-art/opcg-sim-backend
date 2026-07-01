"""データ生成(D-2)の検証：ラベル付き局面を産出・shape整合・value二値・概ね均衡。"""
import conftest  # noqa: F401
import pytest

from cpu_selfplay import _load_db
import rl_encoder as E
import rl_datagen as G


@pytest.fixture(scope="module")
def db():
    return _load_db()


def test_generate_smoke(db):
    vocab = E.build_vocab(db)
    data = G.generate(db, vocab, n_games=2, eps=0.3, max_steps=200, seed0=0, sample_every=3)
    assert data is not None, "採取0（決着しなかった）"
    n = len(data["value"])
    assert n > 0
    assert data["scalars"].shape == (n, 14)
    assert data["field"].shape == (n, 2 * E.MAX_FIELD, E.PER_CHAR)
    assert data["card_idx"].shape == (n, 2 + 2 * E.MAX_FIELD + E.MAX_HAND)
    # value は ±1 の二値、かつ一方に完全には偏らない（自己対戦＝両者勝ちが混ざる）。
    vals = set(float(v) for v in data["value"])
    assert vals <= {1.0, -1.0}
    assert 0.0 < (data["value"] > 0).mean() < 1.0
