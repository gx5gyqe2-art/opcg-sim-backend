"""v11 符号化世代（**リーダー物理要約24**・append-only・2026-08-14）の検証。

v11 は v10（scalars 73）末尾に自/相手リーダーの物理要約ベクトル
（`opcg_sim/src/learned/leader_feat.py`・能力木→毎ターン率12次元）×2 を足す（97）。

なぜ要るか（消去はしご・机上解剖 2026-08-13〜14）:
  - 接戦帯の帰趨を支配するリーダー再帰効果（ドンランプ・回復・ミル・常在修正）が
    現行特徴に **0ビット**（リーダーは power/life のみ）。bb 系は実L接戦帯 r≈0.1、
    リーダーを消すと 0.50（2.6σ）。ラベル品質（bb4）・先生の質（bb5）は棄却済み。
  - ID 非依存（パース木から導出）＝新リーダーへ即汎化（本プロジェクトの根本制約）。

固定する性質:
  - 版マップ: scalars_dim(11) = 73+24 = 97（append-only）
  - 配線: v11 末尾 24 値が `leader_pair_vectors` の直接呼び出しと一致（offset ズレ検出）
  - 接頭辞不変: encode(v10) == encode(v11)[:73]
  - 恒等温スタート（v10→v11 拡張で出力不変）
  - 符号化は観測: encode(v11) が global random / np.random を消費しない
  - 意味の錨: ハンニャバル EB01-021 は don_rate>0・ビビ OP04-001 は atk_disable=1
"""
import os
import random

import conftest  # noqa: F401
import numpy as np
import pytest

import replay_runner as RR
import replay_reeval as RE
import rl_encoder as E
from cpu_selfplay import _load_db
from opcg_sim.src.core.cpu_learned import warm_start_value
from opcg_sim.src.learned.leader_feat import (DIMS, LEADER_FEAT_DIM,
                                              leader_pair_vectors,
                                              leader_static_vector)
from opcg_sim.src.learned.value_net import ValueNet

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（符号化拡張の機構）

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "replays", "gen7_marks_20260728")
OFF = E.SCALARS_V10  # v11 追加ブロックの先頭 offset（73）


@pytest.fixture(scope="module")
def db():
    return _load_db()


@pytest.fixture(scope="module")
def vocab(db):
    return E.build_vocab(db)


def _mark_state(db, fn, idx):
    raw = RE.load_replay_json(os.path.join(FIX, fn))
    rec = raw.get("replay", raw)
    m, who = RR.state_at_action(db, rec, idx)
    return m, (who if isinstance(who, str) else who.name)


def test_version_map_appends_24():
    assert LEADER_FEAT_DIM == 12
    assert E.scalars_dim(11) == E.scalars_dim(10) + 2 * LEADER_FEAT_DIM == 97
    assert 11 in E.known_versions() and E.known_versions() == sorted(E.known_versions())


def test_v11_tail_matches_direct_call_and_prefix(db, vocab):
    m, who = _mark_state(db, "opcg_replay_6563214359889287880.json.gz", 8)
    e10 = E.encode(m, who, vocab, version=10)
    e11 = E.encode(m, who, vocab, version=11)
    assert e11["scalars"].shape[0] == 97
    np.testing.assert_allclose(e11["scalars"][:OFF], e10["scalars"], rtol=0, atol=1e-6)
    lv_me, lv_opp = leader_pair_vectors(m, who)
    np.testing.assert_allclose(e11["scalars"][OFF:OFF + 12], lv_me, rtol=0, atol=1e-6)
    np.testing.assert_allclose(e11["scalars"][OFF + 12:], lv_opp, rtol=0, atol=1e-6)


def test_encode_v11_consumes_no_global_rng(db, vocab):
    m, who = _mark_state(db, "opcg_replay_6563214359889287880.json.gz", 8)
    random.seed(123)
    np.random.seed(123)
    r0 = (random.random(), float(np.random.random()))
    random.seed(123)
    np.random.seed(123)
    E.encode(m, who, vocab, version=11)
    r1 = (random.random(), float(np.random.random()))
    assert r0 == r1, "encode(v11) が global random を消費した（符号化は観測）"


def test_semantic_anchors_known_leaders(db):
    db.parse_all()
    han = leader_static_vector(db.get_card("EB01-021"))
    assert han[DIMS.index("don_rate")] > 0, "ハンニャバル: ドンランプが don_rate に映らない"
    vivi = db.get_card("OP04-001")
    if vivi is not None:
        v = leader_static_vector(vivi)
        assert v[DIMS.index("atk_disable")] == 1.0, "ビビ: アタック不可が映らない"
    nami = db.get_card("OP03-040")
    if nami is not None:
        v = leader_static_vector(nami)
        assert v[DIMS.index("rule_flag")] == 1.0, "ナミ: デッキ0勝利ルールが映らない"


def test_warm_start_identity_v10_to_v11(db, vocab):
    m, who = _mark_state(db, "opcg_replay_6563214359889287880.json.gz", 8)
    old = ValueNet(vocab_size=len(vocab), d_emb=8, hidden=16,
                   feat_dim=E.feature_dim(10), seed=3)
    new = warm_start_value(old, 10, 11)
    assert new.feat_dim == E.feature_dim(11)
    e10 = E.encode(m, who, vocab, version=10)
    e11 = E.encode(m, who, vocab, version=11)
    b10 = {k: e10[k][None, ...] for k in ("scalars", "field", "card_idx")}
    b11 = {k: e11[k][None, ...] for k in ("scalars", "field", "card_idx")}
    assert float(old.predict(b10)[0]) == pytest.approx(float(new.predict(b11)[0]), abs=1e-9)
