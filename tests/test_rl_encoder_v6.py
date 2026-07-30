"""v6 符号化世代（**自手札の資源集約**・append-only・2026-07-30）の検証。

v6 は v5（scalars 55）末尾に自分の手札の資源集約 5 値を足す（60）:
  [カウンター総量, カウンター札枚数, 最大カウンター値, 手札ブロッカー数, イベント数]。

なぜ要るか（ユーザ指摘「手札の価値をどう正確に判断するか」・v23 遮蔽帰属）: v5 までスカラーに
載る手札情報は**枚数だけ**で、質は card_idx の埋め込み経由でしか見えなかった。その結果 value は
「手札が減った＝勝者の相貌」という逆向きの相関を学んでいた（誤着を押し上げる寄与: 手札枚数
+0.084・手札ID +0.165）。山札残（v4）と同じ集計を手札にも与え「手札は防御資源」という
線形の取っ手を作る。相手手札は対象外（公平性契約）。カード個別知識は持たない汎用量。

**恒等温スタート**（v5→v6 拡張で出力不変＝新5行がゼロ）を必達で確認する。
"""
import numpy as np
import pytest

import conftest  # noqa: F401
from opcg_sim.src.models.enums import CardType
from cpu_selfplay import _load_db
from opcg_sim.src.core.gamestate import GameManager, Player
from engine_helpers import make_master, make_instance
import rl_encoder as E
from opcg_sim.src.learned.value_net import ValueNet
from opcg_sim.src.core.cpu_learned import warm_start_value

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（符号化拡張の機構）

HAND_OFF = E.SCALARS_V5   # v6 追加ブロックの先頭 offset（55）


@pytest.fixture(scope="module")
def db():
    return _load_db()


@pytest.fixture(scope="module")
def vocab(db):
    return E.build_vocab(db)


def _game_with_hand(hand_masters):
    """指定マスターだけを手札に持つ最小 GameManager（他ゾーンは空）。"""
    l1 = make_instance(make_master(card_id="L-1", power=5000), owner="p1")
    l2 = make_instance(make_master(card_id="L-2", power=5000), owner="p2")
    p1 = Player("p1", [], l1)
    p2 = Player("p2", [], l2)
    m = GameManager(p1, p2)
    p1.hand = [make_instance(mm, owner="p1") for mm in hand_masters]
    m.turn_player = p1
    return m


def test_version_map_appends_five():
    assert E.scalars_dim(6) == E.scalars_dim(5) + 5 == 60
    assert 6 in E.known_versions() and E.known_versions() == sorted(E.known_versions())


def test_v6_shape_and_prefix_identity(db, vocab):
    """append-only: v6 の先頭 55 は v5 と完全一致（既存の重みの意味を壊さない）。"""
    from opcg_game import OPCGGame
    m = OPCGGame().new_game(db, 4242)
    e5 = E.encode(m, m.p1.name, vocab, version=5)
    e6 = E.encode(m, m.p1.name, vocab, version=6)
    assert len(e5["scalars"]) == 55 and len(e6["scalars"]) == 60
    assert np.allclose(e5["scalars"], e6["scalars"][:55])


def test_hand_aggregate_math(vocab):
    """集計の中身（正規化込み）を固定する。"""
    blk = make_master(card_id="BLK", counter=1000)
    object.__setattr__(blk, "keywords", {"ブロッカー"})   # CardMaster は frozen（v4 テストと同流儀）
    hand = [make_master(card_id="C-2000", counter=2000),
            make_master(card_id="C-1000", counter=1000),
            make_master(card_id="C-0", counter=0),
            blk,
            make_master(card_id="EV", counter=0, type=CardType.EVENT)]
    m = _game_with_hand(hand)
    s = E.encode(m, "p1", vocab, version=6)["scalars"]
    assert s[HAND_OFF + 0] == pytest.approx(4000 / 20000.0)     # カウンター総量
    assert s[HAND_OFF + 1] == pytest.approx(3 / 10.0)           # カウンター札枚数
    assert s[HAND_OFF + 2] == pytest.approx(2000 / 2000.0)      # 最大カウンター
    assert s[HAND_OFF + 3] == pytest.approx(1 / 10.0)           # 手札ブロッカー
    assert s[HAND_OFF + 4] == pytest.approx(1 / 10.0)           # イベント


def test_own_hand_only_not_opponent(vocab):
    """公平性契約: 相手手札の中身は v6 特徴に入らない（枚数のみは v1 から既出）。"""
    m = _game_with_hand([make_master(card_id="C-0", counter=0)])
    m.p2.hand = [make_instance(make_master(card_id="OPP-C", counter=2000), owner="p2")
                 for _ in range(5)]
    s = E.encode(m, "p1", vocab, version=6)["scalars"]
    assert s[HAND_OFF + 0] == pytest.approx(0.0)     # 相手の 10000 カウンターは漏れない
    assert s[HAND_OFF + 2] == pytest.approx(0.0)


def test_empty_hand_safe(vocab):
    """空手札・属性欠落で例外を投げない（探索クローン上で呼ばれる契約）。"""
    m = _game_with_hand([])
    s = E.encode(m, "p1", vocab, version=6)["scalars"]
    assert np.all(np.isfinite(s)) and np.allclose(s[HAND_OFF:], 0.0)


def test_hand_aggregate_reflects_counter_loss(vocab):
    """カウンターを1枚切った（手札から失った）状態は資源集約が下がる＝『減った=良い』の
    逆向きの取っ手になっていること（v23 の相貌学習への対抗が特徴として成立するかの確認）。"""
    rich = _game_with_hand([make_master(card_id="C-2000", counter=2000),
                            make_master(card_id="C-2000b", counter=2000)])
    poor = _game_with_hand([make_master(card_id="C-2000", counter=2000)])
    s_rich = E.encode(rich, "p1", vocab, version=6)["scalars"]
    s_poor = E.encode(poor, "p1", vocab, version=6)["scalars"]
    assert s_rich[HAND_OFF + 0] > s_poor[HAND_OFF + 0]
    assert s_rich[HAND_OFF + 1] > s_poor[HAND_OFF + 1]
    assert s_rich[HAND_OFF + 2] == s_poor[HAND_OFF + 2]        # 最大値は不変（枚数と独立）


def test_warm_start_v5_to_v6_is_identity(db, vocab):
    """v5 ネットを v6 へ拡張しても、同一盤面の予測が**完全に一致**する（新5行ゼロ）。"""
    from opcg_game import OPCGGame
    m = OPCGGame().new_game(db, 777)
    net5 = ValueNet(vocab_size=len(vocab), d_emb=8, hidden=16,
                    feat_dim=E.feature_dim(5), seed=3)
    net6 = warm_start_value(net5, 5, 6)
    assert net6.feat_dim == E.feature_dim(6)
    for name in (m.p1.name, m.p2.name):
        e5 = E.encode(m, name, vocab, version=5)
        e6 = E.encode(m, name, vocab, version=6)
        b5 = {k: e5[k][None, ...] for k in ("scalars", "field", "card_idx")}
        b6 = {k: e6[k][None, ...] for k in ("scalars", "field", "card_idx")}
        assert float(net5.predict(b5)[0]) == pytest.approx(float(net6.predict(b6)[0]), abs=1e-9)
