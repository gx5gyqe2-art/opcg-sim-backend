"""v4 符号化世代（自デッキ残の集約特徴・append-only・cpu_v5_plan.md §4-3）の検証。

v4 は v3（scalars 46）末尾に自分の残ライブラリ（me.deck）の守り/資源集約 5 値を足す（51）:
  [残カウンター総量, カウンター札密度, ブロッカー残, イベント残, 高コストキャラ残]。
「自分の山札にどれだけ守り札/カウンターが残るか」を可視化＝薄いライフの価値(C5)と残ターン読み(D3)を
底上げする。カード個別に依存しない汎用量（counter/keyword/type の集計）で、相手デッキは非公開ゆえ
自分のみ（公平性契約）。**恒等温スタート**（v3→v4 拡張で出力不変＝新5行がゼロ）を必達で確認する。
"""
import numpy as np
import pytest

import conftest  # noqa: F401
from opcg_sim.src.models.enums import CardType
from cpu_selfplay import build_deck, _load_db
from opcg_sim.src.core.gamestate import GameManager, Player
from engine_helpers import make_master, make_instance
import rl_encoder as E
from opcg_sim.src.learned.value_net import ValueNet
from opcg_sim.src.core.cpu_learned import warm_start_value

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（符号化拡張の機構）


@pytest.fixture(scope="module")
def db():
    return _load_db()


@pytest.fixture(scope="module")
def vocab(db):
    return E.build_vocab(db)


def _gm(db, seed=0):
    import random
    random.seed(seed)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    return gm


def test_version_map_appends_five():
    """v4 = v3 + 5（append-only・単調増加）・feature_dim も 5 増える。"""
    assert E.scalars_dim(4) == E.scalars_dim(3) + 5
    assert 4 in E.known_versions()
    assert E.feature_dim(4) == E.feature_dim(3) + 5


def test_v4_shape_and_prefix_identity(db, vocab):
    """v4 scalars は 51・先頭 46 は v3 と完全一致（既存並びを壊さない＝append-only 不変条件）。"""
    gm = _gm(db)
    e3 = E.encode(gm, "p1", vocab, version=3)
    e4 = E.encode(gm, "p1", vocab, version=4)
    assert e4["scalars"].shape == (51,)
    assert (e4["scalars"][:46] == e3["scalars"]).all()
    # card_idx / field は v3 と不変（v4 は scalars 末尾のみ拡張）。
    assert (e4["card_idx"] == e3["card_idx"]).all()
    assert (e4["field"] == e3["field"]).all()


def test_deck_aggregate_math(db, vocab):
    """me.deck を既知構成に差し替え、末尾5特徴が集約定義どおりの値になる。"""
    gm = _gm(db)
    blk = make_master(card_id="ZZ-BLK", name="壁", type=CardType.CHARACTER,
                      cost=2, power=3000, counter=2000, abilities=(), effect_text="")
    object.__setattr__(blk, "keywords", {"ブロッカー"})
    ev = make_master(card_id="ZZ-EV", name="トリック", type=CardType.EVENT,
                     cost=1, power=0, counter=0, abilities=(), effect_text="")
    big = make_master(card_id="ZZ-BIG", name="大物", type=CardType.CHARACTER,
                      cost=8, power=8000, counter=0, abilities=(), effect_text="")
    plain = make_master(card_id="ZZ-PL", name="平", type=CardType.CHARACTER,
                        cost=3, power=4000, counter=1000, abilities=(), effect_text="")
    deck = [make_instance(m, owner="p1") for m in (blk, ev, big, plain)]
    gm.p1.deck[:] = deck
    e = E.encode(gm, "p1", vocab, version=4)
    tail = e["scalars"][46:]
    # counter_total=(2000+0+0+1000)/(50*2000)=0.03 / density=2/4=0.5 / blocker=1/50 / event=1/50 / bigchar=1/50
    assert tail[0] == pytest.approx(3000 / (50 * 2000))
    assert tail[1] == pytest.approx(0.5)
    assert tail[2] == pytest.approx(1 / 50)
    assert tail[3] == pytest.approx(1 / 50)
    assert tail[4] == pytest.approx(1 / 50)   # 大物(cost8>=7)のみ・壁(cost2)平(cost3)は非該当


def test_own_deck_only_not_opponent(db, vocab):
    """相手デッキを変えても自分の末尾5特徴は不変（公平性契約＝自デッキのみ集約）。"""
    gm = _gm(db)
    base = E.encode(gm, "p1", vocab, version=4)["scalars"][46:].copy()
    big = make_master(card_id="ZZ-OPP", name="相手大物", type=CardType.CHARACTER,
                      cost=9, power=9000, counter=2000, abilities=(), effect_text="")
    gm.p2.deck[:] = [make_instance(big, owner="p2") for _ in range(30)]
    after = E.encode(gm, "p1", vocab, version=4)["scalars"][46:]
    assert (base == after).all(), "相手デッキ改変が自分の集約特徴に漏れている（公平性違反）"


def test_empty_deck_safe(db, vocab):
    """残デッキ 0 でも例外を投げず 0 ベクトルを返す（探索クローン上での安全性）。"""
    gm = _gm(db)
    gm.p1.deck[:] = []
    tail = E.encode(gm, "p1", vocab, version=4)["scalars"][46:]
    assert tail.shape == (5,)
    assert np.all(tail == 0.0)


def _batch(enc):
    return {k: enc[k][None, ...] for k in ("scalars", "field", "card_idx")}


def test_warm_start_v3_to_v4_is_identity(db, vocab):
    """v3 ネットを v4 へ温スタート拡張すると、v4 符号化での value/aux 出力が v3 と完全一致する
    （新5スカラーが W1 のゼロ行に当たる＝恒等）。これが「データ再生成しても学習前は挙動不変」の根拠。"""
    gm = _gm(db)
    net3 = ValueNet(vocab_size=len(vocab), d_emb=8, hidden=16,
                    feat_dim=E.feature_dim(3), lead_slots=2, seed=3)
    # aux ヘッドを非ゼロにして aux 側の恒等も確かめる。
    net3.W2t = np.random.default_rng(1).standard_normal((16, 1)) * 0.3
    net4 = warm_start_value(net3, from_version=3, to_version=4)
    assert net4.feat_dim == E.feature_dim(4)
    for me in ("p1", "p2"):
        e3 = E.encode(gm, me, vocab, version=3)
        e4 = E.encode(gm, me, vocab, version=4)
        p3, a3 = net3.predict_with_aux(_batch(e3))
        p4, a4 = net4.predict_with_aux(_batch(e4))
        assert p4 == pytest.approx(p3, abs=1e-9), f"value が恒等でない（{me}）"
        assert a4 == pytest.approx(a3, abs=1e-9), f"aux が恒等でない（{me}）"
