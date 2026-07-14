"""盤面エンコーダ(D-1)の検証：状態が符号化に正しく反映され・決定的・相手手札を漏らさない。"""
import conftest  # noqa: F401
import pytest

from opcg_sim.src.models.enums import CardType
from cpu_selfplay import build_deck, _load_db
from opcg_sim.src.core.gamestate import GameManager, Player
from engine_helpers import make_master, make_instance
import rl_encoder as E


@pytest.fixture(scope="module")
def db():
    return _load_db()


@pytest.fixture(scope="module")
def vocab(db):
    return E.build_vocab(db)


def _gm(db):
    import random
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    return gm


def test_vocab_covers_all_cards(db, vocab):
    n = sum(1 for cid in db.raw_db.keys() if db.get_card(cid) is not None)
    assert len(vocab) == n and min(vocab.values()) == 1   # 0 は PAD 予約


def test_shapes(db, vocab):
    enc = E.encode(_gm(db), "p1", vocab)
    assert enc["scalars"].shape == (14,)
    assert enc["field"].shape == (2 * E.MAX_FIELD, E.PER_CHAR)
    assert enc["card_idx"].shape == (2 + 2 * E.MAX_FIELD + E.MAX_HAND,)


def test_deterministic(db, vocab):
    gm = _gm(db)
    a = E.encode(gm, "p1", vocab)
    b = E.encode(gm, "p1", vocab)
    assert (a["scalars"] == b["scalars"]).all()
    assert (a["field"] == b["field"]).all()
    assert (a["card_idx"] == b["card_idx"]).all()


def test_reflects_field_char(db, vocab):
    """場のバニラキャラの power/cost/keyword/card_id が符号化に出る。"""
    gm = _gm(db)
    m = make_master(card_id="ZZ-001", name="壁", type=CardType.CHARACTER,
                    cost=3, power=6000, counter=1000, abilities=(), effect_text="")
    import builtins  # ブロッカー keyword を付与
    object.__setattr__(m, "keywords", {"ブロッカー"})
    c = make_instance(m, owner="p1"); c.is_rest = False; c.is_newly_played = False
    gm.p1.field[:] = [c]
    enc = E.encode(gm, "p1", vocab)
    # 自場スロット0: [cost/10, power/1e4, rest, don, blocker, 速攻, W2, バニ]
    row = enc["field"][0]
    assert abs(row[0] - 0.3) < 1e-6        # cost 3
    assert abs(row[1] - 0.6) < 1e-6        # power 6000
    assert row[2] == 0.0                   # not rest
    assert row[4] == 1.0                   # ブロッカー flag
    # card_idx 自場先頭(=index 2)はこのカードの vocab idx（DB未登録なら PAD=0）。
    assert enc["card_idx"][2] == vocab.get("ZZ-001", E.PAD)
    assert enc["scalars"][8] == 1.0        # 自場キャラ数=1


def test_opponent_hand_not_leaked(db, vocab):
    """相手手札の中身（card_id）は符号化に一切載らない（枚数のみ）＝公平性。"""
    gm = _gm(db)
    # 相手手札を既知カードで満たす → card_idx にそれらが現れないことを確認。
    opp_ids = [c.master.card_id for c in gm.p2.hand]
    enc = E.encode(gm, "p1", vocab)
    opp_vidx = {vocab.get(cid) for cid in opp_ids if cid in vocab}
    present = set(int(x) for x in enc["card_idx"])
    # 自分の手札/場/リーダーは載るが、相手手札 idx は（自分側と偶然一致する場合を除き）載らない。
    # 厳密化: 相手手札スロット自体が card_idx に存在しない（自手札枠は自分のみ）。
    me_hand_ids = {vocab.get(c.master.card_id) for c in gm.p1.hand if c.master.card_id in vocab}
    leaked = (opp_vidx - me_hand_ids - {vocab.get(gm.p2.leader.master.card_id)})
    # 相手手札に固有(自分側に無い)な idx が card_idx に漏れていないこと
    assert not (leaked & present), "相手手札の card_id が符号化に漏れている"
    # 枚数はスカラに出る
    assert enc["scalars"][7] == len(gm.p2.hand)
