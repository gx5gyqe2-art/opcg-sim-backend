"""PIMC 決定化の透視禁止（v4b Blocker）の検証: `cpu_ai._determinize_hidden`。

不変条件を CI で保証: 元 manager 不変・両者の枚数保存・カード多重集合保存（no loss）・
自分の手札/場は不変（既知/公開）・山札順と裏向きライフは再サンプル（透視禁止）・
表向きライフは保存（公開情報）。L1 の `_determinize_opponent` は変更しない（別関数）。
"""
import random
from collections import Counter

import conftest  # noqa: F401
from cpu_selfplay import build_deck, _load_db
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_ai


def _advance(m, n, seed=1):
    rng = random.Random(seed)
    for _ in range(n):
        pa = m.pending_actor_action()
        if pa is None:
            break
        nm = pa[0]
        a = m.p1 if m.p1.name == nm else m.p2
        lg = m.get_legal_actions(a)
        if not lg:
            break
        try:
            cpu_ai._apply_move_inplace(m, nm, lg[rng.randrange(len(lg))])
        except Exception:
            break


def _multiset(pl):
    return Counter(c.master.card_id for c in list(pl.hand) + list(pl.deck) + list(pl.life))


def _game():
    random.seed(0)
    db = _load_db()
    l1, c1 = build_deck(db, "p1"); l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2)); m.start_game()
    return m


def test_determinize_hidden_invariants():
    m = _game()
    _advance(m, 30)
    me_ms, opp_ms = _multiset(m.p1), _multiset(m.p2)
    my_hand = [c.master.card_id for c in m.p1.hand]
    my_field = [c.master.card_id for c in m.p1.field]
    deck_before = [c.master.card_id for c in m.p1.deck]
    opp_hand_before = [c.master.card_id for c in m.p2.hand]

    clone = cpu_ai._determinize_hidden(m, "p1", random.Random(42))

    # 元 manager は不変
    assert [c.master.card_id for c in m.p1.deck] == deck_before
    # 枚数保存
    assert (len(clone.p1.hand), len(clone.p1.deck), len(clone.p1.life)) == \
           (len(m.p1.hand), len(m.p1.deck), len(m.p1.life))
    # カード多重集合保存（no loss）
    assert _multiset(clone.p1) == me_ms and _multiset(clone.p2) == opp_ms
    # 自分の手札/場は不変（既知/公開）
    assert [c.master.card_id for c in clone.p1.hand] == my_hand
    assert [c.master.card_id for c in clone.p1.field] == my_field
    # 山札順は再サンプル（透視禁止）— 十分な山札で順序が変わる
    assert [c.master.card_id for c in clone.p1.deck] != deck_before
    # 相手手札は再サンプル
    assert [c.master.card_id for c in clone.p2.hand] != opp_hand_before


def test_faceup_life_preserved():
    m = _game()
    m.p2.life[0].is_face_up = True
    face_id = m.p2.life[0].master.card_id
    clone = cpu_ai._determinize_hidden(m, "p1", random.Random(9))
    assert clone.p2.life[0].master.card_id == face_id and clone.p2.life[0].is_face_up
    assert all(not c.is_face_up for c in clone.p2.life[1:])


def test_l1_determinize_opponent_unchanged():
    """L1 用の determinize は自ライフ/自山札を触らない（ゲート基準の安定）。"""
    m = _game()
    _advance(m, 20)
    my_deck = [c.master.card_id for c in m.p1.deck]
    clone = cpu_ai._determinize_opponent(m, "p1", random.Random(3))
    assert [c.master.card_id for c in clone.p1.deck] == my_deck, "L1 determinize が自山札を変えた"
