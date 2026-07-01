"""パラメトリック デッキ生成器（deck_generator）の検証。

v4b 生成器の不変条件を CI で保証: 常に合法（50枚・同名≤4・色一致）・毎回新規（重複なし）・
リーダー多様・held-out 実リストと不一致・極端ノイズも合法・決定的（同seed同deck）・実対局可能。
"""
import random

import conftest  # noqa: F401
import heldout_decks as HD
from cpu_selfplay import _load_db
from deck_generator import DeckGenerator, build_instances
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_ai

_CACHE = {}


def _gen():
    if not _CACHE:
        db = _load_db()
        _CACHE.update(db=db, gen=DeckGenerator(db, seed=0))
    return _CACHE["db"], _CACHE["gen"]


def _assert_legal(db, lid, deck):
    assert sum(deck.values()) == 50
    assert all(1 <= n <= 4 for n in deck.values())
    lcol = {getattr(x, "value", x) for x in db.get_card(lid).colors}
    for cid in deck:
        ccol = {getattr(x, "value", x) for x in db.get_card(cid).colors}
        assert lcol & ccol, f"{cid} 色不一致"


def test_generated_decks_always_legal_unique_diverse():
    db, gen = _gen()
    rng = random.Random(1)
    held = {tuple(sorted(v.items())) for v in HD.all_lists().values()}
    seen, leaders = set(), set()
    for _ in range(100):
        lid, deck = gen.generate(rng)
        _assert_legal(db, lid, deck)
        key = tuple(sorted(deck.items()))
        assert key not in held, "held-out 実リストと一致（リーク）"
        assert key not in seen, "同一リストが再生成された（暗記可能化）"
        seen.add(key); leaders.add(lid)
    assert len(leaders) >= 30, f"リーダー多様性不足: {len(leaders)}"


def test_noise_mode_is_legal():
    db, gen = _gen()
    rng = random.Random(2)
    for _ in range(10):
        lid, deck = gen.generate(rng, noise_prob=1.0)
        _assert_legal(db, lid, deck)


def test_deterministic_with_same_seed():
    db, gen = _gen()
    a = gen.generate(random.Random(7))
    b = gen.generate(random.Random(7))
    assert a == b, "同seedで同デッキにならない（resume/再現性が壊れる）"


def test_generated_decks_playable():
    db, gen = _gen()
    random.seed(3)   # エンジンがグローバル random を使うため軌跡を固定（順序依存フレーク防止）
    rng = random.Random(3)
    lid1, d1 = gen.generate(rng)
    lid2, d2 = gen.generate(rng)
    l1, c1 = build_instances(db, lid1, d1, "p1")
    l2, c2 = build_instances(db, lid2, d2, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    ply = 0
    while ply < 400 and m.winner is None:
        pa = m.pending_actor_action()
        if pa is None:
            break
        nm = pa[0]
        actor = m.p1 if m.p1.name == nm else m.p2
        legal = m.get_legal_actions(actor)
        if not legal:
            break
        try:
            cpu_ai._apply_move_inplace(m, nm, legal[rng.randrange(len(legal))])
        except Exception:
            break   # ランダム乱打の例外手は本テストの対象外（EXCEPTION=0 は監査側の責務）
        ply += 1
    assert ply > 20, "生成デッキで対局が進行しない"
