"""公式サイトの構築例（実デッキ帯の拡張・`tests/fixtures/decks/official_decks_20261022.json`）の健全性。

実デッキ帯は評価・アリーナの検証帯（ホールドアウト）。ファイルが壊れていると検証帯の数字が黙って変わるので、
全デッキが 50 枚・カード DB に在る・構築ルールに合法であること、`official`／`real` の引き方が両席とも合法な
デッキを返すことだけを見る。
"""
import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.loop import deck_synth as DS
from opcg_sim.loop import decks as D


def test_official_decks_are_legal_and_real_pairs_draw_from_them():
    db = D.load_db()
    off = D.official_decks()
    assert len(off) == 111
    for lid, cards in off:
        assert len(cards) == 50
        assert DS.check_legal(db.get_card(lid), [db.get_card(c) for c in cards]) == []
    real = {(l, tuple(c)) for l, c in D.user_decks() + off}
    for seed in range(30):
        for which in ("official", "real"):
            for lid, cards in D.build_pair(db, None, None, seed, which):
                assert (lid, tuple(cards)) in real
