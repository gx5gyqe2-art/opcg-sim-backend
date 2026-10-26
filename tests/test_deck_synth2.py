"""合成デッキ v2（`opcg_sim/loop/deck_synth2.py`・`--decks synth_roles2`・WP `synth-decks-v2`）の契約。

標準（マーカー無し・常時実行）: 教材のデッキを組む器なので、不正なデッキ（50 枚でない・色違い・
同じカード 5 枚）を出すと生成する棋譜そのものが壊れる。守るのは**合法なデッキを出す**ことだけ
（分布の近さは `tests/scripts/synth_deck_compare.py` で測る・`docs/reports/2026-10-22_synth_decks_v2.md`）。
"""
import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.loop import deck_synth as DS
from opcg_sim.loop import decks as D


def test_synth_roles2_builds_legal_decks():
    db = D.load_db()
    for seed in range(40):
        la, lb = D.leader_pair(db, seed, "random")
        for lid, cards in D.build_pair(db, la, lb, seed, "synth_roles2"):
            why = DS.check_legal(db.get_card(lid), [db.get_card(c) for c in cards])
            assert not why, (seed, lid, why)
