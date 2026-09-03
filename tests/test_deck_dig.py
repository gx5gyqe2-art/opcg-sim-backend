"""掘りカード差し込み合成デッキ（`tests/harness/deck_dig.py`・2026-09-02）の契約（基盤健全性）。

なぜ基盤健全性か: 残ドン掘りの方針対照実験のデッキ生成器であり、ゲームプレイの正しさには
触れない（`cpu_infra`）。

守る性質:
  1. 紫を含むリーダーには掘りカードが n_inject 枚差し込まれ、50枚・同名4枚以下を維持する。
  2. 紫を含まないリーダーには差し込まない（色不一致＝構築不能）＝デッキは不変。
  3. 決定論: 同じ seed なら同じデッキ。
  4. builder は run_game 契約（leader, cards, leader, cards）を返す。
"""
import collections

import conftest  # noqa: F401
import pytest

import _bootstrap  # noqa: F401

import deck_dig as DD
import deck_synth as DS
from cpu_selfplay import _load_db
from opcg_sim.src.core.cpu_learned import _is_dig_card

pytestmark = pytest.mark.cpu_infra


@pytest.fixture(scope="module")
def db():
    return _load_db()


def _ids(cards):
    return [DS._master(c).card_id for c in cards]


def _n_dig(cards):
    return sum(1 for c in cards if _is_dig_card(DS._master(c)))


def test_inject_into_purple_leader_without_theme(db):
    """クロコダイル（紫青・テーマ外＝合成では掘り0枚）に 8 枚差し込まれ、50枚・同名4枚以下。"""
    lid = "OP01-062"
    leader, base = DS.synth_deck(db, lid, seed=3, owner="p1")
    assert _n_dig(base) == 0
    cards, n = DD.inject_dig(db, leader, base, "p1", seed=3)
    assert n == DD.DEFAULT_INJECT and _n_dig(cards) == DD.DEFAULT_INJECT
    assert len(cards) == DS.DECK_SIZE
    assert max(collections.Counter(DS._master(c).name for c in cards).values()) <= DS.MAX_COPIES
    assert all(getattr(c, "owner", "p1") == "p1" for c in cards)


def test_enel_keeps_cap_and_size(db):
    """エネル（テーマで既に12枚）でも同名4枚の上限と50枚を壊さない。"""
    leader, base = DS.synth_deck(db, "OP15-058", seed=1, owner="p2")
    cards, _ = DD.inject_dig(db, leader, base, "p2", seed=1)
    assert len(cards) == DS.DECK_SIZE
    assert max(collections.Counter(DS._master(c).name for c in cards).values()) <= DS.MAX_COPIES
    assert _n_dig(cards) >= _n_dig(base)


def test_non_purple_leader_is_untouched(db):
    """シャンクス（赤）には差し込めない＝枚数0・デッキ不変。"""
    leader, base = DS.synth_deck(db, "OP09-001", seed=2, owner="p1")
    cards, n = DD.inject_dig(db, leader, base, "p1", seed=2)
    assert n == 0 and _n_dig(cards) == 0
    assert sorted(_ids(cards)) == sorted(_ids(base))


def test_builder_is_deterministic_and_run_game_shaped(db):
    b = DD.synth_dig_deck_builder("OP01-062", "OP15-058", seed=5)
    l1, c1, l2, c2 = b(db, 0)
    l1b, c1b, l2b, c2b = b(db, 0)
    assert _ids(c1) == _ids(c1b) and _ids(c2) == _ids(c2b)
    assert DS._master(l1).card_id == "OP01-062" and DS._master(l2).card_id == "OP15-058"
    assert len(c1) == len(c2) == DS.DECK_SIZE
    assert _n_dig(c1) >= DD.DEFAULT_INJECT and _n_dig(c2) >= DD.DEFAULT_INJECT
