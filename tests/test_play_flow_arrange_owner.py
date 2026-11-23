"""2 つの基準で選んでデッキの下に置く効果を、実対局の流れで固定する（2026-11-23）。

- OP06-056 天叢雲剣「相手の、コスト2以下のキャラ1枚までとコスト1以下のキャラ1枚までを、持ち主のデッキの下に
  好きな順番で置く」: 2 つ目の基準で、1 つ目に選んだカードを選び直せない（従来は同じ 1 枚を二重に選べ、
  実際には 1 枚しか置かれなかった）。
- OP18-051 Mr.4＆ミス・メリークリスマス「コスト6以下と5以下のキャラ1枚ずつまでを、持ち主が好きな順番で
  デッキの下に置く」（両者のキャラが対象）: 並び替えは置かれるカードの持ち主が行う。相手のキャラ 2 枚なら相手、
  両者に 1 枚ずつなら並び替えは無い（別々のデッキの下へ 1 枚ずつ）。従来は効果の使い手が並べていた。
ゲームプレイの退行を見逃さないための必須/標準テスト。
"""
import os

import pytest

import _bootstrap  # noqa: F401
from harness.rs_scenario import Scenario

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "data", "opcg_effects.json")),
    reason="効果 JSON（生成物）が無い")

VANILLA1 = "EB01-005"     # コスト1・キャラ
BELLAMY2 = "OP01-076"     # コスト2・キャラ（効果なし）
AMENOMURAKUMO = "OP06-056"
MR4 = "OP18-051"


def _gone_to_deck(p, seat, card_id):
    """盤面の公開情報ではデッキの中身が見えない＝場・手札・トラッシュのどこにも無いことで「デッキへ」を確かめる。"""
    for z in ("field", "hand", "trash"):
        assert card_id not in p.ids(seat, z), (seat, z, card_id)


# --- OP06-056 天叢雲剣 --------------------------------------------------------------------------

def test_amenomurakumo_second_tier_cannot_reselect_the_first_pick():
    s = Scenario(turn=5)
    s.don["p1"] = 5
    s.put("p1", AMENOMURAKUMO, "hand")
    s.put("p2", VANILLA1, "field")
    s.put("p2", BELLAMY2, "field")
    p = s.build()
    p.play("P1", AMENOMURAKUMO)
    pe = p.pending()
    assert pe["player_id"] == "P1" and pe["action"] == "SEARCH_AND_SELECT", pe      # 1 つ目（コスト2以下）
    first = p.uuid("p2", "field", VANILLA1)
    assert first in pe["selectable_uuids"]
    p.resolve("P1", [first])
    pe = p.pending()
    if pe and pe.get("action") == "SEARCH_AND_SELECT":
        # 2 つ目（コスト1以下）の候補に、1 つ目で選んだドーマは出ない（ベラミーはコスト2で対象外＝候補 0）。
        assert first not in pe["selectable_uuids"], pe
        p.resolve("P1", [])
    p.settle()
    assert p.ids("p2", "field") == [BELLAMY2]
    _gone_to_deck(p, "p2", VANILLA1)


# --- OP18-051 Mr.4＆ミス・メリークリスマス ---------------------------------------------------------

def _mr4(p1_extra, p2_cards):
    s = Scenario(turn=5)
    s.don["p1"] = 10
    s.put("p1", MR4, "hand")
    for cid in p1_extra:
        s.put("p1", cid, "field")
    for cid in p2_cards:
        s.put("p2", cid, "field")
    p = s.build()
    p.play("P1", MR4)
    pe = p.pending()
    assert pe["player_id"] == "P1" and pe["action"] == "CONFIRM_OPTIONAL", pe       # コスト（ドン!!5枚をレスト）
    p.answer(accept=True)
    return p


def test_mr4_two_opponent_characters_are_ordered_by_their_owner():
    p = _mr4([], [BELLAMY2, VANILLA1])
    a, b = p.uuid("p2", "field", BELLAMY2), p.uuid("p2", "field", VANILLA1)
    pe = p.pending()
    assert pe["player_id"] == "P1" and pe["action"] == "SEARCH_AND_SELECT", pe      # 1 つ目（コスト6以下）
    p.resolve("P1", [a])
    pe = p.pending()
    assert pe["player_id"] == "P1" and pe["action"] == "SEARCH_AND_SELECT", pe      # 2 つ目（コスト5以下）
    assert a not in pe["selectable_uuids"]
    p.resolve("P1", [b])
    pe = p.pending()
    assert pe["player_id"] == "P2" and pe["action"] == "ARRANGE_DECK", pe           # 並べるのは持ち主（相手）
    p.resolve("P2", list(reversed(pe["selectable_uuids"])))
    assert p.action() == "MAIN_ACTION"
    assert p.ids("p2", "field") == []
    _gone_to_deck(p, "p2", BELLAMY2)
    _gone_to_deck(p, "p2", VANILLA1)


def test_mr4_one_card_from_each_side_needs_no_ordering():
    p = _mr4([VANILLA1], [VANILLA1])
    mine, theirs = p.uuid("p1", "field", VANILLA1), p.uuid("p2", "field", VANILLA1)
    pe = p.pending()
    assert pe["action"] == "SEARCH_AND_SELECT", pe
    p.resolve("P1", [theirs])
    pe = p.pending()
    assert pe["action"] == "SEARCH_AND_SELECT", pe
    p.resolve("P1", [mine])
    pe = p.pending()
    assert not (pe and pe.get("action") == "ARRANGE_DECK"), pe                       # 1 枚ずつ＝並び替えなし
    p.settle()
    _gone_to_deck(p, "p1", VANILLA1)
    _gone_to_deck(p, "p2", VANILLA1)
