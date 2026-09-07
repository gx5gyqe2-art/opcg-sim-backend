"""現実的デッキ生成器（`deckgen.build_realistic_deck`）の健全性スモーク。

人間観点の評価（公平モード・実デッキで凡ミス/破綻を観察）の土台。合成デッキ（同色キャラ50・イベント無し）
より実戦に近い「イベント含む・4枚積み・カーブあり」デッキを生成できることを固定する。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_deckgen.py -q -s -p no:cacheprovider
"""
import random
from collections import Counter

import conftest  # noqa: F401
import pytest

import deckgen
from cpu_selfplay import _load_db


@pytest.fixture(scope="module")
def db():
    return _load_db()


@pytest.mark.parametrize("name,lid", list(deckgen.VERIFIED_LEADERS.items()))
def test_realistic_deck_invariants(db, name, lid):
    leader, cards = deckgen.build_realistic_deck(db, "p1", lid, random.Random(0))
    assert leader.master.type.name == "LEADER"
    assert len(cards) == deckgen.DECK_SIZE, f"{name}: 50枚でない ({len(cards)})"
    # 4枚積み上限を厳守
    cc = Counter(c.master.card_id for c in cards)
    assert max(cc.values()) <= deckgen.MAX_COPIES, f"{name}: コピー超過 {max(cc.values())}"
    # イベント（カウンター/除去）を含む＝合成デッキとの主な違い
    types = Counter(c.master.type.name for c in cards)
    assert types.get("EVENT", 0) > 0, f"{name}: イベントを含まない"
    # リーダー色と整合（少なくとも過半が色一致）
    lcolors = set(leader.master.colors or [])
    if lcolors:
        match = sum(1 for c in cards if set(c.master.colors or []) & lcolors)
        assert match >= len(cards) * 0.6, f"{name}: 色一致が少なすぎ ({match}/{len(cards)})"
