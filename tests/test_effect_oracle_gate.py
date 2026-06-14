"""効果オラクル（静的 text↔AST 整合性）のラチェット品質ゲート。

`tests/effect_oracle.py` の検出カテゴリ（HAS_OTHER / PER_TURN_LIMIT_GAP / UP_TO_GAP）を
pytest から実行し、いずれも **0 件**であることを固定する（解消済みの差異が再び開かないよう
ラチェットする）。新たに見つかった真の乖離はカードを直してから 0 に戻すこと。
"""
import os

import conftest  # noqa: F401

from collections import Counter

from effect_oracle import detect
from opcg_sim.src.utils.loader import CardLoader

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "opcg_sim", "data", "opcg_cards.json")


def _counts():
    db = CardLoader(DATA)
    db.load()
    for cid in list(db.raw_db.keys()):
        db.get_card(cid)
    return Counter(f["category"] for f in detect(db)), detect(db)


def test_oracle_categories_are_zero():
    counts, findings = _counts()
    detail = {cat: [f["card_id"] for f in findings if f["category"] == cat]
              for cat in ("HAS_OTHER", "PER_TURN_LIMIT_GAP", "UP_TO_GAP")}
    assert counts.get("HAS_OTHER", 0) == 0, detail["HAS_OTHER"]
    assert counts.get("PER_TURN_LIMIT_GAP", 0) == 0, detail["PER_TURN_LIMIT_GAP"]
    assert counts.get("UP_TO_GAP", 0) == 0, detail["UP_TO_GAP"]
