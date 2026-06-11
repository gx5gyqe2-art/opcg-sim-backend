"""品質ゲート: テキスト↔実行 一致度カウンタのラチェット式回帰ガード。

各カウンタの上限を固定し、修正フェーズごとに引き下げる（上げる変更は退行）。
カウンタの出所:
  - quality_map.collect_buckets : WARN_DIRECTION / STAT_ONLY / NO_IMPL / SELECT_MISMATCH
  - EffectParserV2.unmatched    : レガシーフォールバック原子句数

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_quality_gates.py -p no:capture -q
"""
import json
import os

import conftest  # noqa: F401
import pytest

import quality_map

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "opcg_sim", "data")

# ラチェット上限（現状値で固定。修正が進んだら引き下げる）
MAX_WARN_DIRECTION  = 42
MAX_STAT_ONLY       = 15
MAX_NO_IMPL         = 0
MAX_SELECT_MISMATCH = 3
MAX_FALLBACK        = 19


@pytest.fixture(scope="module")
def buckets():
    return quality_map.collect_buckets()


def test_warn_direction_ratchet(buckets):
    items = buckets.get("WARN_DIRECTION", [])
    assert len(items) <= MAX_WARN_DIRECTION, (
        f"WARN_DIRECTION {len(items)} > {MAX_WARN_DIRECTION}: "
        + ", ".join(f"{r.card_id}/{r.trigger}" for r in items[:10]))


def test_stat_only_ratchet(buckets):
    items = buckets.get("STAT_ONLY", [])
    assert len(items) <= MAX_STAT_ONLY, (
        f"STAT_ONLY {len(items)} > {MAX_STAT_ONLY}: "
        + ", ".join(f"{r.card_id}/{r.trigger}" for r in items[:10]))


def test_no_impl_ratchet(buckets):
    items = buckets.get("NO_IMPL", [])
    assert len(items) <= MAX_NO_IMPL, (
        f"NO_IMPL {len(items)} > {MAX_NO_IMPL}: "
        + ", ".join(f"{r.card_id}/{r.trigger}" for r in items[:10]))


def test_select_mismatch_ratchet(buckets):
    items = buckets.get("SELECT_MISMATCH", [])
    assert len(items) <= MAX_SELECT_MISMATCH, (
        f"SELECT_MISMATCH {len(items)} > {MAX_SELECT_MISMATCH}: "
        + ", ".join(f"{r.card_id}/{r.trigger}" for r in items[:10]))


def test_parser_fallback_ratchet():
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    parser = EffectParserV2()
    with open(os.path.join(DATA, "opcg_cards.json"), encoding="utf-8") as f:
        cards = json.load(f)
    key = "効果(テキスト)"
    for c in cards:
        t = (c.get(key) or "").strip()
        if not t:
            continue
        try:
            parser.parse_card_text(t)
        except Exception:
            pass
    n = len(parser.unmatched)
    assert n <= MAX_FALLBACK, (
        f"フォールバック原子句 {n} > {MAX_FALLBACK}: "
        + " / ".join(list(parser.unmatched)[:5]))
