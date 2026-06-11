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
MAX_WARN_DIRECTION  = 0
MAX_STAT_ONLY       = 0
MAX_NO_IMPL         = 0
MAX_SELECT_MISMATCH = 2
MAX_FALLBACK        = 0


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


# --- Phase 4 深層ハーネスのゲート ---------------------------------------------
MAX_SATISFIED_NO_CHANGE = 9   # 条件/コスト充足盤面で実行しても盤面が動かない（H-5）
MAX_BATTLE_NO_CHANGE    = 0   # バトル文脈で発火しても変化なし（H-6）
MAX_INTERACTIVE_AUDIT   = 0   # 対象クエリとテキストの矛盾（集約監査で誤検知を排除し 0 に締結）


def test_condition_synth_no_change_ratchet():
    """H-5: 条件/コストを満たした盤面で発動しても一切動かない能力の上限。

    残存はリーダーへの特徴付与・既にアクティブな対象・遅延効果・色フィルタ等の
    合成/測定限界（HANDOVER §7 残課題）。新規バグで増えたら気づけるよう固定する。"""
    import condition_synth
    buckets = condition_synth.collect()
    items = buckets.get("SATISFIED_NO_CHANGE", [])
    assert len(items) <= MAX_SATISFIED_NO_CHANGE, (
        f"SATISFIED_NO_CHANGE {len(items)} > {MAX_SATISFIED_NO_CHANGE}: "
        + ", ".join(f"{c}/{t}" for c, _, t, _ in items[:10]))
    assert not buckets.get("ERROR"), f"条件合成で例外: {buckets.get('ERROR')[:5]}"


def test_battle_coverage_no_errors():
    """H-6: バトル文脈（攻撃/ブロック/カウンター）で全トリガーが例外なく発火する。"""
    import battle_coverage
    buckets = battle_coverage.collect()
    assert not buckets.get("ERROR"), f"バトル文脈で例外: {buckets.get('ERROR')[:5]}"
    no_change = buckets.get("BATTLE_NO_CHANGE", [])
    assert len(no_change) <= MAX_BATTLE_NO_CHANGE, (
        f"BATTLE_NO_CHANGE {len(no_change)} > {MAX_BATTLE_NO_CHANGE}: "
        + ", ".join(f"{c}/{t}" for c, _, t, _ in no_change[:10]))


def test_interactive_audit_ratchet():
    """H-7: 対象クエリとテキストの矛盾検出（精度向上後の真陽性近似）。"""
    import interactive_target_audit
    flagged = interactive_target_audit.run(top=0)
    assert len(flagged) <= MAX_INTERACTIVE_AUDIT, (
        f"interactive audit {len(flagged)} > {MAX_INTERACTIVE_AUDIT}: "
        + ", ".join(n for n, _ in flagged[:12]))
