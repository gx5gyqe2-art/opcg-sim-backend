"""検証バケットの回帰アサート裏付け（§8.2 台帳の「✓」を機械保証へ）。

弾×色バケット（EB/PRB/P・ST01-30・OP01-16）の検証は §8.2 台帳への「✓」記録（人手）で
積み上げてきたが、その「✓」自体はテストで固定されていなかった（ドキュメント上の主張）。
本テストは、検証済み弾の各カードが以下で**機械的に**裏付けられることを固定する:

  1. 全カードの全能力が**挙動ベースライン（full_card_baseline.json）に指紋として登録**されている
     ＝以後の挙動変化（退行）は test_full_card_baseline で必ず検出される。
  2. 検証済みバケットに**カテゴリH 構造違反が無い**（先頭ゲート漏れ＝0）。

これにより「台帳の ✓」が「ベースライン＋構造不変条件による継続保証」へ格上げされる
（docs/TEST_SPEC.md §8.5 二層回帰モデル）。
"""
import json
import os
import re

import conftest  # noqa: F401

import structural_invariants as si
from opcg_sim.src.utils.loader import CardLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "opcg_sim", "data", "opcg_cards.json")
BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fixtures", "full_card_baseline.json")

# §8.2 台帳で「✓（一巡済み）」とした弾の接頭辞。これらは弾×色で検証済み。
VERIFIED_SET_RE = re.compile(r'^(OP\d{2}|EB\d{2}|PRB\d{2}|ST\d{2}|P)-')


def _verified_card_ids(db):
    return [cid for cid in sorted(db.raw_db.keys()) if VERIFIED_SET_RE.match(cid)]


def test_verified_buckets_have_behavior_baseline():
    """検証済み弾の全カード・全トリガーがベースライン指紋に登録されている。"""
    db = CardLoader(DATA)
    db.load()
    with open(BASELINE, encoding="utf-8") as f:
        base = set(json.load(f).keys())

    missing = []
    for cid in _verified_card_ids(db):
        master = db.get_card(cid)
        if master is None or not master.abilities:
            continue
        seen = set()
        for ab in master.abilities:
            trig = ab.trigger.name if hasattr(ab.trigger, "name") else str(ab.trigger)
            if trig in seen:
                continue
            seen.add(trig)
            if f"{cid}|{trig}" not in base:
                missing.append(f"{cid}|{trig}")
    assert not missing, (
        f"ベースライン未登録の検証済みカード {len(missing)}: " + ", ".join(missing[:15]))


def test_verified_buckets_have_no_category_h():
    """検証済み弾にカテゴリH（先頭ゲート漏れ）が残っていない（H是正で全弾0）。"""
    db = CardLoader(DATA)
    db.load()
    findings = si.scan(db)
    h = [(cid, trig) for cid, trig, _ in findings.get("H_LEADING_GATE_LEAK", [])
         if VERIFIED_SET_RE.match(cid)]
    assert not h, "検証済み弾の H 残: " + ", ".join(f"{c}/{t}" for c, t in h[:15])
