"""リーダーカード仕様書作成のためのプローブ。

各リーダーについて以下を1枚分まとめて出力する。テキスト精読 → 期待挙動 →
実観測挙動の突き合わせ（既存ツールが拾わない意味バグの発見）を支援する。

  - 生テキスト（効果欄）
  - パース結果（能力ごとの summarize_ability 指紋）
  - 実行時の観測挙動（effect_coverage.classify による status と盤面差分）

使い方:
    OPCG_LOG_SILENT=1 python tests/leader_spec_probe.py OP01-001
    OPCG_LOG_SILENT=1 python tests/leader_spec_probe.py --all            # 全リーダー
    OPCG_LOG_SILENT=1 python tests/leader_spec_probe.py --set OP01       # セット指定
    OPCG_LOG_SILENT=1 python tests/leader_spec_probe.py --json OP01-001  # 機械可読
"""
import argparse
import json
import os
import re
import sys

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

import effect_coverage as cov
from golden.summarize import summarize_ability
from opcg_sim.src.utils.loader import CardLoader
from opcg_sim.src.models.enums import CardType

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "opcg_sim", "data", "opcg_cards.json",
)

_DB = None


def db():
    global _DB
    if _DB is None:
        _DB = CardLoader(DATA)
        _DB.load()
    return _DB


def leader_ids():
    return sorted(cid for cid, raw in db().raw_db.items()
                  if raw.get("種類") == "リーダー")


def _set_of(card_id):
    m = re.match(r"([A-Z]+\d+|[A-Z]+)", card_id)
    return m.group(1) if m else card_id


def _trigger_name(ab):
    return ab.trigger.name if hasattr(ab.trigger, "name") else str(ab.trigger)


def probe(card_id):
    raw = db().raw_db.get(card_id, {})
    m = db().get_card(card_id)
    abilities = []
    for ab in (m.abilities or []):
        abilities.append({
            "trigger": _trigger_name(ab),
            "summary": summarize_ability(ab),
        })
    results = []
    for r in cov.classify(m):
        results.append({
            "trigger": r.trigger,
            "status": r.status,
            "has_other": r.has_other,
            "detail": r.detail,
            "select_issues": r.select_issues,
        })
    return {
        "id": card_id,
        "name": m.name,
        "color": raw.get("色"),
        "life": raw.get("ライフ"),
        "power": raw.get("パワー"),
        "attribute": raw.get("属性"),
        "traits": raw.get("特徴"),
        "text": raw.get("効果(テキスト)", ""),
        "trigger_text": raw.get("トリガー", ""),
        "abilities": abilities,
        "observed": results,
    }


def fmt(p):
    out = []
    out.append("=" * 78)
    out.append(f"{p['id']}  {p['name']}  [{p['color']}] life={p['life']} power={p['power']} attr={p['attribute']}")
    out.append(f"特徴: {p['traits']}")
    out.append("-- 効果テキスト --")
    out.append(p["text"] or "(なし)")
    if p.get("trigger_text"):
        out.append(f"-- トリガー欄 --\n{p['trigger_text']}")
    out.append("-- パース結果 (能力ごと) --")
    for i, a in enumerate(p["abilities"]):
        out.append(f"  [{i}] trigger={a['trigger']}")
        out.append("      " + json.dumps(a["summary"], ensure_ascii=False, default=str))
    out.append("-- 実行観測 (classify) --")
    for r in p["observed"]:
        line = f"  {r['trigger']}: {r['status']}  | {r['detail']}"
        if r["select_issues"]:
            line += f"  | select_issues: {r['select_issues']}"
        out.append(line)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("card", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--set", dest="set_")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.all:
        ids = leader_ids()
    elif args.set_:
        ids = [c for c in leader_ids() if _set_of(c) == args.set_]
    elif args.card:
        ids = [args.card]
    else:
        ap.error("card id か --all/--set を指定してください")

    if args.json:
        print(json.dumps([probe(c) for c in ids], ensure_ascii=False, indent=2, default=str))
    else:
        for c in ids:
            print(fmt(probe(c)))
            print()


if __name__ == "__main__":
    main()
