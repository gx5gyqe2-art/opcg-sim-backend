"""サンプル精査ハーネス（§8.4 ✓信頼度の数値化・任意）。

`docs/TEST_SPEC.md` §8.6 の「弾×色 ✓」は自動ラチェット（構造監査/オラクル/挙動ベースライン）で
退行は守られているが、**自動ゲートが測らない意味的欠陥（カテゴリH 型の見落とし）** の残存率は
人手の精査でしか測れない。本ハーネスは:

  1. 各弾から **決定的（seed 固定）** にランダム抽出する（再現可能な標本）。
  2. 標本各カードに **自動スクリーニング信号** を付す:
       - HAS_OTHER（AST に ActionType.OTHER ＝未実装句）
       - STRUCT（構造不変条件4スキャンのいずれか＝H/Duration/chooser/すべて）
       - ORACLE（effect_oracle の高シグナル検出）
       - TEXT_ACTION_GAP（テキストの主要動詞が AST のアクション型に現れない疑い）
  3. 精査対象の素材（生テキスト＋AST要約＋実行観測）を `--dump` で人手精査用に出力する。

自動信号が 0 でも意味欠陥はあり得る（ベースラインは凍結する）。最終判定は人手（精査ログは
`docs/reports/` のスナップショットへ）。

使い方:
    OPCG_LOG_SILENT=1 python tests/sample_audit.py --per-set 10 --seed 20260615
    OPCG_LOG_SILENT=1 python tests/sample_audit.py --per-set 10 --seed 20260615 --dump > /tmp/sample.txt
"""
import argparse
import collections
import os
import random
import re

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

import effect_coverage as cov
import effect_oracle
import leader_spec_probe as P
import structural_invariants as si
from opcg_sim.src.models.effect_types import GameAction, Sequence, Branch, Choice
from opcg_sim.src.models.enums import ActionType
from opcg_sim.src.utils.loader import CardLoader

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "opcg_sim", "data", "opcg_cards.json")

# テキスト動詞 → 期待される ActionType 群（TEXT_ACTION_GAP の最小ヒューリスティック）。
# 「テキストにこの語があるのに AST に対応アクションが全く無い」場合に疑う。
_VERB_ACTIONS = {
    "KOする": {ActionType.KO}, "KOできる": {ActionType.KO},
    "登場させ": {ActionType.PLAY_CARD},
    "引く": {ActionType.DRAW},
    "捨て": {ActionType.DISCARD},
    "レストにする": {ActionType.REST}, "レストにできる": {ActionType.REST},
    "アクティブにする": {ActionType.ACTIVE, ActionType.ACTIVE_DON},
    "手札に戻す": {ActionType.MOVE_CARD, ActionType.BOUNCE, ActionType.MOVE_TO_HAND},
    "手札に加える": {ActionType.MOVE_CARD, ActionType.MOVE_TO_HAND},
}


def _all_actions(node):
    if isinstance(node, GameAction):
        yield node
    elif isinstance(node, Sequence):
        for a in node.actions:
            yield from _all_actions(a)
    elif isinstance(node, Branch):
        yield from _all_actions(node.if_true)
        yield from _all_actions(node.if_false)
    elif isinstance(node, Choice):
        for o in node.options:
            yield from _all_actions(o)


def _text_action_gap(master) -> str:
    """テキストの主要動詞に対応するアクション型が AST に全く無いものを返す（疑いのみ）。"""
    text = ""
    types = set()
    for ab in master.abilities:
        text += (ab.raw_text or "")
        for nd in (ab.cost, ab.effect):
            for a in _all_actions(nd):
                types.add(a.type)
    gaps = []
    for verb, expected in _VERB_ACTIONS.items():
        if verb in text and not (types & expected):
            gaps.append(verb)
    return ",".join(gaps)


def _set_of(cid):
    m = re.match(r"([A-Z]+\d+|[A-Z]+)", cid)
    return m.group(1) if m else cid


def sample(per_set: int, seed: int):
    db = CardLoader(DATA)
    db.load()
    by_set = collections.defaultdict(list)
    for cid in sorted(db.raw_db.keys()):
        m = db.get_card(cid)
        if m is None or not m.abilities:
            continue
        by_set[_set_of(cid)].append(cid)
    rng = random.Random(seed)
    picked = []
    for s in sorted(by_set):
        ids = by_set[s]
        picked.extend(rng.sample(ids, min(per_set, len(ids))))
    return db, picked


def screen(db, picked):
    # 構造スキャン（全カード）→ 標本に該当する card_id を抽出。
    struct = si.scan(db)
    struct_ids = {cid for cat in struct for cid, _, _ in struct[cat]}
    # オラクル検出（全カード）→ card_id 集合。
    oracle_ids = {f["card_id"] for f in effect_oracle.detect(db)}

    rows = []
    for cid in picked:
        m = db.get_card(cid)
        has_other = cov._has_other_in(m.abilities)
        gap = _text_action_gap(m)
        flags = []
        if has_other:
            flags.append("HAS_OTHER")
        if cid in struct_ids:
            flags.append("STRUCT")
        if cid in oracle_ids:
            flags.append("ORACLE")
        if gap:
            flags.append(f"TEXT_ACTION_GAP[{gap}]")
        rows.append((cid, m.name, _set_of(cid), flags))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-set", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--dump", action="store_true", help="精査素材（生テキスト＋AST＋観測）を出力")
    args = ap.parse_args()

    db, picked = sample(args.per_set, args.seed)
    rows = screen(db, picked)
    flagged = [r for r in rows if r[3]]

    print("=== サンプル精査スクリーニング ===")
    print(f"seed={args.seed} per_set={args.per_set} 標本={len(picked)}枚 弾数={len({r[2] for r in rows})}")
    print(f"自動信号フラグ付き: {len(flagged)} 枚")
    for cid, name, s, flags in flagged:
        print(f"  [{','.join(flags)}] {cid} {name}")

    if args.dump:
        print("\n" + "#" * 78)
        print("# 精査素材（人手 §8.4 チェックリスト用）")
        print("#" * 78)
        for cid in picked:
            print(P.fmt(P.probe(cid)))


if __name__ == "__main__":
    main()
