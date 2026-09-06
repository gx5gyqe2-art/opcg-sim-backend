"""パース済みの効果構造（Ability / EffectNode / Condition / TargetQuery / ValueSource）を
JSON へ落とすツール（Rust エンジン移行 P0・`docs/rust_engine_plan.md` §3・原則 3）。

パーサは Python に残す方針なので、Rust 側は「カード本文 → 効果構造」の結果だけを読む。
本ツールはその受け渡しファイル（既定 `opcg_sim/data/opcg_effects.json`）を生成する。
併せて ActionType / ConditionType / TriggerType の出現頻度表を stdout に出す
（P3 の作業パッケージ分割＝「頻度順に実装」の材料）。

使い方:
    python -m opcg_sim.tools.export_effects_json               # 既定パスへ書き出し＋頻度表
    python -m opcg_sim.tools.export_effects_json --out /tmp/e.json --indent 2 --top 20
    python -m opcg_sim.tools.export_effects_json --dry-run     # 書き出さず頻度表だけ

出力（`version` は形を非互換に変えたら +1 する）:
    {"version": 1,
     "source": {"card_db": ..., "db_hash": ..., "parser": "v2"},
     "counts": {"cards":.., "cards_with_ability":.., "abilities":.., "nodes":..,
                "action_types": {...}, "condition_types": {...}, "trigger_types": {...}},
     "cards": {"OP01-001": {..., "abilities": [<Ability ノード>...]}, ...}}

ノードの形（Rust 側は "node" で dispatch する）:
    Ability     {"node":"Ability","trigger":str,"condition":Condition|null,
                 "cost":Node|null,"effect":Node|null,"raw_text":str,"cost_optional":bool}
    GameAction  {"node":"GameAction","type":str,"target":TargetQuery|null,"value":ValueSource, ...}
    Sequence    {"node":"Sequence","actions":[Node...]}
    Choice      {"node":"Choice","message":str,"options":[Node...],"option_labels":[str...],"player":str}
    Branch      {"node":"Branch","condition":Condition|null,"if_true":Node|null,"if_false":Node|null}
    Condition   {"node":"Condition","type":str,"target":TargetQuery|null,"player":str,
                 "operator":str,"value":int|str|ValueSource,"args":[Condition...],"raw_text":str}
    TargetQuery {"node":"TargetQuery", ...(dataclass のフィールドそのまま)}
    ValueSource {"node":"ValueSource","base":int,...}

規約: Enum は**名前文字列**（`ActionType.KO` → "KO"）、None は null、set/tuple は
（決定論のため set はソートして）list。dataclass のフィールドは省略せず全て出す
（Rust 側で「読み込み時に未知キーはエラー」にできるよう、形を固定する）。
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

from opcg_sim.src.models.effect_types import (
    Ability, Branch, Choice, Condition, GameAction, Sequence, TargetQuery, ValueSource,
)
from opcg_sim.src.utils.loader import CardLoader, make_parser

# このファイル: opcg_sim/tools/export_effects_json.py -> data は opcg_sim/data
_DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
CARD_DB_PATH = os.path.join(_DATA_DIR, "opcg_cards.json")
DEFAULT_OUT = os.path.join(_DATA_DIR, "opcg_effects.json")

# 出力形式のバージョン（Rust 側の読み込みと対で管理する）。
EXPORT_VERSION = 1

# dataclass 名 -> 出力の "node" 名。ここに無い dataclass が出てきたら型名をそのまま使い、
# 集計時に unknown として警告する（黙って捨てない＝計画 §6「未知の種類はエラー」に備える）。
_NODE_NAMES = {
    Ability: "Ability",
    GameAction: "GameAction",
    Sequence: "Sequence",
    Branch: "Branch",
    Choice: "Choice",
    Condition: "Condition",
    TargetQuery: "TargetQuery",
    ValueSource: "ValueSource",
}


class Stats:
    """走査中に貯める頻度カウンタ。"""

    def __init__(self) -> None:
        self.action_types: Counter = Counter()
        self.condition_types: Counter = Counter()
        self.trigger_types: Counter = Counter()
        self.node_kinds: Counter = Counter()
        self.unknown_nodes: Counter = Counter()


def _encode(obj: Any, stats: Stats) -> Any:
    """効果構造の任意の値を JSON 化可能な形へ変換する（Enum→名前 / set→ソート list）。"""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, enum.Enum):
        return obj.name
    if isinstance(obj, (set, frozenset)):
        # set は順不同なので、決定論のため文字列化してソートする。
        return sorted((_encode(v, stats) for v in obj), key=repr)
    if isinstance(obj, (list, tuple)):
        return [_encode(v, stats) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _encode(v, stats) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        name = _NODE_NAMES.get(cls)
        if name is None:
            name = cls.__name__
            stats.unknown_nodes[name] += 1
        stats.node_kinds[name] += 1
        if cls is GameAction:
            stats.action_types[obj.type.name if isinstance(obj.type, enum.Enum) else str(obj.type)] += 1
        elif cls is Condition:
            stats.condition_types[obj.type.name if isinstance(obj.type, enum.Enum) else str(obj.type)] += 1
        elif cls is Ability:
            stats.trigger_types[obj.trigger.name if isinstance(obj.trigger, enum.Enum) else str(obj.trigger)] += 1
        out: Dict[str, Any] = {"node": name}
        for f in dataclasses.fields(obj):
            out[f.name] = _encode(getattr(obj, f.name), stats)
        return out
    # 想定外（EffectNode のサブクラスでない何か）は型名を添えて文字列化する。
    stats.unknown_nodes[type(obj).__name__] += 1
    return {"node": type(obj).__name__, "repr": repr(obj)}


def export(db: CardLoader, stats: Stats) -> Dict[str, Any]:
    """CardLoader の全カードを効果構造つきの dict へ落とす（card_id 昇順で決定論）。"""
    cards: Dict[str, Any] = {}
    for card_id in sorted(db.raw_db.keys()):
        master = db.get_card(card_id)
        if master is None:      # ダミー行など（_create_card_master が None を返す）
            continue
        cards[master.card_id] = {
            "card_id": master.card_id,
            "name": master.name,
            "type": master.type.name,
            "colors": [c.name for c in master.colors],
            "cost": master.cost,
            "power": master.power,
            "counter": master.counter,
            "attribute": master.attribute.name,
            "traits": list(master.traits),
            "life": master.life,
            "block_icon": master.block_icon,
            "keywords": sorted(master.keywords),
            "name_aliases": list(master.name_aliases),
            "effect_text": master.effect_text,
            "trigger_text": master.trigger_text,
            "abilities": [_encode(ab, stats) for ab in master.abilities],
        }
    return cards


def _table(counter: Counter, total: int, top: int) -> List[str]:
    lines = []
    for i, (key, n) in enumerate(counter.most_common(top), 1):
        share = (100.0 * n / total) if total else 0.0
        lines.append(f"  {i:>3}. {key:<24} {n:>6}  {share:5.1f}%")
    rest = len(counter) - min(top, len(counter))
    if rest > 0:
        lines.append(f"       …他 {rest} 種")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="パース済み効果構造を JSON へ書き出す（Rust 移行 P0）")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"出力パス（既定: {DEFAULT_OUT}）")
    ap.add_argument("--card-db", default=CARD_DB_PATH, help="カード DB の JSON")
    ap.add_argument("--indent", type=int, default=0, help="JSON のインデント（0=最小化・既定）")
    ap.add_argument("--top", type=int, default=20, help="頻度表に出す上位件数（既定 20）")
    ap.add_argument("--dry-run", action="store_true", help="書き出さず頻度表だけ出す")
    args = ap.parse_args(argv)

    db = CardLoader(args.card_db)
    db.load()
    stats = Stats()
    cards = export(db, stats)

    n_abilities = sum(len(c["abilities"]) for c in cards.values())
    n_with = sum(1 for c in cards.values() if c["abilities"])
    payload = {
        "version": EXPORT_VERSION,
        "source": {
            "card_db": os.path.basename(args.card_db),
            "db_hash": db.db_hash(),
            "parser": os.environ.get("OPCG_PARSER", "v2"),
            "parser_class": type(make_parser()).__name__,
        },
        "counts": {
            "cards": len(cards),
            "cards_with_ability": n_with,
            "abilities": n_abilities,
            "nodes": sum(stats.node_kinds.values()),
            "action_types": dict(sorted(stats.action_types.items())),
            "condition_types": dict(sorted(stats.condition_types.items())),
            "trigger_types": dict(sorted(stats.trigger_types.items())),
            "node_kinds": dict(sorted(stats.node_kinds.items())),
        },
        "cards": cards,
    }

    size = 0
    if not args.dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True,
                      indent=(args.indent or None),
                      separators=((", ", ": ") if args.indent else (",", ":")))
            f.write("\n")
        os.replace(tmp, args.out)
        size = os.path.getsize(args.out)

    n_actions = sum(stats.action_types.values())
    n_conds = sum(stats.condition_types.values())
    print(f"[export_effects_json] cards={len(cards)} with_ability={n_with} "
          f"abilities={n_abilities} nodes={sum(stats.node_kinds.values())} "
          f"action_nodes={n_actions} condition_nodes={n_conds}")
    print(f"[export_effects_json] ActionType={len(stats.action_types)}種 "
          f"ConditionType={len(stats.condition_types)}種 "
          f"TriggerType={len(stats.trigger_types)}種")
    if not args.dry_run:
        print(f"[export_effects_json] wrote {args.out} ({size/1e6:.2f} MB)")

    print(f"\n== ActionType 出現頻度（上位 {args.top}／全 {n_actions} ノード） ==")
    print("\n".join(_table(stats.action_types, n_actions, args.top)))
    print(f"\n== ConditionType 出現頻度（上位 {args.top}／全 {n_conds} ノード） ==")
    print("\n".join(_table(stats.condition_types, n_conds, args.top)))
    print(f"\n== TriggerType 出現頻度（全 {sum(stats.trigger_types.values())} 能力） ==")
    print("\n".join(_table(stats.trigger_types, sum(stats.trigger_types.values()), args.top)))

    if stats.unknown_nodes:
        print(f"\n[warn] 未知ノード種別: {dict(stats.unknown_nodes)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
