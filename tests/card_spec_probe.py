"""弾×色での効果検証プローブ（非リーダー含む全カード対応）。

`leader_spec_probe.py` の per-card プローブ（生テキスト＋パース指紋＋classify 実観測）を
リーダー以外にも広げ、`--set`／`--color`／`--type` で「弾×色」のバケット単位に絞って
出力する。TEST_SPEC §8 の手動検証（デッキ単位 → 弾×色バケット単位）の起点に使う。

使い方:
    OPCG_LOG_SILENT=1 python tests/card_spec_probe.py --set OP16 --color 赤
    OPCG_LOG_SILENT=1 python tests/card_spec_probe.py --set OP16 --color 黒 --json
    OPCG_LOG_SILENT=1 python tests/card_spec_probe.py --set OP16            # 弾の全色
    OPCG_LOG_SILENT=1 python tests/card_spec_probe.py --set OP16 --buckets  # 色別の枚数一覧
    OPCG_LOG_SILENT=1 python tests/card_spec_probe.py OP16-001              # 1枚

色は主色一致（"緑/青" は --color 緑 でも --color 青 でもヒット）。
"""
import argparse
import collections
import json

import conftest  # noqa: F401

import leader_spec_probe as L


def _all_ids():
    return sorted(L.db().raw_db.keys())


def _color_match(raw_color, want):
    if not want:
        return True
    return want in (raw_color or "").split("/")


def select_ids(set_=None, color=None, type_=None):
    out = []
    for cid in _all_ids():
        raw = L.db().raw_db.get(cid, {})
        if set_ and L._set_of(cid) != set_:
            continue
        if not _color_match(raw.get("色"), color):
            continue
        if type_ and raw.get("種類") != type_:
            continue
        out.append(cid)
    return out


def buckets(set_):
    g = collections.defaultdict(list)
    for cid in select_ids(set_=set_):
        g[L.db().raw_db.get(cid, {}).get("色")].append(cid)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("card", nargs="?")
    ap.add_argument("--set", dest="set_")
    ap.add_argument("--color")
    ap.add_argument("--type", dest="type_", help="種類（キャラクター/イベント/ステージ/リーダー）")
    ap.add_argument("--buckets", action="store_true", help="弾の色別枚数だけ出す")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.buckets:
        if not args.set_:
            ap.error("--buckets は --set が必要")
        for col, ids in sorted(buckets(args.set_).items()):
            print(f"{col:8} {len(ids):3}  {ids[0]}..{ids[-1]}")
        return

    if args.card:
        ids = [args.card]
    elif args.set_ or args.color or args.type_:
        ids = select_ids(set_=args.set_, color=args.color, type_=args.type_)
    else:
        ap.error("card id か --set/--color/--type を指定してください")

    if args.json:
        print(json.dumps([L.probe(c) for c in ids], ensure_ascii=False, indent=2, default=str))
    else:
        for c in ids:
            print(L.fmt(L.probe(c)))
            print()


if __name__ == "__main__":
    main()
