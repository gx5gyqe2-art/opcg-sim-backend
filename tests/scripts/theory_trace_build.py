#!/usr/bin/env python3
"""理論の注釈つき棋譜（`theory_trace.py` の JSON）を、**カード画像を同梱した 1 枚の HTML** に組み立てる。

ビューアーの雛形（`docs/tools/theory_trace_viewer.html`）の 2 か所の差し込み口
（`/*__TRACE_DATA__*/null` と `/*__CARD_IMAGES__*/null`）に、棋譜の JSON と
`{card_id: "data:image/webp;base64,…"}` を埋め込む。画像はフロントエンドと同じ公開バケット
（`storage.googleapis.com/opcg-images/<card_id>.png`）から取り、縮小して WebP にする
（1 枚 10KB 前後）。取った画像は `--cache` に置き、2 回目からは取りに行かない。

埋め込まない（`--no-images`）と、ビューアーはバケットの URL を直接読む
（手元のブラウザで開くなら十分・アーティファクトのように外部画像を禁じる場では映らない）。

使い方:

    python tests/scripts/theory_trace_build.py --trace trace.json --out viewer.html [--cache DIR]
"""

import argparse
import base64
import io
import json
import os
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
TEMPLATE = os.path.join(_ROOT, "docs", "tools", "theory_trace_viewer.html")
IMAGE_BASE = "https://storage.googleapis.com/opcg-images"
#: 盤面の絵で使う固定の画像（カードの裏・ドン!!の表）。キーはビューアーと共有する。
SPECIAL = {"__back": "OPCG_back", "__don": "DON"}
TRACE_SLOT = "/*__TRACE_DATA__*/null"
IMAGE_SLOT = "/*__CARD_IMAGES__*/null"
SIZE = (160, 224)          # 600×838 の元画像とほぼ同じ縦横比


def card_ids(trace):
    """棋譜に現れる全カードの card_id（リーダー・場・手札・ステージ・トラッシュの一番上）。"""
    ids = set()
    for g in trace.get("games") or []:
        for d in g.get("decisions") or []:
            for side in (d.get("board") or {}).values():
                for c in [side.get("leader"), side.get("stage")] + list(side.get("field") or []) \
                        + list(side.get("hand") or []):
                    if isinstance(c, dict) and c.get("card_id"):
                        ids.add(c["card_id"])
                if side.get("trash_top"):
                    ids.add(side["trash_top"])
    return sorted(ids)


def to_webp(png_bytes, size=SIZE, quality=70):
    from PIL import Image
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB").resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=quality, method=6)
    return buf.getvalue()


def fetch(name, cache):
    """`<cache>/<name>.webp` が在ればそれを、無ければバケットから取って縮小して置く。取れなければ `None`。"""
    path = os.path.join(cache, name + ".webp")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    try:
        with urllib.request.urlopen(f"{IMAGE_BASE}/{name}.png", timeout=30) as r:
            data = to_webp(r.read())
    except Exception:                                      # noqa: BLE001  1 枚欠けても組み立ては止めない
        return None
    os.makedirs(cache, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return data


def data_uri(webp_bytes):
    return "data:image/webp;base64," + base64.b64encode(webp_bytes).decode("ascii")


def build_html(template, trace, images):
    """雛形の 2 つの差し込み口に棋譜と画像表を入れる（`</` は `<\\/` に逃がして script を閉じさせない）。"""
    if TRACE_SLOT not in template or IMAGE_SLOT not in template:
        raise ValueError("雛形に差し込み口が無い")
    t = json.dumps(trace, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    i = json.dumps(images, separators=(",", ":")) if images is not None else "null"
    return template.replace(TRACE_SLOT, t, 1).replace(IMAGE_SLOT, i, 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="棋譜 JSON → カード画像つきの単体ビューアー HTML")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=TEMPLATE)
    ap.add_argument("--cache", default=os.path.join(os.path.expanduser("~"), ".cache", "opcg_card_webp"))
    ap.add_argument("--no-images", action="store_true")
    a = ap.parse_args(argv)
    with open(a.trace, encoding="utf-8") as f:
        trace = json.load(f)
    with open(a.template, encoding="utf-8") as f:
        template = f.read()
    images, missing = None, []
    if not a.no_images:
        images = {}
        for key, name in list(SPECIAL.items()) + [(c, c) for c in card_ids(trace)]:
            b = fetch(name, a.cache)
            if b is None:
                missing.append(name)
            else:
                images[key] = data_uri(b)
    html = build_html(template, trace, images)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(json.dumps({"out": a.out, "bytes": len(html.encode("utf-8")),
                      "images": (len(images) if images is not None else 0), "missing": missing},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
