"""**移植の段 3**: 核の呼び出しの記録（`theory_core_rs`・`OPCG_THEORY_CORE_CAPTURE`）→ `cargo test` の golden（gzip）。

    python tests/scripts/theory_core_golden.py <記録のディレクトリ> rust/opcg_engine/tests/fixtures/theory_core_golden.jsonl.gz

`<記録のディレクトリ>/<源>/<器>/<名前>.jsonl`（源＝`rec`・`w41`・`w39`・残す候補の名前など）。

**間引きの規則**（固定）: 入力の字句（`fn`・`a`・`g`・`pre`）が同じ行は源をまたいで 1 つ。`(名前, 源)` ごとに入力の字句の CRC32 が
小さい順に `N` 行（`N` は名前ごと・`CAP`）。**`jv.value` は手札の読みごとにまとめて**選ぶ: 1 つの読みの物の覚え書き
（`_val`・`_plan`）は残った札だけが鍵＝呼び出しの履歴に依る（E52）ので、選んだ読みの `jv.value` は記録の順に**全部**入れる
（源ごとに手札の行の字句の CRC32 が小さい順に `JV_READINGS` 個）。札の読みの番号は `jv:<源>/<器>/<番号>` に付け替える。
手札の行 → 選んだ読みの `jv.value`（記録の順）→ 残り（名前と CRC32 の順）の順に書く。移植の間だけの道具（段 7 で消す）。
"""
import gzip
import hashlib
import heapq
import json
import os
import sys
import zlib

CAP = {"ev.card_value": 60, "hp.apply_inflow": 40, "hs.use_value": 80, "hs.free_value": 80, "sp.search_value": 30,
       "sp.card_gain": 60, "ev.continuous_self_mods": 60}
DEFAULT_CAP = 120
JV_READINGS = 8


def _dumps(v):
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def _key(rec):
    return _dumps([rec["fn"], rec["a"], rec["g"], rec.get("pre", [])])


def _dirs(root):
    for src in sorted(os.listdir(root)):
        sd = os.path.join(root, src)
        if not os.path.isdir(sd):
            continue
        for tool in sorted(os.listdir(sd)):
            td = os.path.join(sd, tool)
            if os.path.isdir(td):
                yield src, "%s/%s" % (src, tool), td


def main(argv=None):
    """2 回読む（記録は数 GB＝全部を覚えない）: 1 回目で手札の読みと各 `(名前, 源)` の CRC32 の小さい `N` 行を選び、
    2 回目で選んだ読みの `jv.value` を記録の順に集める。"""
    argv = sys.argv[1:] if argv is None else argv
    root, out = argv[0], argv[1]
    heaps = {}        # (fn, src) -> 最大ヒープ [(-crc, -h, 字句, 行)]（小さい N 個を残す）
    hands = {}        # hid -> 行
    readings = {}     # src -> [(crc, hid)]
    seen = set()
    for src, tag, td in _dirs(root):
        for fn in sorted(f for f in os.listdir(td) if f.endswith(".jsonl")):
            if fn == "jv_value.jsonl":
                continue
            with open(os.path.join(td, fn), encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec["fn"] == "tb.joint_valuer":
                        hid = "jv:%s/%s" % (tag, rec["r"]["obj"][3:])
                        rec["r"] = {"obj": hid}
                        hands[hid] = rec
                        readings.setdefault(src, []).append((zlib.crc32(_dumps(rec["a"]).encode()), hid))
                        continue
                    k = _key(rec)
                    h = hashlib.blake2b(k.encode(), digest_size=12).digest()
                    if h in seen:
                        continue
                    seen.add(h)
                    crc = zlib.crc32(k.encode())
                    hp = heaps.setdefault((rec["fn"], src), [])
                    n = CAP.get(rec["fn"], DEFAULT_CAP)
                    item = (-crc, k, rec)
                    if len(hp) < n:
                        heapq.heappush(hp, (-crc, _Rev(k), rec))
                    elif -crc > hp[0][0] or (-crc == hp[0][0] and _Rev(k) > hp[0][1]):
                        heapq.heapreplace(hp, (-crc, _Rev(k), rec))
                    del item
    chosen = []
    for src in sorted(readings):
        chosen.extend(hid for _c, hid in sorted(readings[src])[:JV_READINGS * 4])
    jv_lines = {hid: [] for hid in chosen}
    for src, tag, td in _dirs(root):
        p = os.path.join(td, "jv_value.jsonl")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                a = dict(rec["a"]["d"])
                hid = "jv:%s/%s" % (tag, a["jv"]["obj"][3:])
                if hid in jv_lines:
                    rec["a"] = {"d": [["jv", {"obj": hid}], ["keep", a["keep"]]]}
                    jv_lines[hid].append(rec)
    head, jv_out = [], []
    for src in sorted(readings):
        got = 0
        for _c, hid in sorted(readings[src]):
            if got >= JV_READINGS:
                break
            if not jv_lines.get(hid):
                continue
            head.append(hands[hid])
            jv_out.extend(jv_lines[hid])
            got += 1
    lines = []
    for key in sorted(heaps):
        lst = sorted(heaps[key], key=lambda x: (-x[0], x[1].k))
        lines.extend(rec for _c, _k, rec in lst)
    with gzip.GzipFile(out, "wb", mtime=0, compresslevel=9) as fh:
        for rec in head + jv_out + lines:
            fh.write((_dumps(rec) + "\n").encode())
    per = {}
    for rec in head + jv_out + lines:
        per[rec["fn"]] = per.get(rec["fn"], 0) + 1
    print(_dumps({"lines": len(head) + len(jv_out) + len(lines), "per": per, "bytes": os.path.getsize(out)}))


class _Rev:
    """ヒープの同点を字句の逆順で（大きい字句から捨てる）。"""

    def __init__(self, k):
        self.k = k

    def __lt__(self, o):
        return self.k > o.k

    def __gt__(self, o):
        return self.k < o.k

    def __eq__(self, o):
        return self.k == o.k


if __name__ == "__main__":
    main()
