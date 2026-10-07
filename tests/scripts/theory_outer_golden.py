"""**移植の段 4**: 守る側の外側と耐久の呼び出しの記録（`theory_outer_rs`・`OPCG_THEORY_CORE_CAPTURE`）→ `cargo test` の golden（gzip）。

    python tests/scripts/theory_outer_golden.py <記録のディレクトリ> rust/opcg_engine/tests/fixtures/theory_outer_golden.jsonl.gz

`<記録のディレクトリ>/<源>/<器>/<名前>.jsonl`（源＝`rec`・`w41`・`w39`）。

**間引きの規則**（固定・段 3 の `theory_core_golden.py` と同じ型）: 入力の字句（`fn`・`a`・`g`・`pre`）が同じ行は源をまたいで 1 つ。
`(名前, 源)` ごとに入力の字句の CRC32 が小さい順に `N` 行（`N` は名前ごと・`CAP`）。**曲線**（`cp_curve.jsonl`＝`cp.curve_of_row` と
その曲線の `cv.*`）は曲線ごとにまとめて選ぶ: 1 つの曲線の中の `JointValuer` の覚え書きは残った札だけが鍵＝呼び出しの履歴に依る（E52）
ので、選んだ曲線の行は記録の順に**全部**入れる（源ごとに曲線の行の字句の CRC32 が小さい順に `CURVES` 個）。曲線の番号は
`cv:<源>/<器>/<番号>` に付け替える。曲線 → 残り（名前と CRC32 の順）の順に書く。移植の間だけの道具（段 7 で消す）。
"""
import gzip
import heapq
import json
import os
import sys
import zlib

CAP = {"cb.rule_don_solve": 40, "cb.threshold_parts_side": 60, "cb.rule_don_plan_for": 60, "cb.attacker_ctx": 60,
       "hp.search_context": 40, "tb.guard_hand_reading": 40}
DEFAULT_CAP = 100
CURVES = 6


def _dumps(v):
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def _key(rec):
    return _dumps([rec["fn"], rec["a"], rec["g"], rec.get("pre", [])])


class _Rev(str):
    def __lt__(self, o):
        return str.__gt__(self, o)


def _dirs(root):
    for src in sorted(os.listdir(root)):
        sd = os.path.join(root, src)
        if not os.path.isdir(sd):
            continue
        for tool in sorted(os.listdir(sd)):
            td = os.path.join(sd, tool)
            if os.path.isdir(td):
                yield src, "%s/%s" % (src, tool), td


def _curves_of(path, tag):
    """`cp_curve.jsonl` → [(曲線の行, [cv の行…])]（記録の順・番号を付け替え）。"""
    out, by = [], {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["fn"] == "cp.curve_of_row":
                r = rec["r"]
                if r is None:
                    continue
                d = {k: v for k, v in r["d"]}
                cid = "cv:%s/%s" % (tag, d["id"]["obj"][3:])
                rec["r"] = {"d": [["id", {"obj": cid}], ["s", d["s"]]]}
                ent = (rec, [])
                by[d["id"]["obj"]] = (cid, ent)
                out.append(ent)
            else:
                a = {k: v for k, v in rec["a"]["d"]}
                old = a["cv"]["obj"]
                if old not in by:
                    continue
                cid, ent = by[old]
                rec["a"] = {"d": [[k, ({"obj": cid} if k == "cv" else v)] for k, v in rec["a"]["d"]]}
                ent[1].append(rec)
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root, out = argv[0], argv[1]
    heaps = {}
    curves = {}       # src -> 最大ヒープ [(-crc, 字句, 曲線)]
    seen = set()
    n_in = 0
    for src, tag, td in _dirs(root):
        for fn in sorted(f for f in os.listdir(td) if f.endswith(".jsonl")):
            path = os.path.join(td, fn)
            if fn == "cp_curve.jsonl":
                for ent in _curves_of(path, tag):
                    k = _dumps(ent[0]["a"])
                    crc = zlib.crc32(k.encode())
                    hp = curves.setdefault(src, [])
                    item = (-crc, _Rev(k), ent)
                    if len(hp) < CURVES:
                        heapq.heappush(hp, item)
                    elif (-crc, _Rev(k)) > (hp[0][0], hp[0][1]):
                        heapq.heapreplace(hp, item)
                continue
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    n_in += 1
                    rec = json.loads(line)
                    k = _key(rec)
                    if k in seen:
                        continue
                    seen.add(k)
                    crc = zlib.crc32(k.encode())
                    hp = heaps.setdefault((rec["fn"], src), [])
                    n = CAP.get(rec["fn"], DEFAULT_CAP)
                    item = (-crc, _Rev(k), rec)
                    if len(hp) < n:
                        heapq.heappush(hp, item)
                    elif (-crc, _Rev(k)) > (hp[0][0], hp[0][1]):
                        heapq.heapreplace(hp, item)
    lines = []
    for src in sorted(curves):
        for _c, _k, (crec, cvs) in sorted(curves[src], key=lambda x: (-x[0], str(x[1]))):
            lines.append(_dumps(crec))
            lines.extend(_dumps(r) for r in cvs)
    for (fn, src) in sorted(heaps):
        for _c, _k, rec in sorted(heaps[(fn, src)], key=lambda x: (-x[0], str(x[1]))):
            lines.append(_dumps(rec))
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=9) as fh:
        for ln in lines:
            fh.write(ln + "\n")
    print("golden: %d 行（記録 %d 行から）→ %s (%d bytes)" % (len(lines), n_in, out, os.path.getsize(out)))


if __name__ == "__main__":
    main()
