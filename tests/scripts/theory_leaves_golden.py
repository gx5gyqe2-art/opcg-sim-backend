"""**移植の段 2**: 葉の呼び出しの記録（`theory_capture`）→ `cargo test` が読む golden（`rust/opcg_engine/tests/fixtures/`）。

    python tests/scripts/theory_leaves_golden.py <記録のディレクトリ>     # <dir>/<src>/<器>/<葉>.jsonl

* `theory_leaves_golden.jsonl.gz` — 1 行 = 葉の 1 呼び出し（`src` を足しただけ・中身は記録のまま）。
  **間引きの規則（決まった規則・記録の順に依らない）**: 同じ入力（`fn`・`a`・`g`・`s` の字句）は記録の源（`rec`・実・合成）を
  またいで 1 つにし、`(fn, src)` ごとに入力の字句の CRC32 が小さい順に `CAP` 行まで残す（numpy の配列を引数に持つ葉は
  `CAP_ND`）。全部の記録は `OPCG_THEORY_CAPTURE_REPLAY` で手で解き直せる（`tests_leaves.rs`）。
* `theory_cards.json.gz` — カード表と語彙（`theory_rs.card_table_json`・Rust の葉が引く表）。

golden は「その時点の Python の出力」であって、正しさの独立した証拠ではない（CLAUDE.md の golden の作法と同じ）。
移植の間だけの道具（段 7 で Python を消す直前に、最終の記録で取り直す）。
"""
import gzip
import json
import os
import sys
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(_HERE))
import _bootstrap  # noqa: F401,E402

OUT_DIR = os.path.join(_ROOT, "rust", "opcg_engine", "tests", "fixtures")
CAP = 240
CAP_ND = 60
SRC_ORDER = ("rec", "w41", "w39")


def build(root):
    seen = set()
    per = {}
    for src in sorted(os.listdir(root), key=lambda s: (SRC_ORDER.index(s) if s in SRC_ORDER else 9, s)):
        for tool in sorted(os.listdir(os.path.join(root, src))):
            d = os.path.join(root, src, tool)
            for f in sorted(os.listdir(d)):
                if not f.endswith(".jsonl"):
                    continue
                with open(os.path.join(d, f), encoding="utf-8") as fh:
                    for line in fh:
                        rec = json.loads(line)
                        key = json.dumps([rec["fn"], rec["a"], rec["g"], rec["s"]], separators=(",", ":"), ensure_ascii=False)
                        if key in seen:
                            continue
                        seen.add(key)
                        per.setdefault((rec["fn"], src), []).append((zlib.crc32(key.encode("utf-8")), key, rec))
    out = []
    counts = {}
    for (fn, src), lst in sorted(per.items()):
        cap = CAP_ND if any('"nd"' in k for _c, k, _r in lst[:5]) else CAP
        lst.sort(key=lambda x: (x[0], x[1]))
        keep = lst[:cap]
        counts[(fn, src)] = (len(lst), len(keep))
        for _c, _k, rec in keep:
            out.append({"src": src, **rec})
    return out, counts


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit(__doc__)
    out, counts = build(argv[0])
    p = os.path.join(OUT_DIR, "theory_leaves_golden.jsonl.gz")
    with gzip.GzipFile(p, "wb", mtime=0) as fh:
        for rec in out:
            fh.write((json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    import theory_rs as RS
    pc = os.path.join(OUT_DIR, "theory_cards.json.gz")
    with gzip.GzipFile(pc, "wb", mtime=0) as fh:
        fh.write(RS.card_table_json().encode("utf-8"))
    tot_all = sum(a for a, _b in counts.values())
    for (fn, src), (a, b) in sorted(counts.items()):
        print("%-28s %-4s distinct=%6d kept=%4d" % (fn, src, a, b))
    print("lines kept=%d distinct=%d  golden=%.2f MB  cards=%.2f MB" % (len(out), tot_all, os.path.getsize(p) / 1e6,
                                                                      os.path.getsize(pc) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
