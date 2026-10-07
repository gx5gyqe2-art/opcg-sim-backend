"""**移植の段 5／6**: 局の駆動と行の関数の呼び出しの記録（`theory_rows_rs`・`OPCG_THEORY_ROWS_CAPTURE`）→ `cargo test` の golden（gzip）。

    python tests/scripts/theory_rows_golden.py <記録の根> rust/opcg_engine/tests/fixtures/theory_rows_golden.jsonl.gz

`<記録の根>/<組>/<源>/<器>.rows.jsonl.gz`（組＝`main`〔既定の枝〕・`cand_opp`・`cand_mis`・`cand_misp`・`cand_joint`〔残す候補・`rec`〕、
源＝`rec`・`w41`・`w39`）。1 ファイル＝1 プロセス＝**1 列**（`<組>/<源>/<器>`）。

**間引きの規則**（固定）: 列は**丸ごと**入れる（覚え書きが局をまたいで育つ＝列の途中の 1 局だけを抜くと値が変わりうる・
1 局目から切った前半なら正しい）。列の数が予算（`BUDGET` バイト・gzip 後の見積もり）を超えるなら、組 `main` の列を全部、
残りは列の名前の CRC32 の小さい順に予算まで。局の枠（列の型とバイト）は字句ごとに 1 度だけ `{"frame_def": <番号>, "frame": …}` で
書き、呼び出しの行は `"fref": <番号>` で指す（同じ局を何本もの器が読むので）。移植の間だけの道具（段 7 で消す）。
"""
import gzip
import json
import os
import sys
import zlib

BUDGET = 9_000_000


def _dumps(v):
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def seqs_of(root):
    out = []
    for grp in sorted(os.listdir(root)):
        gd = os.path.join(root, grp)
        if not os.path.isdir(gd):
            continue
        for src in sorted(os.listdir(gd)):
            sd = os.path.join(gd, src)
            if not os.path.isdir(sd):
                continue
            for f in sorted(os.listdir(sd)):
                if f.endswith(".rows.jsonl.gz"):
                    out.append(("%s/%s/%s" % (grp, src, f[: -len(".rows.jsonl.gz")]), os.path.join(sd, f)))
    return out


def build(root, dst):
    seqs = seqs_of(root)
    if not seqs:
        raise SystemExit("記録が無い: %s" % root)
    frames = {}
    blocks = []
    for name, path in seqs:
        lines = []
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    lines.append(json.loads(ln))
        body = []
        for rec in lines:
            out = {"seq": name, "tool": rec["tool"], "game": rec["game"]}
            if "frame" in rec:
                key = _dumps(rec["frame"])
                if key not in frames:
                    frames[key] = len(frames)
                out["fref"] = frames[key]
            out["payload"] = rec["payload"]
            out["result"] = rec["result"]
            body.append(_dumps(out))
        txt = "\n".join(body) + "\n"
        blocks.append((name, txt, len(zlib.compress(txt.encode("utf-8"), 6))))
    # 予算: `main` を全部 → 残りは CRC32 の順
    main = [b for b in blocks if b[0].startswith("main/")]
    rest = sorted((b for b in blocks if not b[0].startswith("main/")), key=lambda b: zlib.crc32(b[0].encode("utf-8")))
    chosen, used = [], 0
    fsz = sum(len(zlib.compress(k.encode("utf-8"), 6)) for k in frames)
    used = fsz
    for b in main + rest:
        if b in rest and used + b[2] > BUDGET:
            continue
        chosen.append(b)
        used += b[2]
    need = set()
    for _n, txt, _s in chosen:
        for ln in txt.splitlines():
            j = json.loads(ln)
            if "fref" in j:
                need.add(j["fref"])
    with gzip.open(dst, "wt", encoding="utf-8", compresslevel=9) as fh:
        for key, i in frames.items():
            if i in need:
                fh.write(_dumps({"frame_def": i, "frame": json.loads(key)}) + "\n")
        for _n, txt, _s in sorted(chosen, key=lambda b: b[0]):
            fh.write(txt)
    print("列 %d／%d・局の枠 %d・%s %d バイト" % (len(chosen), len(blocks), len(need), dst, os.path.getsize(dst)))
    for n, _t, s in sorted(blocks):
        print("  %s %s %d" % ("+" if any(c[0] == n for c in chosen) else "-", n, s))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    build(sys.argv[1], sys.argv[2])
