"""**移植の段 5／6**: 局の駆動と行の関数の呼び出しの記録（`theory_rows_rs`・`OPCG_THEORY_ROWS_CAPTURE`）→ `cargo test` の golden（gzip）。

    python tests/scripts/theory_rows_golden.py <記録の根> rust/opcg_engine/tests/fixtures/theory_rows_golden.jsonl.gz

`<記録の根>/<組>/<源>/<器>.rows.jsonl.gz`（組＝`main`〔既定の枝〕・`cand_opp`・`cand_mis`・`cand_misp`・`cand_joint`〔残す候補・`rec`〕、
源＝`rec`・`w41`・`w39`）。1 ファイル＝1 プロセス＝**1 列**（`<組>/<源>/<器>`）。

**間引きの規則**（固定）: 覚え書きは局をまたいで育つので、列は**頭から**切る（局の駆動の行は先頭の `N` 本・`N` は下の表・
表に無い列は入れない）。行の関数（`fn:*`＝`probs_of`・`clock_scale`）は核の状態を読みも書きもしない＝列のどこに在っても全部入れる。
`cargo test` は debug で回るので、重い局の駆動（交点の橋・`theory_bridge`・`relative_ledger`）は `main/rec` の 1 局目だけ、
軽いもの（`kappa_vector`・`transition_ledger`・`price_realised`・決着の旗〔`win_calib` の列の `lethal_rule`〕）は全部。
局の枠（列の型とバイト）は字句ごとに 1 度だけ `{"frame_def": <番号>, "frame": …}` で書き、呼び出しの行は `"fref": <番号>` で指す
（同じ局を何本もの器が読むので）。全部の記録の再生は `OPCG_THEORY_ROWS_REPLAY`（手で回す）。移植の間だけの道具（段 7 で消す）。
"""
import gzip
import json
import os
import sys

ALL = 10 ** 9
CHEAP = {"kappa_vector": ALL, "transition_ledger": ALL, "price_realised": ALL}
#: `(組, 源)` → `{器: 局の駆動の行の本数}`（`win_calib`・`pre_settle_asymmetry` の列は決着の旗の行が先に並ぶ）
PREFIX = {
    ("main", "rec"): dict(CHEAP, theory_bridge=1, relative_ledger=1, crossing_bridge=1, win_calib=2, pre_settle_asymmetry=2),
    ("main", "w41"): dict(CHEAP, win_calib=5),
    ("main", "w39"): dict(CHEAP, win_calib=5),
    ("cand_opp", "rec"): CHEAP, ("cand_mis", "rec"): CHEAP, ("cand_misp", "rec"): CHEAP, ("cand_joint", "rec"): CHEAP,
}


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
                    out.append((grp, src, f[: -len(".rows.jsonl.gz")], os.path.join(sd, f)))
    return out


def build(root, dst):
    seqs = seqs_of(root)
    if not seqs:
        raise SystemExit("記録が無い: %s" % root)
    frames = {}
    out_lines = []
    report = []
    for grp, src, tool, path in seqs:
        n_max = PREFIX.get((grp, src), {}).get(tool, 0)
        name = "%s/%s/%s" % (grp, src, tool)
        kept = drv = 0
        if n_max:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for ln in fh:
                    if not ln.strip():
                        continue
                    rec = json.loads(ln)
                    is_fn = "frame" not in rec
                    if not is_fn:
                        if drv >= n_max:
                            continue
                        drv += 1
                    o = {"seq": name, "tool": rec["tool"], "game": rec["game"]}
                    if not is_fn:
                        key = _dumps(rec["frame"])
                        if key not in frames:
                            frames[key] = len(frames)
                        o["fref"] = frames[key]
                    o["payload"] = rec["payload"]
                    o["result"] = rec["result"]
                    out_lines.append(_dumps(o))
                    kept += 1
        report.append((name, kept))
    with gzip.open(dst, "wt", encoding="utf-8", compresslevel=9) as fh:
        for key, i in frames.items():
            fh.write(_dumps({"frame_def": i, "frame": json.loads(key)}) + "\n")
        for ln in out_lines:
            fh.write(ln + "\n")
    print("列 %d／%d・行 %d・局の枠 %d・%s %d バイト" % (sum(1 for _n, k in report if k), len(report), len(out_lines), len(frames),
                                                  dst, os.path.getsize(dst)))
    for n, k in report:
        print("  %s %s %d" % ("+" if k else "-", n, k))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    build(sys.argv[1], sys.argv[2])
