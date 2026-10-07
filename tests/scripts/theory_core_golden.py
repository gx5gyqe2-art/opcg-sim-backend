"""**移植の段 3**: 核の呼び出しの記録（`theory_core_rs`・`OPCG_THEORY_CORE_CAPTURE`）→ `cargo test` の golden（gzip）。

    python tests/scripts/theory_core_golden.py <記録のディレクトリ> rust/opcg_engine/tests/fixtures/theory_core_golden.jsonl.gz

**間引きの規則**（固定）: 入力の字句（`fn`・`a`・`g`・`pre`）が同じ行は源をまたいで 1 つ。`(名前, 源)` ごとに入力の字句の CRC32 が
小さい順に `N` 行（`N` は名前ごと・`CAP`）。`jv.value` の行が使う手札の行（`tb.joint_valuer`）は必ず入れ、札の読みの番号を
`jv:<源>/<器>/<番号>` に付け替える（ファイルの中で一意）。手札の行を先に・残りは名前と CRC32 の順に書く。
移植の間だけの道具（段 7 で消す）。
"""
import gzip
import json
import os
import sys
import zlib

CAP = {"jv.value": 40, "tb.joint_valuer": 0, "ev.card_value": 60, "hp.apply_inflow": 40, "hs.use_value": 80,
       "hs.free_value": 80, "sp.search_value": 30, "sp.card_gain": 60, "ev.continuous_self_mods": 60}
DEFAULT_CAP = 120


def _key(rec):
    return json.dumps([rec["fn"], rec["a"], rec["g"], rec.get("pre", [])], ensure_ascii=False, separators=(",", ":"))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root, out = argv[0], argv[1]
    picked = {}       # (fn, src) -> [(crc, key, rec, tooltag)]
    hands = {}        # "jv:<src>/<tool>/<id>" -> rec
    seen = set()
    for src in sorted(os.listdir(root)):
        sd = os.path.join(root, src)
        if not os.path.isdir(sd):
            continue
        for tool in sorted(os.listdir(sd)):
            td = os.path.join(sd, tool)
            if not os.path.isdir(td):
                continue
            tag = "%s/%s" % (src, tool)
            for fn in sorted(os.listdir(td)):
                if not fn.endswith(".jsonl"):
                    continue
                with open(os.path.join(td, fn), encoding="utf-8") as fh:
                    for line in fh:
                        rec = json.loads(line)
                        if rec["fn"] == "tb.joint_valuer":
                            hid = "jv:%s/%s" % (tag, rec["r"]["obj"][3:])
                            rec["r"] = {"obj": hid}
                            hands[hid] = rec
                            continue
                        if rec["fn"] == "jv.value":
                            a = dict(rec["a"]["d"])
                            hid = "jv:%s/%s" % (tag, a["jv"]["obj"][3:])
                            rec["a"] = {"d": [["jv", {"obj": hid}], ["keep", a["keep"]]]}
                            k = _key(dict(rec, a={"d": [["hand", hands[hid]["a"]]], ["keep", a["keep"]]]}))
                        else:
                            k = _key(rec)
                        if k in seen:
                            continue
                        seen.add(k)
                        picked.setdefault((rec["fn"], src), []).append((zlib.crc32(k.encode()), k, rec))
    lines = []
    need = set()
    for (name, src), lst in sorted(picked.items()):
        lst.sort(key=lambda x: (x[0], x[1]))
        for _c, _k, rec in lst[:CAP.get(name, DEFAULT_CAP)]:
            lines.append(rec)
            if name == "jv.value":
                need.add(dict(rec["a"]["d"])["jv"]["obj"])
    head = [hands[h] for h in sorted(need)]
    with gzip.GzipFile(out, "wb", mtime=0, compresslevel=9) as fh:
        for rec in head + lines:
            fh.write((json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
    per = {}
    for rec in head + lines:
        per[rec["fn"]] = per.get(rec["fn"], 0) + 1
    print(json.dumps({"lines": len(head) + len(lines), "per": per, "bytes": os.path.getsize(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
