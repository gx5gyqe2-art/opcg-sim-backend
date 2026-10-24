"""再生 golden の作り直しの差分を 1 局ずつ読む（1 回きり・WP rs-unpayable-activate）。
HEAD の golden と作り直した golden で、最初に違った段（合法手／盤面／イベント）と、その段の合法手の差・打った手を出す。
    python n1_results/unpayable_activate/golden_diff.py > n1_results/unpayable_activate/golden_diff.txt"""
import glob, json, os, subprocess

D = "tests/fixtures/rs_goldens/replay"
changed = subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", D], text=True).split()
out = []
for path in sorted(changed):
    new = json.load(open(path))
    old = json.loads(subprocess.check_output(["git", "show", f"HEAD:{path}"]))
    first = None
    for kind in ("legal", "states", "events"):
        a, b = old["hashes"][kind], new["hashes"][kind]
        for i in range(min(len(a), len(b))):
            if a[i] != b[i]:
                if first is None or i < first[0]:
                    first = (i, kind)
                break
    i, kind = first
    so, sn = old["input"]["steps"], new["input"]["steps"]
    def mv(m):
        p = m.get("payload") or {}
        return (m.get("action_type"), p.get("uuid") or m.get("card_uuid"))
    # 段 i の「打つ前」の合法手は記録に無い（記録は打った後の合法手）＝打った手そのものと、直前の段の後の合法手で読む。
    rec = {"file": os.path.basename(path), "first_diff_step": i, "first_kind": kind,
           "old_summary": old["summary"], "new_summary": new["summary"],
           "old_move": mv(so[i]["move"]) if i < len(so) else None, "new_move": mv(sn[i]["move"]) if i < len(sn) else None,
           "prev_moves_same": [mv(x["move"]) for x in so[max(0, i - 2):i]] == [mv(x["move"]) for x in sn[max(0, i - 2):i]],
           "old_prev_move": mv(so[i - 1]["move"]) if i > 0 else None}
    # uuid → card_id（初期の盤面から）
    names = {}
    for P in old["input"]["setup"]["hidden"]["players"].values():
        for z, cs in P.items():
            for c in (cs if isinstance(cs, list) else [cs] if isinstance(cs, dict) else []):
                if isinstance(c, dict) and "card_id" in c:
                    names[c["uuid"]] = c["card_id"]
    rec["old_move_card"] = names.get((rec["old_move"] or [None, None])[1])
    rec["new_move_card"] = names.get((rec["new_move"] or [None, None])[1])
    rec["old_prev_move_card"] = names.get((rec["old_prev_move"] or [None, None])[1])
    # その段の盤面の差の中身（打った後の要求）。
    rec["old_pending_after"] = (so[i].get("move") or {}).get("action_type"), None
    out.append(rec)
    print(json.dumps(rec, ensure_ascii=False))
