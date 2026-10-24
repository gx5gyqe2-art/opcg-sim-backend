"""a1 帯の再生 golden の差分を読む（1 回きり・WP rs-window-tree-blocker）。

  golden_new_legacy.json  この WP の wheel ＋ window_tree_counter_legacy（＝修正前の打ち方）→ HEAD の golden と一致するはず
  golden_new.json         この WP の wheel（既定）→ 作り直した golden と一致するはず（golden の作り直しの後に --committed で見る）
"""
import json, subprocess, sys
D = "n1_results/window_tree_blocker/"
new = json.load(open(D + "golden_new.json"))
leg = json.load(open(D + "golden_new_legacy.json"))
key = lambda m: json.dumps(m, sort_keys=True, ensure_ascii=False)


def moves_of(g):
    return [st["move"] for st in g["input"]["steps"]]


def div(a, b):
    ka, kb = [key(m) for m in a], [key(m) for m in b]
    if ka == kb:
        return None
    return next((i for i in range(min(len(ka), len(kb))) if ka[i] != kb[i]), min(len(ka), len(kb)))


ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
rows, cnt = [], {"legacy_vs_head": 0, "new_vs_legacy": 0, "new_vs_committed": 0}
for s in sorted(new, key=int):
    head = moves_of(json.loads(subprocess.check_output(["git", "show", f"{ref}:tests/fixtures/rs_goldens/replay/a1_{s}.json"])))
    com = moves_of(json.load(open(f"tests/fixtures/rs_goldens/replay/a1_{s}.json")))
    r = {"seed": int(s), "legacy_vs_head": div(leg[s]["moves"], head), "new_vs_legacy": div(leg[s]["moves"], new[s]["moves"]),
         "new_vs_committed": div(new[s]["moves"], com)}
    for k in cnt:
        cnt[k] += r[k] is not None
    i = r["new_vs_legacy"]
    if i is not None:
        r["at"] = {"act": leg[s]["acts"][i] if i < len(leg[s]["acts"]) else None,
                   "kind": new[s]["kinds"][i] if i < len(new[s]["kinds"]) else None,
                   "legacy": leg[s]["moves"][i] if i < len(leg[s]["moves"]) else None,
                   "new": new[s]["moves"][i] if i < len(new[s]["moves"]) else None}
        r["winner"] = {"legacy": leg[s]["winner"], "new": new[s]["winner"]}
    rows.append(r)
print(json.dumps(cnt))
for r in rows:
    if r["new_vs_legacy"] is not None:
        print(json.dumps(r, ensure_ascii=False)[:400])
json.dump({"ref": ref, "counts": cnt, "rows": rows}, open(D + "golden_diff.json", "w"), ensure_ascii=False, indent=1)
