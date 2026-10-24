"""a1 帯の再生 golden の差分を (A) と (B) と「ネットの差し替え」に分ける（1 回きり・WP rs-overflow-trigger）。

入力（golden_play.py の出力・a1 帯 50 局・読み 32 回・出荷既定のネット＝r22）:
  golden_old.json        修正前の wheel（本線 80f8f9c）
  golden_new_legacy.json 修正後の wheel ＋ confirm_trigger_legacy（＝(A) だけ）
  golden_new.json        修正後の wheel（(A)＋(B)）
HEAD の golden（git show）は r13d の時代に作った＝修正前の wheel で打ち直しても変わる（ネットの差し替え・r22 採用時に作り直していない）。
"""
import json, subprocess, sys
D = "n1_results/overflow_trigger/"
old = json.load(open(D + "golden_old.json"))
leg = json.load(open(D + "golden_new_legacy.json"))
new = json.load(open(D + "golden_new.json"))
key = lambda m: json.dumps(m, sort_keys=True, ensure_ascii=False)

def div(a, b):
    ka, kb = [key(m) for m in a["moves"]], [key(m) for m in b["moves"]]
    if ka == kb:
        return None
    return next((i for i in range(min(len(ka), len(kb))) if ka[i] != kb[i]), min(len(ka), len(kb)))

rows = []
cnt = {"head_vs_old": 0, "A": 0, "B": 0, "new_vs_committed": 0}
for s in sorted(old, key=int):
    g = json.loads(subprocess.check_output(["git", "show", f"HEAD:tests/fixtures/rs_goldens/replay/a1_{s}.json"]))
    head = {"moves": [st["move"] for st in g["input"]["steps"]]}
    com = json.load(open(f"tests/fixtures/rs_goldens/replay/a1_{s}.json"))
    commit = {"moves": [st["move"] for st in com["input"]["steps"]]}
    r = {"seed": int(s)}
    r["head_vs_old"] = div(head, old[s])
    r["A"] = div(old[s], leg[s])            # 修正前 → (A) だけ
    r["B"] = div(leg[s], new[s])            # (A) だけ → (A)＋(B)
    r["new_vs_committed"] = div(new[s], commit)
    for k in cnt:
        cnt[k] += r[k] is not None
    if r["A"] is not None:
        i = r["A"]
        r["A_at"] = {"act_old": old[s]["acts"][i] if i < len(old[s]["acts"]) else None, "kind_old": old[s]["kinds"][i] if i < len(old[s]["kinds"]) else None,
                     "old": old[s]["moves"][i] if i < len(old[s]["moves"]) else None, "new": leg[s]["moves"][i] if i < len(leg[s]["moves"]) else None}
    if r["B"] is not None:
        i = r["B"]
        r["B_at"] = {"act": leg[s]["acts"][i] if i < len(leg[s]["acts"]) else None, "kind": new[s]["kinds"][i] if i < len(new[s]["kinds"]) else None,
                     "legacy": leg[s]["moves"][i] if i < len(leg[s]["moves"]) else None, "new": new[s]["moves"][i] if i < len(new[s]["moves"]) else None}
    r["winner"] = {"old": old[s]["winner"], "A": leg[s]["winner"], "AB": new[s]["winner"]}
    rows.append(r)
print(json.dumps(cnt))
for r in rows:
    if r["A"] is not None or r["B"] is not None:
        print(json.dumps(r, ensure_ascii=False)[:700])
json.dump({"counts": cnt, "rows": rows}, open(D + "golden_diff.json", "w"), ensure_ascii=False, indent=1)
