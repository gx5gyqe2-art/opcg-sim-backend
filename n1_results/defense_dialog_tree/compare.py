"""入れない形（base_<m>＝既存の監査を分かれた点から先で絞った）と入れた形（partial_<m>）の守りの監査の数を並べる（§7.85・1 回きり）。"""
import json, os
D = os.path.dirname(os.path.abspath(__file__))
out = {}
for m in ("life1", "all"):
    row = {}
    for tag in ("base", "partial", "partial_r2"):
        sub = f"partial_{m}_r2" if tag == "partial_r2" else f"{tag}_{m}"
        p = os.path.join(D, sub, "summary.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        v = s["verify"]["resp"]
        r = s["resp"]
        row[tag] = {"games": s["games"], "void": s["void_games"], "points": r["points"],
                    "threat": r["threat"]["k"], "threat_otsu": r["threat_otsu"]["k"],
                    "miss_ko_raw": v["miss_ko_raw"], "miss_otsu_raw": v["miss_otsu_raw"],
                    "miss_ko_full": v["miss_ko_full"], "miss_otsu_full": v["miss_otsu_full"],
                    "playout_kept": (v["playout"] or {}).get("kept"), "playout_kept_otsu": (v["playout"] or {}).get("kept_otsu"),
                    "types_otsu": v.get("miss_otsu_full_type")}
        # 確かめ後の甲の点の一覧（局・手・種類・打ち比べ）
        vf = [json.loads(l) for l in open(os.path.join(D, sub, "verify.jsonl"))]
        row[tag]["kept_points"] = sorted((x["seed"], x["step"]) for x in vf if (x.get("playout") or {}).get("kept"))
        row[tag]["miss_full_points"] = sorted((x["seed"], x["step"]) for x in vf if x.get("miss_ko_full"))
    out[m] = row
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(os.path.join(D, "compare.json"), "w"), ensure_ascii=False, indent=1)
