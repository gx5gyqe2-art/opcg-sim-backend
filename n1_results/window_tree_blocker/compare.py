"""守りの再監査の修正前後を並べる（WP rs-window-tree-blocker・1 回きり）。

修正前 = rs-window-read の既定（3013000〜: `n1_results/window_read/defense_new`）と reaudit-wr（3351000〜: `n1_results/reaudit_wr/defense`）。
修正後 = `defense_<B>/merged`（分かれなかった局は修正前の記録・分かれた局は打ち直し）。
確かめ後の見逃し（乙）を 1 点ずつ「ライフ ≤ 1 のカウンターの窓」かどうかで分ける。
"""
import json
import sys

sys.path.insert(0, "tests/scripts")
import defense_audit as DA  # noqa: E402

D = "n1_results/window_tree_blocker"
SIDES = {3013000: {"before": "n1_results/window_read/defense_new/points_3013000_300.jsonl.gz",
                   "before_v": "n1_results/window_read/defense_new/verify.jsonl"},
         3351000: {"before": "n1_results/reaudit_wr/defense/points_3351000_300.jsonl",
                   "before_v": "n1_results/reaudit_wr/defense/verify.jsonl"}}


def side(path, vpath):
    games, pts = DA._load(path)
    ver = {(r["kind"], r["seed"], r["seat"], r["step"]): r for r in map(json.loads, open(vpath))}
    resp = [p for p in pts if p["kind"] == "resp"]
    rows = []
    for p in resp:
        if not p["miss_ko"]:
            continue
        v = ver.get(("resp", p["seed"], p["seat"], p["step"]), {})
        rows.append({"seed": p["seed"], "seat": p["seat"], "step": p["step"], "turn": p["turn"],
                     "action": p.get("action"), "my_life": p.get("my_life"), "opp_life": p.get("opp_life"),
                     "otsu": bool(p.get("miss_otsu")), "ko_verified": bool(v.get("miss_ko_full")),
                     "otsu_verified": bool(v.get("miss_otsu_full")), "live_kind": v.get("live_kind"),
                     "cpu_move": next((r["move"] for r in p["rows"] if r.get("cpu")), None),
                     "low_counter": p.get("action") == "SELECT_COUNTER" and p.get("my_life", 9) <= 1,
                     "winner": games[p["seed"]]["winner"]})
    low = [p for p in resp if p.get("action") == "SELECT_COUNTER" and p["my_life"] <= 1]
    return {
        "games": len(games), "void": sum(1 for g in games.values() if g.get("void")),
        "resp_points": len(resp),
        "threat_ko": sum(1 for p in resp if p["threat"]), "threat_otsu": sum(1 for p in resp if p.get("threat_otsu")),
        "miss_ko": sum(1 for p in resp if p["miss_ko"]), "miss_otsu": sum(1 for p in resp if p.get("miss_otsu")),
        "miss_ko_verified": sum(r["ko_verified"] for r in rows),
        "miss_otsu_verified": sum(r["otsu_verified"] for r in rows),
        "low_counter_points": len(low), "low_counter_threat_ko": sum(1 for p in low if p["threat"]),
        "low_counter_miss_ko_verified": sum(r["ko_verified"] for r in rows if r["low_counter"]),
        "low_counter_miss_otsu_verified": sum(r["otsu_verified"] for r in rows if r["low_counter"]),
        "verified_rows": [r for r in rows if r["ko_verified"] or r["otsu_verified"]],
        "p1_wins": sum(1 for g in games.values() if g["winner"] == "p1"),
    }


def main():
    out = {}
    for b, s in SIDES.items():
        div = {d["seed"] for d in map(json.loads, open(f"{D}/diverge_{b}.jsonl")) if d["diverged"]}
        before = side(s["before"], s["before_v"])
        after = side(f"{D}/defense_{b}/merged/points_{b}_300.jsonl.gz", f"{D}/defense_{b}/merged/verify.jsonl")
        key = lambda r: (r["seed"], r["seat"], r["step"])
        bk = {key(r) for r in before["verified_rows"] if r["ko_verified"]}
        ak = {key(r) for r in after["verified_rows"] if r["ko_verified"]}
        out[b] = {"diverged_games": len(div), "before": before, "after": after,
                  "ko_verified_only_before": sorted(bk - ak), "ko_verified_only_after": sorted(ak - bk)}
        print(b, "diverged", len(div))
        for k in ("void", "resp_points", "threat_ko", "threat_otsu", "miss_ko", "miss_otsu", "miss_ko_verified",
                  "miss_otsu_verified", "low_counter_points", "low_counter_threat_ko",
                  "low_counter_miss_ko_verified", "low_counter_miss_otsu_verified", "p1_wins"):
            print(f"  {k:32s} {before[k]:6} {after[k]:6}")
        for t in ("before", "after"):
            for r in out[b][t]["verified_rows"]:
                print("   ", t, r["seed"], r["step"], r["action"], "life", r["my_life"], "ko" if r["ko_verified"] else "",
                      "otsu" if r["otsu_verified"] else "", r["live_kind"], r["cpu_move"], "div" if r["seed"] in div else "")
    json.dump(out, open(f"{D}/compare.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
