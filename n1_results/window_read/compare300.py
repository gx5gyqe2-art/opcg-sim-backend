"""前後の 300 局の守りの監査を並べる（WP rs-window-read・1 回きり）。"""
import json
import sys

sys.path.insert(0, "/home/user/opcg-sim-backend/tests/scripts")
import defense_audit as DA  # noqa: E402

B = "/home/user/opcg-sim-backend/n1_results/"


def load(d):
    games, pts = DA._load(B + d + "/points_3013000_300.jsonl" + (".gz" if d.startswith("commit_hidden") else ""))
    ver = {}
    for line in open(B + d + "/verify.jsonl"):
        r = json.loads(line)
        ver[(r["kind"], r["seed"], r["seat"], r["step"])] = r
    return games, pts, ver


def side(d):
    games, pts, ver = load(d)
    resp = [p for p in pts if p["kind"] == "resp"]
    miss_ot = [p for p in resp if p.get("miss_otsu")]
    rows = []
    for p in miss_ot:
        v = ver.get(("resp", p["seed"], p["seat"], p["step"]), {})
        rows.append({"seed": p["seed"], "seat": p["seat"], "step": p["step"], "turn": p["turn"],
                     "action": p.get("action"), "my_life": p.get("my_life"),
                     "verified": bool(v.get("miss_otsu_full")), "live_kind": v.get("live_kind"),
                     "fresh_same_as_live": v.get("fresh_same_as_live"), "fresh_safe": v.get("fresh_safe"),
                     "winner": games[p["seed"]]["winner"], "lost": games[p["seed"]]["winner"] != p["seat"]})
    ver_rows = [r for r in rows if r["verified"]]
    return {
        "games": len(games), "void": sum(1 for g in games.values() if g.get("void")),
        "resp_points": len(resp),
        "threat_ko": sum(1 for p in resp if p["threat"]), "threat_otsu": sum(1 for p in resp if p.get("threat_otsu")),
        "miss_ko": sum(1 for p in resp if p["miss_ko"]), "miss_otsu": len(miss_ot),
        "miss_otsu_verified": len(ver_rows),
        "miss_otsu_verified_by_kind": {k: sum(1 for r in ver_rows if r["live_kind"] == k) for k in ("commit", "window", "main")},
        "commit_rows": [r for r in ver_rows if r["live_kind"] == "commit"],
        "games_with_verified_miss": len({r["seed"] for r in ver_rows}),
        "lost_seat": sum(1 for r in {(r["seed"], r["seat"]): r for r in ver_rows}.values() if r["lost"]),
        "p1_wins": sum(1 for g in games.values() if g["winner"] == "p1"),
        "mean_turns": round(sum(g["turns"] for g in games.values()) / len(games), 2),
    }, games


def main():
    new, gn = side("window_read/defense_new")
    old, go = side("commit_hidden/new300")
    same = [s for s in gn if s in go and (gn[s]["winner"], gn[s]["turns"], gn[s]["points"]) == (go[s]["winner"], go[s]["turns"], go[s]["points"])]
    out = {"legacy": old, "new": new, "same_game_shape": len(same),
           "same_winner": sum(1 for s in gn if s in go and gn[s]["winner"] == go[s]["winner"]),
           "diff_seeds": sorted(s for s in gn if s not in same)}
    json.dump(out, open(B + "window_read/compare300.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
