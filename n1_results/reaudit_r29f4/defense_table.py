"""守りの確かめ後の見逃し（甲）を 1 点ずつ（読み取りだけ）。"""
import json
V = [json.loads(l) for l in open("n1_results/reaudit_r29f4/defense_all/verify.jsonl")]
print(len(V), list(V[0].keys()))
for v in V:
    if not v.get("miss_ko_full"): continue
    pl = v.get("playout") or {}
    r, f = pl.get("real") or {}, pl.get("fair") or {}
    print(v.get("seed"), v.get("step"), v.get("action"), "life", v.get("my_life"), v.get("opp_life"), "otsu", v.get("miss_otsu_full"),
          "| cpu", (v.get("cpu_move") or "")[:60], "-> safe", (v.get("safe_move") or str(v.get("safe_moves"))[:60])[:60],
          "| real cpu/safe", r.get("wa"), r.get("wb"), "d", r.get("diff"), r.get("ci"), "lost_this_turn", r.get("cpu_lost_this_turn"), r.get("safe_lost_this_turn"),
          "| fair", f.get("wa"), f.get("wb"), "| kept", pl.get("kept"), "| live", v.get("live_kind"), json.dumps(v.get("live_window"))[:60])
