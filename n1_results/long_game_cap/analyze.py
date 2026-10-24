"""long_probe の結果をまとめる（1 回きり・WP long-game-cap）。
    python analyze.py PROBE_DIR > summary.txt  （summary.json も PROBE_DIR に書く）"""
import collections
import json
import os
import sys

d = sys.argv[1]
rows = [json.loads(x) for x in open(os.path.join(d, "games.jsonl"))]
rows.sort(key=lambda r: r["seed"])
N = len(rows)
VEGA = "OP07-097"
MOMO = "OP16-084"


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def passive(r):
    """ターン 5 以降で 2 手以下のターン（すぐ終える・攻めない手番の目安）の割合。"""
    tr = r.get("turn_rows") or []
    b = [t["step"] for t in tr] + [r["steps"]]
    per = [(t["turn"], y - x) for t, x, y in zip(tr, b, b[1:])]
    late = [n for t, n in per if t >= 5]
    return (sum(1 for n in late if n <= 2) / len(late)) if late else 0.0


err = [r for r in rows if r.get("error")]
cap = [r for r in rows if r.get("reached_cap")]
fin = [r for r in rows if not r.get("error") and r.get("winner")]
drop = [r for r in rows if r.get("drop400")]
momo = [r for r in rows if any(k.endswith(MOMO) for k in (r.get("noops") or {}))]
noop_any = [r for r in rows if r.get("noops")]
out = {"games": N, "errors": len(err), "reached_cap400": len(cap), "drop400": len(drop),
       "momonosuke_noop_games": [r["seed"] for r in momo],
       "noop_games": {r["seed"]: r["noops"] for r in noop_any}}
print(f"局 {N}・例外 {len(err)}・400 手に届いた {len(cap)}・生成なら捨てた {len(drop)}")
print(f"何も変えない手（NOOP）を打った局 {len(noop_any)}・うちモモの助 OP16-084 {len(momo)}")
for r in noop_any:
    print("  noop", r["seed"], r["leaders"], r["noops"], "steps", r["steps"], "winner", r.get("winner"))

# 長さの分布（決着した局の手数・ターン数）
steps = [r["steps"] for r in fin]
turns = [r["turns"] for r in fin]
print("\n決着した局", len(fin), "手数 p50/p90/p99/最大", q(steps, .5), q(steps, .9), q(steps, .99), max(steps))
print("               ターン p50/p90/p99/最大", q(turns, .5), q(turns, .9), q(turns, .99), max(turns))
hist = collections.Counter(min(s // 100, 16) * 100 for s in [r["steps"] for r in rows])
print("手数の分布（100 手刻み・全局）", dict(sorted(hist.items())))
cut = {}
for c in (300, 400, 500, 600, 800, 1000, 1200, 1600):
    cut[c] = sum(1 for r in rows if r["steps"] >= c and not (r["steps"] == c and r.get("winner")))
print("上限 c で捨てる局（c 手で決着していない）", cut)
tcut = {}
for c in (25, 30, 35, 40, 50, 60):
    tcut[c] = sum(1 for r in rows if r["turns"] > c)
print("ターン数 > c の局", tcut)
out.update(steps_q={"p50": q(steps, .5), "p90": q(steps, .9), "p99": q(steps, .99), "max": max(steps)},
           turns_q={"p50": q(turns, .5), "p90": q(turns, .9), "p99": q(turns, .99), "max": max(turns)},
           steps_hist=hist, drop_at_cap=cut, turns_over=tcut)

# 400 手に届いた局
print("\n400 手に届いた局:")
cap_rows = []
for r in cap:
    tr = r.get("turn_rows") or []
    lifes = [(t["p1_life"], t["p2_life"]) for t in tr]
    lmax = [max(x[i] for x in lifes) for i in (0, 1)] if lifes else None
    ext = (r.get("winner"), r["turns"], r["steps"])
    c = {"seed": r["seed"], "leaders": r["leaders"], "vega": VEGA in r["leaders"],
         "turn_at_400": r.get("turn_at_400"), "ext_winner": r.get("winner"), "ext_turns": r["turns"],
         "ext_steps": r["steps"], "ext_capped": r.get("ext_capped"), "turn_sig_repeat": r.get("turn_sig_repeat"),
         "noops": r.get("noops"), "life_max": lmax, "final": r.get("final"),
         "turn_steps_max": r.get("turn_steps_max"), "passive_frac": round(passive(r), 3),
         "steps_per_turn": round(r["steps"] / max(1, r["turns"]), 1)}
    cap_rows.append(c)
    print(" ", c["seed"], c["leaders"], "vega" if c["vega"] else "    ", "@400 turn", c["turn_at_400"], "→", ext,
          "rep", c["turn_sig_repeat"], "noop", sum((c["noops"] or {}).values()), "life_max", lmax,
          "maxT", c["turn_steps_max"], "passive", c["passive_frac"], "spt", c["steps_per_turn"])
out["cap_games"] = cap_rows

# リーダーごと（ベガパンクの局と、それ以外）
lc = collections.Counter()
for r in cap:
    for ld in set(r["leaders"]):
        lc[ld] += 1
allc = collections.Counter(ld for r in rows for ld in set(r["leaders"]))
print("\n400 手に届いた局のリーダー（届いた局／その リーダーの局）:",
      {k: f"{v}/{allc[k]}" for k, v in lc.most_common()})
vg = [r for r in rows if VEGA in r["leaders"]]
print("ベガパンクの局", len(vg), "400 手に届いた", sum(1 for r in vg if r.get("reached_cap")),
      "手数 p50", q([r["steps"] for r in vg], .5), "ターン p50", q([r["turns"] for r in vg], .5))
pv = [passive(r) for r in fin if not r.get("reached_cap")]
pc = [passive(r) for r in cap]
print("すぐ終えるターンの割合 平均: 届かない局", round(sum(pv) / max(1, len(pv)), 3),
      "・届いた局", round(sum(pc) / max(1, len(pc)), 3))
spt_n = [r["steps"] / max(1, r["turns"]) for r in fin if not r.get("reached_cap")]
print("1 ターンの手数 平均: 届かない局", round(sum(spt_n) / max(1, len(spt_n)), 2),
      "・届いた局", round(sum(r["steps"] / r["turns"] for r in cap) / max(1, len(cap)), 2))
secs = [r["sec"] for r in rows]
print("1 局の秒 平均", round(sum(secs) / N, 1), "・届いた局の平均", round(sum(r["sec"] for r in cap) / max(1, len(cap)), 1))
out.update(cap_leaders=dict(lc), vega_games=len(vg), passive_mean={"short": sum(pv) / max(1, len(pv)),
           "cap": sum(pc) / max(1, len(pc))}, sec_mean=sum(secs) / N)
for r in err:
    print("ERR", r["seed"], r["leaders"], r["error"])
json.dump(out, open(os.path.join(d, "summary.json"), "w"), ensure_ascii=False, indent=1, default=str)
