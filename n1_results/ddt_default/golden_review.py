"""a1 golden: off（旧い形）と all（新しい既定）の局を突き合わせ、最初に分かれた手の判断点を分類する。"""
import json, sys, collections
import opcg_engine
from opcg_sim.loop import engine as E
E.engine()
OFF = sys.argv[1]; NEW = "tests/fixtures/rs_goldens/replay"
BATTLE = {"SELECT_BLOCKER", "SELECT_COUNTER"}
def states(g):
    return json.loads(opcg_engine.replay(json.dumps(g["input"], ensure_ascii=False)))["states"]
def tp(st):
    ti = st.get("turn_info") or {}
    return ti.get("active_player_id")
def qualifies(st):
    pr = st.get("pending_request") or {}
    return bool(st.get("active_battle")) and pr.get("player_id") != tp(st) and pr.get("action") not in BATTLE
rows = []; nq_total = 0; nq_div = collections.Counter()
for i in range(50):
    f = f"a1_{5000150+i}.json"
    a = json.load(open(f"{OFF}/{f}")); b = json.load(open(f"{NEW}/{f}"))
    sa = states(a)
    setup_state = None
    nq = sum(1 for st in sa[:-1] if qualifies(st))   # 判断の前の盤面＝前の段の後の盤面
    nq_total += nq
    if a["hashes"] == b["hashes"]:
        rows.append({"seed": a["seed"], "same": True, "qual_points": nq}); continue
    ma = [s["move"] for s in a["input"]["steps"]]; mb = [s["move"] for s in b["input"]["steps"]]
    k = next((j for j in range(min(len(ma), len(mb))) if ma[j] != mb[j]), min(len(ma), len(mb)))
    pre = sa[k-1] if k >= 1 else None
    pr = (pre or {}).get("pending_request") or {}
    rows.append({"seed": a["seed"], "same": False, "first_div_step": k, "qual_points": nq,
                 "qualifies": bool(pre and qualifies(pre)), "action": pr.get("action"),
                 "decider": pr.get("player_id"), "turn_player": tp(pre) if pre else None,
                 "battle": bool(pre and pre.get("active_battle")),
                 "msg": pr.get("message"), "off_move": ma[k] if k < len(ma) else None,
                 "all_move": mb[k] if k < len(mb) else None,
                 "winner_off": a["summary"]["winner"], "winner_all": b["summary"]["winner"],
                 "steps_off": len(ma), "steps_all": len(mb)})
json.dump(rows, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
d = [r for r in rows if not r["same"]]
print("games", len(rows), "changed", len(d), "first-div at qualifying point", sum(r["qualifies"] for r in d))
print("qualifying points total (off games)", nq_total)
print(collections.Counter(r["action"] for r in d))
for r in d:
    if not r["qualifies"]: print("NOT QUAL", r)
