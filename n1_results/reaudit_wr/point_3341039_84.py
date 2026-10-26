"""3341039・84（確かめ済みの損・自分の対話）の要求の中身を読む（1 回きり）。"""
import json, sys
sys.path.insert(0, "tests/scripts")
import blunder_audit as BA
BA._init_judge(1)
BA.OUT_DIR = "n1_results/reaudit_wr/blunder"
g = next(json.loads(l) for l in open("n1_results/reaudit_wr/blunder/games.jsonl") if json.loads(l)["seed"] == 3341039)
game = BA.replay(g, 84)
pend = json.loads(game.pending_json())
print(json.dumps({k: v for k, v in pend.items() if k != "candidates"}, ensure_ascii=False)[:800])
print([(c.get("card_id"), c.get("name")) for c in pend.get("candidates", [])])
print("played at 83:", BA.desc(BA.replay(g, 83), g["moves"][83]))
