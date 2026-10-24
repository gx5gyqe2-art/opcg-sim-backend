"""WP weak-points §2: 【ダブルアタック】を「効果で得ている」ユニット（印字に無い）が盤面に在る判断点を数える（1 回きり）。
NRel のトークン状態は今のキーワードのうちブロッカー（非レスト）と速攻（召喚酔い）しか持たず、【ダブルアタック】は
カード表の印字の旗（本文の先頭 60 字に語があるか）だけ＝効果で得た分は見えない。reaudit_wr の 100 局の手の列を当て直す。"""
import json, sys, collections
sys.path.insert(0, "tests/scripts")
import blunder_audit as BA
BA._init_judge(1)
db = {c["number"]: c for c in json.load(open("opcg_sim/data/opcg_cards.json"))}


def printed(cid):
    c = db.get(cid) or {}
    t = (c.get("効果(テキスト)") or "") + (c.get("トリガー") or "")
    return "ダブルアタック" in t[:60]


cnt = collections.Counter()
ex = []
for l in open("n1_results/reaudit_wr/blunder/games.jsonl"):
    g = json.loads(l)
    if g.get("void"):
        continue
    game = BA.replay(g, 0)
    seen_game = False
    for s, mv in enumerate(g["moves"]):
        pend = json.loads(game.pending_json())
        h = json.loads(game.hidden_json())
        cnt["points"] += 1
        tp = h["manager"]["turn_player"]
        P = h["players"][tp]
        for c in [P["leader"]] + P["field"]:
            kws = (c.get("current_keywords") or []) + (c.get("timed_keywords") or [])
            if "ダブルアタック" in kws:
                key = "granted" if not printed(c["card_id"]) else "printed"
                cnt[key + "_unit_points"] += 1
                if key == "granted":
                    cnt["granted_leader" if c is P["leader"] else "granted_char"] += 1
                    if not seen_game:
                        seen_game = True
                        cnt["games_with_granted"] += 1
                        ex.append([g["seed"], s, c["card_id"]])
        name = pend["player_id"]
        BA.apply(game, name, mv)
    # 印字で旗が立つのに自分は得ていない札（「自分のリーダーは【ダブルアタック】を得る」の類）
print(json.dumps(dict(cnt), ensure_ascii=False))
print(ex[:20])
fl = [n for n, c in db.items() if "ダブルアタック" in ((c.get("効果(テキスト)") or "")[:60]) and not (c.get("効果(テキスト)") or "").startswith("【ダブルアタック】")]
print("text-head flag but not own keyword:", len(fl), fl[:15])
json.dump({"counts": cnt, "examples": ex, "flag_not_own": fl}, open("n1_results/weak_points/da_scan.json", "w"), ensure_ascii=False, indent=1)
