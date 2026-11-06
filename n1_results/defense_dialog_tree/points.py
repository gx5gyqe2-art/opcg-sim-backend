"""§7.85（読みは旗のほかに畳む形 life1_fold も）: 4462251・172（守り）と 4461056・63（大きな損）を旗 defense_dialog_tree の off／life1／all で打ち直し、
16 局の打ち比べ（今のターンの残りは両席とも出荷の設定＋同じ旗・その先は審判 r29f4 160 回）をする（1 回きり）。
python n1_results/defense_dialog_tree/points.py [K]"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "scripts")); sys.path.insert(0, ROOT); os.chdir(ROOT)
import blunder_audit as BA
import defense_audit as DA
import move_judge as MJ
from opcg_sim.loop import engine as E, decks as D

K = int(sys.argv[1]) if len(sys.argv) > 1 else 16
NET = "opcg_sim/data/learned/nrel_r29f4.npz"
E.engine()
net = E.load_net(NET)
BA._G["db"] = D.load_db()
ref = E.SeatSpec(NET, sims=MJ.REFEREE_SIMS)
MODES = ("off", "life1", "all")


def ship(mode):
    return E.SeatSpec(NET, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=E.PLAY_WORLDS,
                      lethal_proof=E.PLAY_LETHAL_PROOF, **({} if mode == "off" else {"defense_dialog_tree": mode}))


def reads(game, who, base_opts, n=8):
    """旗ごとに seed を n 通り替えて読み、手と候補の値を並べる（#0 は実局の seed）。"""
    out = {}
    for mode in MODES + ("life1_fold",):
        rows = []
        for i in range(n):
            o = dict(base_opts)
            if i:
                o["search_seed"] = (o["search_seed"] ^ (i * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
            if mode == "life1_fold":
                o["defense_dialog_tree"], o["defense_dialog_fold"] = "life1", True
            elif mode != "off":
                o["defense_dialog_tree"] = mode
            d = json.loads(game.fork().decide(who, json.dumps(o)))
            rows.append({"move": BA.desc(game, d.get("move")), "mv": d.get("move"), "window": d.get("window"),
                         "Q": [round(x, 3) for x in (d.get("stats") or {}).get("Q", [])]})
        out[mode] = rows
    return out


def playout(game, who, seed, step, real):
    """旗ごとに: 点の手も今のターンの残りもその旗の出荷の設定で打ち、K 局の勝率。"""
    res = {}
    for mode in ("off", "life1", "all"):
        sp = ship(mode)
        ws_l, lost = [], 0
        for i in range(K):
            ws = MJ.world_seed(seed, step, DA.PLAYOUT_OFFSET + i)
            g = game.fork()
            # 点の手: その旗で、配り方 i の探索の乱数で読む
            o = json.loads(sp.decide_opts(int(ws), int(game.turn_count or 0), who, None))
            mv = json.loads(g.decide(who, json.dumps(o)))["move"]
            info = {}
            w, err = BA.rollout_ship(game, who, mv, ws, ref, sp, both=True, real=real, info=info)
            ws_l.append(w)
            lost += int(bool(info.get("lost_this_turn")))
        got = [x for x in ws_l if x is not None]
        res[mode] = {"n": len(got), "win": round(sum(got) / max(1, len(got)), 3), "lost_this_turn": lost, "w": ws_l}
    return res


ONLY = os.environ.get("ONLY", "")


def p4462251():
    # 守りの監査の局: 両席 r29f4・世界 1 本 × 1,280 回・予算なし・証明 off・random × synth_roles
    a = argparse.Namespace(net=None, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=1, lethal_proof=None, window_worlds=None)
    game, move, got = DA.replay_to(4462251, a, "p1", 172)
    base = json.loads(got["spec"].decide_opts(4462251, got["turn"], "p1", None))
    h = json.loads(game.hidden_json())
    r = {"seed": 4462251, "step": 172, "who": "p1", "live": BA.desc(game, move),
         "life": [len(h["players"][s]["life"]) for s in ("p1", "p2")],
         "pending": json.loads(game.pending_json()).get("action"),
         "legal": [BA.desc(game, m) for m in json.loads(game.legal_json("p1"))], "reads_audit_cfg": reads(game, "p1", base)}
    r["reads_ship"] = reads(game, "p1", json.loads(ship("off").decide_opts(4462251, got["turn"], "p1", None)))
    r["playout_real"] = playout(game, "p1", 4462251, 172, True)
    return r


def p4461056():
    # 大きな損の局: 両席 r29f4・出荷の設定・random × real
    g = next(json.loads(l) for l in open("n1_results/reaudit_r29f4/blunder/games.jsonl") if json.loads(l)["seed"] == 4461056)
    p = next(x for x in g["points"] if x["step"] == 63)
    game = BA.replay(g, 63)
    who = p["seat"]
    h = json.loads(game.hidden_json())
    base = {"net": net, "search_seed": E.search_seed(4461056, p["turn"], who), "sims": E.PLAY_SIMS, "budget": E.PLAY_BUDGET,
            "worlds": E.PLAY_WORLDS, "lethal_proof": E.PLAY_LETHAL_PROOF}
    r = {"seed": 4461056, "step": 63, "who": who, "live": BA.desc(game, p["move"]),
         "life": [len(h["players"][s]["life"]) for s in ("p1", "p2")],
         "pending": json.loads(game.pending_json()).get("action"),
         "legal": [BA.desc(game, m) for m in json.loads(game.legal_json(who))], "reads_ship": reads(game, who, base)}
    r["playout_real"] = playout(game, who, 4461056, 63, True)
    r["playout_fair"] = playout(game, who, 4461056, 63, False)
    return r


fns = {"4462251": p4462251, "4461056": p4461056}
for key, fn in fns.items():
    if ONLY and ONLY != key:
        continue
    r = fn()
    print(json.dumps(r, ensure_ascii=False), flush=True)
    json.dump(r, open(f"n1_results/defense_dialog_tree/point_{key}.json", "w"), ensure_ascii=False, indent=1)
