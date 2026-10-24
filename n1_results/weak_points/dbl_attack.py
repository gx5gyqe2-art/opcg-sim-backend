"""WP weak-points §2: 3301005・107（自分のライフ 0・ブロッカー 1 体・相手の OP16-003 のアタック）を読み直す（1 回きり）。
対照 3301025・117。局は reaudit_wr/tree_budget の手の列（blunder_audit.replay）。

候補（素通し／ブロック）ごとに:
  1. 子の盤面で相手のこのターンの詰みを総当たり（defense_audit.judge_full・本当の手札）
  2. 出荷の CPU（r13d・blunder_audit.ship_spec）が両席を打ち継いで戦闘が終わった盤面（相手のメインに戻った点）:
     使ったカウンター・ネットの静的な値（r13d／r22・両側から）・総当たり（このターンの詰み）・深い読み（5,120 回・世界 4 本）
  3. 審判 r13d（160 回）K 局: 勝率と「このターンのうちに負けた」割合（審判の打ち損じの量）
"""
import json, os, sys, time
sys.path.insert(0, "tests/scripts")
import numpy as np
import blunder_audit as BA
import defense_audit as DA
import move_judge as MJ
import lethal_audit as LA
from opcg_sim.loop import engine as E
from opcg_sim.learned.train import n_rel_train as T
from opcg_sim.learned.train.n_eff_feat import build_eff_tables

K = int(os.environ.get("K", "32"))
BA._init_judge(K)
S = "n1_results/reaudit_wr/tree_budget"
games = {g["seed"]: g for g in BA.read_jsonl(f"{S}/games.jsonl")}
pts = BA.read_jsonl(f"{S}/points.jsonl")
stats, ab, abm, pwr, isl, _v = build_eff_tables()
PYNETS = {n: T.NRelNet.load(f"opcg_sim/data/learned/nrel_{n}.npz", tables=(stats, ab, abm, pwr, isl)) for n in ("r13d", "r22")}


def static(game, who):
    out = {}
    enc = json.loads(game.encode(who))
    sc = np.asarray(enc["scalars"], np.float32)[None]
    ci = np.asarray(enc["card_idx"][:22], np.int64)[None]
    tok = np.asarray(enc["tokens"], np.float32).reshape(1, 22, -1)
    rom = np.asarray(enc["rel_om"], np.float32).reshape(1, 16, 6, -1)
    roo = np.asarray(enc["rel_oo"], np.float32).reshape(1, 16, 16, -1)
    for n, net in PYNETS.items():
        out[n] = round(float(net.value(sc, ci, tok, np.zeros_like(rom), np.zeros_like(roo))[0]), 3)
    return out


def deep(game, who, seed):
    opts = {"net": BA._G["net"], "search_seed": seed, **BA.DEEP}
    o = json.loads(game.fork().decide(who, json.dumps(opts)))
    rs = BA.ranked(o)
    tt = sum(r["n"] for r in rs)
    v = sum(r["n"] * r["q"] for r in rs) / tt if tt else None
    return {"best": BA.desc(game, o.get("move")), "v_mover": None if v is None else round(v, 3), "kind": o.get("kind")}


def follow_battle(game, me, opp, seed, turn):
    """出荷の CPU で両席を打ち継ぎ、戦闘が終わって相手のメインに戻るまで（か決着まで）。"""
    used = []
    for _ in range(60):
        if game.winner() is not None:
            break
        pend = json.loads(game.pending_json())
        h = json.loads(game.hidden_json())
        if pend.get("player_id") == opp and pend.get("action") == "MAIN_ACTION" and h["manager"]["active_battle"] is None:
            break
        name = pend["player_id"]
        out = json.loads(game.decide(name, BA._G["ship"].decide_opts(seed, turn, name, None)))
        mv = out["move"]
        used.append([name, pend.get("action"), BA.desc(game, mv)])
        BA.apply(game, name, mv)
    return used


if __name__ == "__main__":
    rows = []
    for p in pts:
        seed, step, me = p["seed"], p["step"], p["seat"]
        opp = "p2" if me == "p1" else "p1"
        game = BA.replay(games[seed], step)
        turn = int(game.turn_count)
        r = {"seed": seed, "step": step, "board": BA.board(game, me), "static_root": {"me": static(game, me), "opp": static(game, opp)}}
        for tag, mv in (("pass", p["a"] if p["a"]["action_type"] == "PASS" else p["b"]),
                        ("block", p["b"] if p["a"]["action_type"] == "PASS" else p["a"])):
            c = game.fork()
            c.apply_search_move(me, json.dumps(mv, ensure_ascii=False))
            t0 = time.time()
            d = {"move": BA.desc(game, mv), "exact_child": DA.judge_full(c, opp, 120.0, 3_000_000)}
            e = c.fork()
            d["follow"] = follow_battle(e, me, opp, seed, turn)
            d["end_winner"] = e.winner()
            if e.winner() is None:
                d["end_board"] = BA.board(e, me)
                d["end_static"] = {"me": static(e, me), "opp": static(e, opp)}
                d["end_exact"] = DA.judge_full(e, opp, 120.0, 3_000_000)
                dd = deep(e, opp, seed ^ 0x55AA)
                d["end_deep"] = dict(dd, v_me=None if dd["v_mover"] is None else -dd["v_mover"])
            # 審判: 勝ち・このターンのうちに負けた
            wins, lost_now = [], 0
            for i in range(K):
                ws = MJ.world_seed(seed, step, i)
                g = game.fork_world(me, "fair", int(ws))
                g.apply_search_move(me, json.dumps(mv, ensure_ascii=False))
                w, _ = MJ.play_out(g, BA._G["refs"]["r13d"], int(ws))
                wins.append(None if w is None else int(w == me))
                if w == opp and int(g.turn_count) == turn:
                    lost_now += 1
            d["ref"] = {"k": K, "win": round(sum(x for x in wins if x is not None) / max(1, sum(x is not None for x in wins)), 3),
                        "lost_this_turn": lost_now}
            d["sec"] = round(time.time() - t0, 1)
            r[tag] = d
        print(json.dumps(r, ensure_ascii=False), flush=True)
        rows.append(r)
    json.dump(rows, open("n1_results/weak_points/dbl_attack.json", "w"), ensure_ascii=False, indent=1)
