"""3461302（生成と同じ設定・生成役 r22）を打ち、p2 のメインの判断点で場に光月モモの助 OP16-084 が居る最初の盤面を
tests/fixtures/unpayable_activate_3461302.json に書き出す（1 回きり・WP rs-unpayable-activate）。
    python n1_results/unpayable_activate/capture_3461302.py"""
import json, sys
sys.path.insert(1, ".")
from opcg_sim.loop import record_gen as RG, driver as DR, decks as D

search = {"worlds": 1, "select_rule": "q_min_n", "q_min_frac": 0.125, "root_prior_temp": 2.0, "budget": 0}
RG._init_worker(1280, "opcg_sim/data/learned/nrel_r22.npz", 0.25, 4, "real", True, 0.03, 0.03, search, 0.0, None)
SEED = 3461302
OUT = "tests/fixtures/unpayable_activate_3461302.json"


class Found(Exception):
    pass


spec, db = RG._G["spec"], RG._G["db"]
rec = RG._Recorder(seed=SEED, db=db, eps_play=0.03, eps_hold=0.03, eps_target=0.0)


def obs(game, name, turn, step, out, move):
    rec(game, name, turn, step, out, move)
    p = json.loads(game.pending_json())
    if name != "p2" or p.get("action") != "MAIN_ACTION":
        return
    h = json.loads(game.hidden_json())
    momo = [c for c in h["players"]["p2"]["field"] if c["card_id"] == "OP16-084"]
    if momo:
        body = {"about": "WP rs-unpayable-activate（docs/search_efficiency_plan.md §7.46）の確かめの盤面。生成と同じ設定（生成役 r22・"
                         "gen_n52 の引数・デッキ real）で seed 3461302 を打ち、p2 のメインの判断点で場に光月モモの助 OP16-084 が居る最初の盤面"
                         "（n1_results/unpayable_activate/capture_3461302.py）。修正前はここから p2 が OP16-084 の【起動メイン】を繰り返して 400 手の上限に当たった。",
                "seed": SEED, "step": step, "turn": turn, "momo_uuid": momo[0]["uuid"], "hidden": h}
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            f.write("\n")
        print("captured step", step, "turn", turn, momo[0]["uuid"])
        raise Found


la, lb = D.leader_pair(db, SEED, "random")
p1, p2 = D.build_pair(db, la, lb, SEED, "real", with_kinds=True)[:2]
try:
    DR.run_game(SEED, {"p1": spec, "p2": spec}, p1, p2, max_steps=RG.MAX_STEPS, observer=obs, post=rec.post, swap=rec.swap)
    print("not found")
except Found:
    pass
