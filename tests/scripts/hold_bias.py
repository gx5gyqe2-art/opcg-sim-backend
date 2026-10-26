"""「しない」を選ぶ・選ばない判断の偏り（WP `hold-bias`・`docs/search_efficiency_plan.md` §7.59・台帳 2+455）。

読み取りだけの計測（エンジン・ネット・既定の挙動は変えない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。
土台は `blunder_audit.py`（局の記録・`pick_b`・`rollout_ship`）をそのまま使う。

1. ``play``: 両席 r27・**出荷の設定**（`E.PLAY_*`＝1,280 回・世界 4 本・枝予算なし・詰みの証明 root・公平）・random × real で
   局を打ち、全ての判断点を記録する（`blunder_audit.play_one` と同じ記録・`games.jsonl`）。
2. ``judge``: 候補 2 つ以上の判断点ごとに、`blunder_audit.pick_b` で打った手 a と比べる手 b（根の次点）を取り、
   **a か b のどちらかが「しない」側で、もう一方が「する」側の点だけ**を判定する（`hold_scene`）。
   審判は「今のターンの残りを両席とも出荷の設定で打ち継ぎ、その先を審判〔r27・160 回・世界 1 本〕」
   （`blunder_audit.rollout_ship(both=True)`・§7.59 の手順 3）。公平な配り直し・a と b の対・K 通り（既定 16）。
   点は場面ごとに上限まで（既定 150・ターン終了 300・カウンター 200）を pure な順位で選び、場面を交互に並べて判定する。
   続き（commit）・証明（lethal）の点は数えない（打った手は前の判断点で決めた計画の続き）。
3. ``summary``: 場面ごとに「する − しない」の勝率の差の平均と 95% 区間（点の間の t 区間）・CPU が選んだ側の別・
   ターンの段階（序盤 1〜4・中盤 5〜8・終盤 9〜）の別（`summary.json`）。

    python tests/scripts/hold_bias.py play --games 150 --workers 4
    python tests/scripts/hold_bias.py judge --workers 4
    python tests/scripts/hold_bias.py summary

再開できる（終えた局・点は飛ばす）。
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blunder_audit as BA  # noqa: E402
import move_judge as MJ  # noqa: E402

from opcg_sim.loop import decks as D, engine as E  # noqa: E402

OUT_DIR = os.path.join("n1_results", "hold_bias")
NET = "opcg_sim/data/learned/nrel_r27.npz"
SEED_BASE = 4171000
#: 判定の配り方の番号のずらし（blunder_audit の judge／recheck／verify と重ねない）。
OFFSET = 3000
_G = {}


# ---------------------------------------------------------------- 「しない」の判定

def _is_hold(mv, act=None):
    """手が「しない」側か（True）・「する」側か（False）・どちらでもないか（None）。``act`` は待ちの種類
    （確認〔CONFIRM_*〕は accepted で、選択は選んだ札の有無で決める）。"""
    if mv is None:
        return None
    at = mv.get("action_type")
    p = mv.get("payload") or {}
    if at == "TURN_END":
        return True
    if at in ("PLAY", "ATTACK", "ATTACH_DON", "DON_BOX", "ACTIVATE_MAIN", "SETUP_BOX"):
        return False
    if at == "PASS":
        return True
    if at in ("SELECT_COUNTER", "SELECT_BLOCKER"):
        return False
    if at == "RESOLVE_EFFECT_SELECTION":
        if p.get("accepted") is False:
            return True
        if str(act or "").startswith("CONFIRM"):
            return False
        return not p.get("selected_uuids")
    return None


def _do_type(mv):
    at = mv.get("action_type")
    p = mv.get("payload") or {}
    if at in ("DON_BOX", "ATTACK"):
        return "attack" if p.get("target_ids") else "attach"
    if at == "ATTACH_DON":
        return "attach"
    return {"PLAY": "play", "ACTIVATE_MAIN": "activate", "SETUP_BOX": "setup"}.get(at, str(at).lower())


def hold_scene(p, a, b):
    """(場面, a が「しない」側か) または None（「する」と「しない」の比べでない点）。

    場面: ``main_end``（メインの行動 vs ターン終了・行動の種類で細分 ``main_end:play`` 等）・``counter``／``blocker``
    （戦闘の窓の素通し）・``confirm``（任意効果・【トリガー】を使わない＝accepted false）・``select_none``（「最大 n 枚」の
    選択で何も選ばない）。効果の対話は自分の効果（``own``）と相手の効果の対話を席で分ける（``:opp``）。"""
    act = p.get("act") or ""
    ha, hb = _is_hold(a, act), _is_hold(b, act)
    if ha is None or hb is None or ha == hb:
        return None
    hold, do = (a, b) if ha else (b, a)
    at = hold.get("action_type")
    if at == "TURN_END":
        return "main_end:" + _do_type(do), ha
    if at == "PASS":
        return ("blocker" if act == "SELECT_BLOCKER" or do.get("action_type") == "SELECT_BLOCKER" else "counter"), ha
    if at == "RESOLVE_EFFECT_SELECTION":
        if do.get("action_type") != "RESOLVE_EFFECT_SELECTION":
            return None
        sub = ("trigger" if act == "CONFIRM_TRIGGER" else "confirm") if (hold.get("payload") or {}).get("accepted") is False \
            else "select_none"
        return sub + ("" if p.get("own") else ":opp"), ha
    return None


# ---------------------------------------------------------------- 1. play

def cmd_play(a):
    a.net = NET
    BA.cmd_play(a)


# ---------------------------------------------------------------- 2. judge

def _init(k):
    E.engine()
    _G["db"] = D.load_db()
    BA._G["db"] = _G["db"]
    BA._G["net"] = E.load_net(NET)
    BA._G["ship"] = BA.ship_spec(NET)
    _G["ref"] = E.SeatSpec(NET, sims=MJ.REFEREE_SIMS)
    _G["k"] = int(k)


def judge_point(job):
    g, p = job
    t0 = time.time()
    seed, step, name = g["seed"], p["step"], p["seat"]
    row = {"seed": seed, "step": step, "seat": name, "turn": p["turn"], "kind": p["kind"], "dtype": p["dtype"],
           "own": p["own"], "act": p["act"], "n_cand": p["n_cand"], "my_life": p["my_life"], "opp_life": p["opp_life"],
           "my_hand": p["my_hand"], "opp_hand": p["opp_hand"], "ok": False, "why": None}
    try:
        game = BA.replay(g, step)
    except Exception as exc:  # noqa: BLE001
        row["why"] = f"replay: {type(exc).__name__}"
        return row
    pend = json.loads(game.pending_json())
    if pend.get("player_id") != name or int(game.turn_count or 0) != p["turn"]:
        row["why"] = "seat/turn"
        return row
    try:
        a, b, info = BA.pick_b(game, dict(p, seed=seed))
    except Exception as exc:  # noqa: BLE001
        row["why"] = f"pick_b: {type(exc).__name__}: {exc}"[:160]
        return row
    row.update(info)
    sc = hold_scene(p, a, b)
    if sc is None:
        row["why"] = "not_hold"
        return row
    row["scene"], row["a_hold"] = sc
    row["a"], row["b"] = a, b
    row["a_desc"], row["b_desc"] = BA.desc(game, a), BA.desc(game, b)
    row["top"] = [{"d": BA.desc(game, t["mv"]), "n": t["n"], "q": t["q"]} for t in p["top"][:4]]
    wa, wb, errs = [], [], {}
    for i in range(_G["k"]):
        ws = MJ.world_seed(seed, step, OFFSET + i)
        ra, ea = BA.rollout_ship(game, name, a, ws, _G["ref"], BA._G["ship"], both=True)
        rb, eb = BA.rollout_ship(game, name, b, ws, _G["ref"], BA._G["ship"], both=True)
        wa.append(ra)
        wb.append(rb)
        for e in (ea, eb):
            if e:
                errs[e] = errs.get(e, 0) + 1
    row["ref"] = dict(MJ.paired(wa, wb), a=wa, b=wb, errs=errs or None)
    row["ok"] = row["ref"]["diff"] is not None
    if not row["ok"]:
        row["why"] = "no_pairs"
    row["sec"] = round(time.time() - t0, 1)
    return row


def prefilter(p):
    """局の記録だけで決まる「しない」の場面（粗い名前）か None。main・window は記録の上位から a・b が決まる。
    続き・証明（読み直しで b が決まる）は数えない（notes: 打った手は前の判断点で決めた計画の続き）。"""
    if (p["n_cand"] or 0) < 2 or p["kind"] not in ("main", "window"):
        return None
    a = (p["a_box"] or p["move"]) if p["kind"] == "main" else p["move"]
    others = [t for t in p["top"] if BA.key(t["mv"]) != BA.key(a)]
    if not others:
        return None
    # window は同じカードの別の 1 枚を飛ばすが、「しない」かどうかは種類で決まるので上位の最初の別手で足りる
    sc = hold_scene(p, a, others[0]["mv"])
    return None if sc is None else coarse(sc[0])


def coarse(scene):
    return "main_end" if scene.startswith("main_end") else scene


def _rank(seed, step):
    """場面の中の点の並び（局を足しても既に選んだ点の順位が変わらない・pure）。"""
    return E.search_seed(int(seed) ^ 0x401D_B1A5, int(step), "p1")


CAPS = {"main_end": 300, "counter": 200}


def cmd_judge(a):
    games = {g["seed"]: g for g in BA.read_jsonl(os.path.join(a.out, "games.jsonl"))}
    rpath = os.path.join(a.out, "points.jsonl")
    done = {(x["seed"], x["step"]) for x in BA.read_jsonl(rpath)}
    pools = {}
    for s in sorted(games):
        g = games[s]
        if g["winner"] is None and not g["points"]:
            continue
        light = {k: v for k, v in g.items() if k != "points"}
        for p in g["points"]:
            sc = prefilter(p)
            if sc is not None:
                pools.setdefault(sc, []).append((light, p))
    # 場面ごとに上限まで（並びは pure な順位）・場面をまたいで交互に並べる（途中で止めても場面が偏らない）
    queues = []
    for sc, v in sorted(pools.items()):
        v.sort(key=lambda j: _rank(j[0]["seed"], j[1]["step"]))
        v = v[:CAPS.get(sc, a.cap)]
        print(f"SCENE {sc} pool={len(pools[sc])} take={len(v)}", flush=True)
        queues.append([j for j in v if (j[0]["seed"], j[1]["step"]) not in done])
    jobs = []
    while any(queues):
        for q in queues:
            if q:
                jobs.append(q.pop(0))
    print(f"JUDGE todo {len(jobs)} (done {len(done)}) k={a.k}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.k,)) as pool:
        for i, row in enumerate(pool.imap_unordered(judge_point, jobs, chunksize=1)):
            row["k"] = a.k
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            r = row.get("ref") or {}
            print(f"JUDGE seed={row['seed']} step={row['step']} {row['kind']} {row.get('scene')} a_hold={row.get('a_hold')} "
                  f"why={row['why']} d={r.get('diff')} sec={row.get('sec')} ({i + 1}/{len(jobs)} {time.time() - t0:.0f}s)",
                  flush=True)
    print("JUDGE_DONE", flush=True)


# ---------------------------------------------------------------- 3. summary

def mean_ci(xs):
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "ci": [None, None]}
    m = sum(xs) / n
    if n == 1:
        return {"n": 1, "mean": round(m, 4), "ci": [None, None]}
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = 1.96 * sd / math.sqrt(n)
    return {"n": n, "mean": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)], "sd": round(sd, 4)}


def do_minus_hold(x):
    """点の「する − しない」（手を打った側から見た勝率の差・b − a を向きで直す）。"""
    d = x["ref"]["diff"]
    return d if x["a_hold"] else -d


def block(rows):
    """場面の集計: 全点の「する − しない」・CPU が「する」を選んだ点／「しない」を選んだ点での「選んだ側 − 次点」。"""
    dm = [do_minus_hold(x) for x in rows]
    chose_do = [x for x in rows if not x["a_hold"]]
    chose_hold = [x for x in rows if x["a_hold"]]
    out = {"all_do_minus_hold": mean_ci(dm),
           "chose_do": {"n": len(chose_do), "chosen_minus_alt": mean_ci([-x["ref"]["diff"] for x in chose_do])},
           "chose_hold": {"n": len(chose_hold), "chosen_minus_alt": mean_ci([-x["ref"]["diff"] for x in chose_hold])},
           "big_loss": sum(1 for x in rows if x["ref"]["diff"] >= 0.20 and (x["ref"]["ci"][0] or -1) > 0)}
    for side in ("chose_do", "chose_hold"):
        ci = out[side]["chosen_minus_alt"]["ci"]
        out[side]["bias"] = bool(ci[1] is not None and ci[1] < 0)
    return out


def phase(t):
    return "early_1-4" if t <= 4 else "mid_5-8" if t <= 8 else "late_9+"


def cmd_summary(a):
    games = BA.read_jsonl(os.path.join(a.out, "games.jsonl"))
    rows = BA.read_jsonl(os.path.join(a.out, "points.jsonl"))
    ok = [x for x in rows if x["ok"]]
    s = {"games": len(games), "void": sum(1 for g in games if g["winner"] is None),
         "seeds": [min(g["seed"] for g in games), max(g["seed"] for g in games)] if games else None,
         "decisions": sum(g["n_points"] for g in games), "eligible": sum(g["n_elig"] for g in games),
         "rows": len(rows), "ok": len(ok), "k": rows[0]["k"] if rows else None, "not_ok": {},
         "sec_per_game": round(sum(g["sec"] for g in games) / len(games), 1) if games else None,
         "sec_per_point": round(sum(x["sec"] for x in ok) / len(ok), 1) if ok else None}
    for x in rows:
        if not x["ok"]:
            s["not_ok"][x["why"]] = s["not_ok"].get(x["why"], 0) + 1
    by = {}
    for x in ok:
        by.setdefault(x["scene"].split(":")[0] if x["scene"].startswith("main_end") else x["scene"], []).append(x)
    s["scenes"] = {k: block(v) for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))}
    s["scenes_fine"] = {}
    for x in ok:
        s["scenes_fine"].setdefault(x["scene"], []).append(x)
    s["scenes_fine"] = {k: block(v) for k, v in sorted(s["scenes_fine"].items(), key=lambda kv: -len(kv[1]))}
    s["by_phase"] = {}
    for k, v in by.items():
        for ph in ("early_1-4", "mid_5-8", "late_9+"):
            sub = [x for x in v if phase(x["turn"]) == ph]
            if sub:
                s["by_phase"][k + "|" + ph] = block(sub)
    s["by_kind"] = {}
    for x in ok:
        s["by_kind"].setdefault(x["scene"].split(":")[0] + "|" + x["kind"], []).append(x)
    s["by_kind"] = {k: block(v) for k, v in s["by_kind"].items()}
    s["all"] = block(ok)
    # 1 局あたりの点の数（1 席・判定した点の母集団＝候補 2 つ以上で「する」と「しない」の比べの点）
    n_games = len({x["seed"] for x in rows}) or 1
    s["per_game_per_seat"] = {k: round(len(v) / n_games / 2, 2) for k, v in by.items()}
    txt = json.dumps(s, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    for k, v in s["scenes"].items():
        print(k, v["all_do_minus_hold"], "do:", v["chose_do"], "hold:", v["chose_hold"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play")
    p.add_argument("--games", type=int, default=150)
    p.add_argument("--seed-base", type=int, default=SEED_BASE)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("judge")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cap", type=int, default=150, help="場面ごとの点の上限（main_end 300・counter 200 は別）")
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("summary")
    p.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)
    {"play": cmd_play, "judge": cmd_judge, "summary": cmd_summary}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
