"""WP defect-hunt-2: 実デッキの局を打ち、手ごとに盤面の不変量を確かめ、札ごとの「手札に在った／合法だった／選ばれた」を数える（1 回きりの台本）。

不変量（手を適用した直後の盤面で毎回）:
  cards  各席の 山札+手札+ライフ+場+トラッシュ+ステージ が開始時と同じ
  don    ドン!!デッキ+アクティブ+レスト+付与（リーダー・場・ステージ）が開始時と同じ
  field  場のキャラ ≤ 5
  uuid   全ゾーンで uuid が重複しない・持ち主（owner_id）と置き場の席が一致
  count  life_count／hand_count が置き場の枚数と一致
  t1     【ターン1回】を本文に 1 つだけ持つ札の【起動メイン】が同じターンに 2 回
使い方: python hunt_play.py OUT.jsonl SEED0 N [--sims 160] [--deck IDX ...] [--workers 4]
"""
import argparse, json, os, sys, time, collections, random, traceback
from multiprocessing import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("OPCG_LOG_SILENT", "1")
from opcg_sim.loop import engine as E, driver as DR, decks as D

DS = None
SPEC = None
TEXT = {}


def zones_of(p):
    z = p["zones"]
    st = p.get("stage") or z.get("stage")
    return z, st


def check(game, init, viol, step, turn, t1_seen, move, name):
    b = json.loads(game.board_json())
    dc = json.loads(game.deck_counts_json())
    seen = {}
    for seat, p in b["players"].items():
        z, st = zones_of(p)
        n_cards = dc[seat] + len(z["hand"]) + len(z["life"]) + len(z["field"]) + len(z["trash"]) + (1 if st else 0)
        att = int(p["leader"].get("attached_don") or 0) + sum(int(c.get("attached_don") or 0) for c in z["field"]) \
            + (int(st.get("attached_don") or 0) if st else 0)
        n_don = int(p["don_deck_count"]) + len(p["don_active"]) + len(p["don_rested"]) + att
        if seat not in init:
            init[seat] = (n_cards, n_don)
        if n_cards != init[seat][0]:
            viol.append({"k": "cards", "seat": seat, "step": step, "turn": turn, "got": n_cards, "want": init[seat][0], "move": move})
            init[seat] = (n_cards, init[seat][1])
        if n_don != init[seat][1]:
            viol.append({"k": "don", "seat": seat, "step": step, "turn": turn, "got": n_don, "want": init[seat][1], "move": move})
            init[seat] = (init[seat][0], n_don)
        if len(z["field"]) > 5:
            viol.append({"k": "field", "seat": seat, "step": step, "turn": turn, "got": len(z["field"]), "move": move})
        if int(p["life_count"]) != len(z["life"]) or int(p["hand_count"]) != len(z["hand"]):
            viol.append({"k": "count", "seat": seat, "step": step, "turn": turn, "move": move})
        allc = [("leader", p["leader"])] + [(zn, c) for zn in ("hand", "life", "field", "trash") for c in z[zn]] + ([("stage", st)] if st else [])
        for zn, c in allc:
            u = c.get("uuid")
            if u in seen:
                viol.append({"k": "uuid_dup", "seat": seat, "zone": zn, "other": seen[u], "step": step, "turn": turn, "cid": c.get("card_id"), "move": move})
            seen[u] = (seat, zn)
            if c.get("owner_id") not in (None, seat):
                viol.append({"k": "owner", "seat": seat, "zone": zn, "step": step, "turn": turn, "cid": c.get("card_id"), "owner": c.get("owner_id"), "move": move})
    return b


def play(seed):
    global DS, SPEC
    if DS is None:
        DS = D.user_decks() + D.official_decks()
        E.engine()
        SPEC = E.SeatSpec(sims=ARGS["sims"])
    rng = random.Random(seed * 31 + 7)
    if ARGS["decks"]:
        a = ARGS["decks"][seed % len(ARGS["decks"])]
        b = rng.randrange(len(DS))
        pair = (a, b) if seed % 2 == 0 else (b, a)
    else:
        pair = (rng.randrange(len(DS)), rng.randrange(len(DS)))
    p1, p2 = DS[pair[0]], DS[pair[1]]
    init, viol = {}, []
    stats = collections.defaultdict(lambda: [0, 0, 0, 0, 0])  # (seat,cid): hand-at-main, play-legal, play-chosen, act-legal, act-chosen
    t1 = collections.Counter()
    pend_kinds = collections.Counter()
    trig = collections.defaultdict(set)  # (turn, src, trigger) -> steps
    src_cid = {}
    uu2cid = {}
    steps_log = []

    def obs(game, name, turn, step, out, move):
        legal = json.loads(game.legal_json(name))
        if not any(a.get("action_type") == "TURN_END" for a in legal):
            return
        b = json.loads(game.board_json())
        p = b["players"][name]
        for c in p["zones"]["hand"]:
            uu2cid[c["uuid"]] = c["card_id"]
        for c in [p["leader"]] + p["zones"]["field"] + ([p.get("stage") or p["zones"].get("stage")] if (p.get("stage") or p["zones"].get("stage")) else []):
            uu2cid[c["uuid"]] = c["card_id"]
        for cid in {c["card_id"] for c in p["zones"]["hand"]}:
            stats[(name, cid)][0] += 1
        pl = set(); ac = set()
        for a in legal:
            u = (a.get("payload") or {}).get("uuid") or a.get("card_uuid")
            if a.get("action_type") == "PLAY" and u in uu2cid: pl.add(uu2cid[u])
            if a.get("action_type") == "ACTIVATE_MAIN" and u in uu2cid: ac.add(uu2cid[u])
        for cid in pl: stats[(name, cid)][1] += 1
        for cid in ac: stats[(name, cid)][3] += 1
        u = (move.get("payload") or {}).get("uuid") or move.get("card_uuid")
        if move.get("action_type") == "PLAY" and u in uu2cid: stats[(name, uu2cid[u])][2] += 1
        if move.get("action_type") == "ACTIVATE_MAIN" and u in uu2cid:
            stats[(name, uu2cid[u])][4] += 1
            t1[(turn, u)] += 1

    def post(game, name, turn, step, move, events):
        mv = None if move is None else {"at": move.get("action_type"), "u": (move.get("payload") or {}).get("uuid") or move.get("card_uuid")}
        pend = json.loads(game.pending_json() or "null") or {}
        pk = pend.get("action")
        pend_kinds[(bool(game.has_interaction), pk)] += 1
        if not game.has_interaction:
            check(game, init, viol, step, turn, None, mv, name)
        for e in events:
            if e.get("type") == "EFFECT" and e.get("source_uuid") and e.get("trigger") not in (None, "ACTIVATE_MAIN"):
                trig[(turn, e["source_uuid"], e.get("trigger"))].add(step)
                src_cid[e["source_uuid"]] = e.get("card_name")
        if move is not None:
            steps_log.append((turn, name, mv["at"], mv["u"] and uu2cid.get(mv["u"], mv["u"][:8]), [(e.get("type"), e.get("card_name"), e.get("action"), e.get("trigger")) for e in events if e.get("type") == "EFFECT"]))

    t0 = time.time()
    err = None
    try:
        r = DR.run_game(seed, {"p1": SPEC, "p2": SPEC}, p1, p2, observer=obs, post=post)
        winner, turns, steps = r["winner"], r["turns"], r["steps"]
    except Exception as exc:  # noqa: BLE001
        winner, turns, steps, err = None, None, len(steps_log), f"{type(exc).__name__}: {exc}"[:500]
    t1d = [{"turn": k[0], "uuid": k[1], "cid": uu2cid.get(k[1]), "n": n} for k, n in t1.items() if n >= 2]
    trig2 = [{"turn": k[0], "src": src_cid.get(k[1]), "cid": uu2cid.get(k[1]), "trigger": k[2], "steps": sorted(v)} for k, v in trig.items() if len(v) >= 2]
    out = {"seed": seed, "trig_multi": trig2, "decks": list(pair), "leaders": [p1[0], p2[0]], "winner": winner, "turns": turns,
           "steps": steps, "err": err, "viol": viol[:50], "n_viol": len(viol), "act_twice": t1d,
           "stats": {f"{s}|{c}": v for (s, c), v in stats.items()}, "sec": round(time.time() - t0, 1), "pend": {f"{a}|{b}": n for (a, b), n in pend_kinds.items()}}
    if viol or err or (steps or 0) > 600 or (turns or 99) <= 6 or ARGS.get("log"):
        out["log"] = steps_log
    return out


ARGS = {}


def init_worker(a):
    ARGS.update(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("seed0", type=int); ap.add_argument("n", type=int)
    ap.add_argument("--sims", type=int, default=160); ap.add_argument("--deck", type=int, nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    cfg = {"sims": a.sims, "decks": a.deck, "log": a.log}
    with Pool(a.workers, initializer=init_worker, initargs=(cfg,)) as pool, open(a.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(play, range(a.seed0, a.seed0 + a.n))):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            print(i, r["seed"], r["winner"], r["turns"], r["steps"], r["n_viol"], r["err"], r["sec"], flush=True)
