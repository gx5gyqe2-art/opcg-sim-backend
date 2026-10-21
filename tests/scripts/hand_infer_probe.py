"""相手の手札はどれだけ当てられるか（`docs/search_efficiency_plan.md` §7.12・WP `hand-infer-a`）。

**問い**: 公開情報（両席の場・トラッシュ・相手の手札の枚数・相手が打った手・**打てたのに打たなかった手**・
ターン・DON・ライフの枚数・相手のデッキの中身〔既知〕）から、相手の手札にどのカード種が何枚あるかを、
無作為（残りの山から一様＝超幾何）よりどれだけ当てられるか。

3 段（同じファイル・本線のコードは触らない）:

1. `record`: 両席 r13d・160 回（生成の物差し・`SeatSpec(None)`）・random × synth_roles で打ち、
   **各席の MAIN の判断点ごと**に (i) 公開情報の特徴 (ii) 正解＝相手の本当の手札（`Game.hidden_json`）
   (iii) 相手の残りの山＝デッキの中身 − 公開済み（場・ステージ・トラッシュ・表向きのライフ）を JSONL（gzip）に書く。
   席 A＝p1（先手）が本来の対象。p2 の視点の行も同じ局から書く（`me` 列）＝集計は両方出す。
2. `analyze`: 局で分けた holdout（seed % 7 == 0）で「相手の手札にカード種 k が 1 枚以上ある確率」を
   (a) 超幾何 (a') 超幾何＋公開で手札に入ったと分かっているカード (b) ロジスティック回帰（numpy・
   カード種を通して共有・入力は特徴＋超幾何の確率の logit＋カードの属性との交差項）(c) 小さな MLP（numpy）で当てる。
3. 測る: カード種ごとの対数損失と Brier の改善（全体・ターン帯・役割別）／相手の手札のカウンター値の
   合計の誤差／カウンター 1 枚以上の AUC／効いた特徴 10 個。

公開情報だけを特徴にする約束（**hidden から特徴を作らない**）: 盤面 dict（`board_json`）は相手の手札と
裏向きのライフの中身も持つので、使うのは場・ステージ・トラッシュ・リーダー・**表向きの**ライフ・枚数だけ。
カウンターの要求（`SELECT_COUNTER`）の `selectable_uuids` は守る側にしか見えないので読まない。
`hidden_json` は正解と、公開されたカード（名前と uuid 先頭 4 桁がイベントに載る）の uuid の解決にだけ使う。

  python tests/scripts/hand_infer_probe.py record --games 2000 --seed-base 3001000 --jobs 4 \\
      --out n1_results/hand_infer/rows.jsonl.gz
  python tests/scripts/hand_infer_probe.py analyze --rows n1_results/hand_infer/rows.jsonl.gz \\
      --out n1_results/hand_infer/result.json
"""
import argparse
import collections
import gzip
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("OPCG_LOG_SILENT", "1")

import numpy as np  # noqa: E402

from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402

SEATS = ("p1", "p2")
_TARGET_RE = re.compile(r"\(([0-9a-f]{4})\)$")
_COUNTER_RE = re.compile(r"\(\+(\d+)\)")


def _other(s):
    return "p2" if s == "p1" else "p1"


# --- 1. 記録 ------------------------------------------------------------------

class View:
    """1 席（me）から見た相手（opp）の公開情報の台帳。"""

    def __init__(self, me, deck_list):
        self.me, self.opp = me, _other(me)
        self.deck = collections.Counter(deck_list)      # 相手のデッキの中身（既知）
        self.mulligan = False
        self.known = {}                                  # 公開で手札に入った uuid → card_id
        self.battles = []                                # 自分の攻撃に対する相手の守り
        self.open_battle = None
        self.turn_end = None                             # 相手の直前の手番の終わり
        self.plays = collections.Counter()               # 相手の手番ごとの登場・起動の数
        self.played_cost = collections.Counter()
        self.events_played = collections.Counter()
        self.counter_used = 0                            # 局を通して相手が使ったカウンター
        self.counter_value = 0
        self.counter_events = 0


def _public(board, seat):
    """盤面 dict の公開領域（場・ステージ・トラッシュ・表向きのライフ）の uuid → card_id。"""
    z = board["players"][seat]["zones"]
    out = {}
    for c in (z.get("field") or []):
        out[c["uuid"]] = c["card_id"]
    if z.get("stage"):
        out[z["stage"]["uuid"]] = z["stage"]["card_id"]
    for c in (z.get("trash") or []):
        out[c["uuid"]] = c["card_id"]
    for c in (z.get("life") or []):
        if c.get("is_face_up"):
            out[c["uuid"]] = c["card_id"]
    return out


def _power_of(board, uuid):
    for s in SEATS:
        p = board["players"][s]
        if p["leader"]["uuid"] == uuid:
            return p["leader"]["power"], True
        for c in (p["zones"].get("field") or []):
            if c["uuid"] == uuid:
                return c["power"], False
    return None, False


def _resolve(target, uuids):
    m = _TARGET_RE.search(target or "")
    if not m:
        return None
    hits = [u for u in uuids if u.startswith(m.group(1))]
    return hits[0] if len(hits) == 1 else None


def _features(v, board, turn):
    """行の特徴（公開情報だけ）。"""
    me, opp = board["players"][v.me], board["players"][v.opp]
    f = {"turn": turn, "opp_hand": opp["hand_count"], "opp_life": opp["life_count"],
         "my_life": me["life_count"], "opp_don_active": len(opp["don_active"]),
         "opp_don_rested": len(opp["don_rested"]), "opp_don_deck": opp["don_deck_count"],
         "my_hand": me["hand_count"], "opp_mulligan": int(v.mulligan),
         "opp_field_n": len(opp["zones"].get("field") or []),
         "opp_field_cost": sum(int(c.get("cost") or 0) for c in (opp["zones"].get("field") or [])),
         "opp_trash_n": len(opp["zones"].get("trash") or []),
         "opp_counter_used": v.counter_used, "opp_counter_value": v.counter_value,
         "opp_counter_events": v.counter_events, "known_n": len(v.known)}
    te = v.turn_end
    if te is not None:
        f.update({"te_seen": 1, "te_unused_don": te["unused_don"], "te_don_total": te["don_total"],
                  "te_hand": te["hand"], "te_plays": te["plays"], "te_played_cost": te["played_cost"],
                  "te_events_played": te["events_played"], "te_field_n": te["field_n"],
                  "te_ago": turn - te["turn"]})
    else:
        f.update({"te_seen": 0, "te_unused_don": 0, "te_don_total": 0, "te_hand": 0, "te_plays": 0,
                  "te_played_cost": 0, "te_events_played": 0, "te_field_n": 0, "te_ago": 0})
    for tag, t in (("cur", turn), ("prev", turn - 2)):
        bs = [b for b in v.battles if b["turn"] == t and b.get("closed")]
        nc = [b for b in bs if b["counters"] == 0]
        ncl = [b for b in nc if b["leader"] and b["margin"] is not None]
        f.update({
            f"{tag}_battles": len(bs), f"{tag}_nocounter": len(nc),
            f"{tag}_countered": len(bs) - len(nc),
            f"{tag}_counter_value": sum(b["value"] for b in bs),
            f"{tag}_blocked": sum(1 for b in bs if b["blocked"]),
            # 「1 枚で止まる攻撃を通した」＝1000／2000 のカウンター 1 枚で守れた差（攻撃側は同値で勝つ）
            f"{tag}_nc_save1k": sum(1 for b in ncl if b["margin"] < 1000),
            f"{tag}_nc_save2k": sum(1 for b in ncl if b["margin"] < 2000),
            f"{tag}_nc_char": sum(1 for b in nc if not b["leader"]),
            f"{tag}_nc_min_margin": (min(b["margin"] for b in ncl) / 1000.0) if ncl else 9.0,
        })
    return f


def play_one(job):
    g, a = job
    if a["pin"]:
        wid = (mp.current_process()._identity or (1,))[0] - 1
        os.sched_setaffinity(0, {a["cores"][wid % len(a["cores"])]})
    E.engine()
    db = D.load_db()
    spec = E.SeatSpec(a["net"], sims=a["sims"])
    seed = a["seed_base"] + g
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, a["decks"])
    decks = {"p1": p1[1], "p2": p2[1]}
    views = {"p1": View("p1", decks["p2"]), "p2": View("p2", decks["p1"])}
    rows = []
    diag = collections.Counter()
    prev_public = {s: {} for s in SEATS}
    all_uuids = {}

    def opp_uuids(game, seat):
        if seat not in all_uuids:
            h = json.loads(game.hidden_json())["players"][seat]
            us = []
            for zone in ("deck", "hand", "life", "field", "trash"):
                us += [c["uuid"] for c in (h.get(zone) or [])]
            all_uuids[seat] = us
        return all_uuids[seat]

    def observer(game, name, turn, step, out, move):
        pend = json.loads(game.pending_json())
        act = pend.get("action")
        board = None
        for v in views.values():
            if name == v.opp and act == "SELECT_COUNTER":
                board = board or json.loads(game.board_json())
                ab = board.get("active_battle") or {}
                key = (turn, ab.get("attacker_uuid"), ab.get("target_uuid"))
                if v.open_battle is None or v.open_battle["key"] != key:
                    ap, _ = _power_of(board, ab.get("attacker_uuid"))
                    dp, is_leader = _power_of(board, ab.get("target_uuid"))
                    margin = None if ap is None or dp is None else ap - dp - int(ab.get("counter_buff") or 0)
                    v.open_battle = {"key": key, "turn": turn, "margin": margin, "leader": is_leader,
                                     "counters": 0, "value": 0, "blocked": v.open_battle is not None
                                     and v.open_battle.get("pre_block") == turn,
                                     "opp_hand": board["players"][v.opp]["hand_count"],
                                     "opp_life": board["players"][v.opp]["life_count"]}
                    v.battles.append(v.open_battle)
                if move.get("action_type") == "SELECT_COUNTER":
                    v.open_battle["counters"] += 1
                else:
                    v.open_battle["closed"] = True
            if name == v.opp and act == "SELECT_BLOCKER" and move.get("action_type") == "SELECT_BLOCKER":
                v.open_battle = {"key": None, "pre_block": turn}
            if name == v.opp and act == "MAIN_ACTION" and move.get("action_type") == "TURN_END":
                board = board or json.loads(game.board_json())
                o = board["players"][v.opp]
                v.turn_end = {"turn": turn, "unused_don": len(o["don_active"]),
                              "don_total": len(o["don_active"]) + len(o["don_rested"]),
                              "hand": o["hand_count"], "plays": v.plays[turn],
                              "played_cost": v.played_cost[turn], "events_played": v.events_played[turn],
                              "field_n": len(o["zones"].get("field") or [])}
            if name == v.me and act == "MAIN_ACTION":
                board = board or json.loads(game.board_json())
                hid = json.loads(game.hidden_json())["players"][v.opp]
                pub = _public(board, v.opp)
                for u in list(v.known):
                    if u in pub:
                        del v.known[u]
                seen = collections.Counter(pub.values())
                pool = {cid: n - seen.get(cid, 0) for cid, n in v.deck.items() if n - seen.get(cid, 0) > 0}
                truth = collections.Counter(c["card_id"] for c in hid["hand"])
                hand_uuids = {c["uuid"] for c in hid["hand"]}
                known = collections.Counter(v.known.values())
                diag["rows"] += 1
                diag["known"] += len(v.known)
                diag["known_ok"] += sum(1 for u in v.known if u in hand_uuids)
                diag["truth_outside_pool"] += sum(max(0, n - pool.get(cid, 0)) for cid, n in truth.items())
                diag["hand_count_mismatch"] += int(len(hid["hand"]) != board["players"][v.opp]["hand_count"])
                rows.append({"g": seed, "me": v.me, "turn": turn, "step": step,
                             "h": board["players"][v.opp]["hand_count"],
                             "f": _features(v, board, turn), "pool": pool,
                             "known": dict(known), "truth": dict(truth)})

    def post(game, name, turn, step, move, events):
        board = json.loads(game.board_json())
        for v in views.values():
            for e in events:
                who = e.get("player")
                typ = e.get("type")
                if who == v.opp and typ == "MULLIGAN":
                    v.mulligan = True
                if who == v.opp and typ in ("PLAY", "ACTIVATE_MAIN"):
                    v.plays[turn] += 1
                if who == v.opp and typ == "COUNTER":
                    v.counter_used += 1
                    m = _COUNTER_RE.search(e.get("message") or "")
                    val = int(m.group(1)) if m else 0
                    v.counter_value += val
                    if v.open_battle is not None and v.open_battle.get("key") is not None:
                        v.open_battle["value"] += val
                if who == v.opp and typ == "EFFECT" and e.get("trigger") == "COUNTER":
                    v.counter_events += 1
                if typ == "EFFECT" and (e.get("dest") == "HAND" or e.get("action") in ("BOUNCE", "MOVE_TO_HAND")):
                    us = opp_uuids(game, v.opp)
                    src_text = ""
                    if e.get("source_uuid"):
                        hid = None
                        for t in (e.get("targets") or []):
                            u = _resolve(t, us)
                            if u is None:
                                continue
                            if u in prev_public[v.opp]:
                                v.known[u] = prev_public[v.opp][u]
                                diag["reveal_from_public"] += 1
                                continue
                            if not src_text:
                                src_text = _source_text(db, game, e["source_uuid"])
                            if who == v.opp and "公開" in src_text:
                                hid = hid or json.loads(game.hidden_json())["players"][v.opp]
                                cid = next((c["card_id"] for c in hid["hand"] if c["uuid"] == u), None)
                                if cid is not None:
                                    v.known[u] = cid
                                    diag["reveal_search"] += 1
            if move is not None and name == v.opp and move.get("action_type") == "PLAY":
                # 手札から出たカード＝PLAY イベントの名前と一致し、直前の公開領域に無かった uuid（場／ステージ／トラッシュ）
                played = [e.get("card_name") for e in events if e.get("type") == "PLAY" and e.get("player") == v.opp]
                pub = _public(board, v.opp)
                for nm in played[:1]:
                    for u, cid in pub.items():
                        c = db.get_card(cid)
                        if u not in prev_public[v.opp] and c is not None and c.name == nm:
                            v.played_cost[turn] += int(c.cost or 0)
                            if getattr(c.type, "name", "") == "EVENT":
                                v.events_played[turn] += 1
                            break
        for s in SEATS:
            prev_public[s] = _public(board, s)

    t0 = time.perf_counter()
    try:
        res = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, observer=observer, post=post)
        winner, turns = res["winner"], res["turns"]
    except DR.GameAborted as exc:
        winner, turns = None, -1
        diag["aborted"] += 1
        print(f"aborted {seed}: {exc}", flush=True)
    return {"game": g, "seed": seed, "winner": winner, "turns": turns, "rows": rows,
            "leaders": [la, lb], "decks": decks, "diag": dict(diag),
            "sec": round(time.perf_counter() - t0, 2)}


_SRC_CACHE = {}


def _source_text(db, game, uuid):
    if uuid in _SRC_CACHE:
        return _SRC_CACHE[uuid]
    h = json.loads(game.hidden_json())
    cid = None
    for s in SEATS:
        p = h["players"][s]
        cards = [p.get("leader")] + ([p["stage"]] if p.get("stage") else [])
        for zone in ("deck", "hand", "life", "field", "trash", "temp_zone"):
            cards += list(p.get(zone) or [])
        for c in cards:
            if c and c.get("uuid") == uuid:
                cid = c["card_id"]
    c = db.get_card(cid) if cid else None
    txt = (c.effect_text or "") if c is not None else ""
    _SRC_CACHE[uuid] = txt
    return txt


def cmd_record(a):
    done = set()
    if os.path.exists(a.out):
        with gzip.open(a.out, "rt") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["game"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cfg = {"sims": a.sims, "seed_base": a.seed_base, "net": a.net, "decks": a.decks,
           "pin": not a.no_pin, "cores": sorted(os.sched_getaffinity(0))}
    jobs = [(g, cfg) for g in range(a.games) if g not in done]
    print(f"{len(done)} done, {len(jobs)} to go", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.jobs) as pool, gzip.open(a.out, "at") as f:
        for i, res in enumerate(pool.imap_unordered(play_one, jobs, chunksize=4)):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                f.flush()
                print(f"[{i + 1}/{len(jobs)}] game {res['game']} rows {len(res['rows'])} "
                      f"{time.time() - t0:.0f}s", flush=True)


# --- 2. 当てる ------------------------------------------------------------------

def _lchoose(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def p_any(N, n, h):
    """超幾何: N 枚から h 枚を一様に引いて、n 枚ある種が 1 枚以上入る確率。"""
    if n <= 0 or h <= 0:
        return 0.0
    if N - n < h:
        return 1.0
    return 1.0 - math.exp(_lchoose(N - n, h) - _lchoose(N, h))


class CardInfo:
    def __init__(self, db):
        from opcg_sim.loop import deck_roles
        self.db, self.dr, self.c = db, deck_roles, {}

    def get(self, cid):
        if cid in self.c:
            return self.c[cid]
        m = self.db.get_card(cid)
        typ = getattr(getattr(m, "type", None), "name", "")
        txt = (m.effect_text or "") if m is not None else ""
        counter_event = typ == "EVENT" and any(
            getattr(ab.trigger, "name", "") == "COUNTER" for ab in (m.abilities or ()))
        blocker = any(str(k) == "ブロッカー" for k in (m.keywords or ())) if m is not None else False
        removal = bool(self.dr.classify(m)) if m is not None else False
        counter = int(getattr(m, "counter", 0) or 0)
        if removal:
            role = "removal"
        elif blocker:
            role = "blocker"
        elif counter > 0 or counter_event:
            role = "counter"
        else:
            role = "other"
        info = {"counter": counter, "cost": int(getattr(m, "cost", 0) or 0), "char": typ == "CHARACTER",
                "event": typ == "EVENT", "stage": typ == "STAGE", "counter_event": counter_event,
                "blocker": blocker, "removal": removal, "trigger": bool(getattr(m, "trigger_text", "")),
                "has_counter": counter > 0 or counter_event, "role": role,
                "onplay_search": "公開" in txt}
        self.c[cid] = info
        return info


ROW_KEYS = None


def _row_vec(f):
    """行の特徴 → 数値ベクトル（名前つき）。"""
    t = f["turn"]
    out = collections.OrderedDict()
    out["turn"] = t
    out["turn_le2"] = float(t <= 2)
    out["turn_34"] = float(3 <= t <= 4)
    for k in ("opp_hand", "opp_life", "my_life", "opp_don_active", "opp_don_rested", "opp_don_deck", "my_hand",
              "opp_mulligan", "opp_field_n", "opp_field_cost", "opp_trash_n", "opp_counter_used",
              "opp_counter_value", "opp_counter_events", "known_n", "te_seen", "te_unused_don", "te_don_total",
              "te_hand", "te_plays", "te_played_cost", "te_events_played", "te_field_n"):
        out[k] = float(f[k])
    for tag in ("cur", "prev"):
        for k in ("battles", "nocounter", "countered", "counter_value", "blocked", "nc_save1k", "nc_save2k",
                  "nc_char", "nc_min_margin"):
            out[f"{tag}_{k}"] = float(f[f"{tag}_{k}"])
    return out


def build(rows_path, ci):
    """(行, カード種) の組の表を作る。"""
    from array import array
    pairs = collections.defaultdict(lambda: array("f"))
    ROLE_ID = {"counter": 0, "blocker": 1, "removal": 2, "other": 3}
    rowtab = collections.defaultdict(list)
    names = None
    n_games = 0
    with gzip.open(rows_path, "rt") as fh:
        for line in fh:
            if not line.strip():
                continue
            gm = json.loads(line)
            n_games += 1
            for r in gm["rows"]:
                h = r["h"]
                pool, known, truth = r["pool"], r["known"], r["truth"]
                N = sum(pool.values())
                if N <= 0 or h <= 0:
                    continue
                rv = _row_vec(r["f"])
                if names is None:
                    names = list(rv.keys())
                rvec = list(rv.values())
                kn_total = sum(known.values())
                N2, h2 = N - kn_total, h - kn_total
                hold = int(r["g"] % 7 == 0)
                te_un = r["f"]["te_unused_don"] if r["f"]["te_seen"] else -1
                dt = r["f"]["te_don_total"]
                pc = []
                exp_counter = 0.0
                n_has_counter = 0
                n_2k = n_cev = 0
                for cid, n in pool.items():
                    info = ci.get(cid)
                    p = p_any(N, n, h)
                    kn = known.get(cid, 0)
                    p2 = 1.0 if kn > 0 else p_any(N2, n - kn, h2)
                    y = int(truth.get(cid, 0) > 0)
                    pc.append(info)
                    exp_counter += info["counter"] * n * h / N
                    n_has_counter += n if info["has_counter"] else 0
                    n_2k += n if info["counter"] >= 2000 else 0
                    n_cev += n if info["counter_event"] else 0
                    pairs["row"].append(len(rowtab["g"]))
                    pairs["y"].append(y)
                    pairs["cnt"].append(truth.get(cid, 0))
                    pairs["p"].append(p)
                    pairs["p2"].append(p2)
                    pairs["n"].append(n)
                    pairs["known"].append(kn)
                    pairs["counter"].append(info["counter"] / 1000.0)
                    pairs["counter_event"].append(float(info["counter_event"]))
                    pairs["cost"].append(float(info["cost"]))
                    pairs["char"].append(float(info["char"]))
                    pairs["event"].append(float(info["event"]))
                    pairs["stage"].append(float(info["stage"]))
                    pairs["blocker"].append(float(info["blocker"]))
                    pairs["removal"].append(float(info["removal"]))
                    pairs["trigger"].append(float(info["trigger"]))
                    pairs["afford_te"].append(float(te_un >= 0 and info["cost"] <= te_un and not info["event"]))
                    pairs["afford_te_any"].append(float(te_un >= 0 and info["cost"] <= dt and not info["event"]))
                    pairs["role"].append(ROLE_ID[info["role"]])
                truth_counter = sum(ci.get(c)["counter"] * k for c, k in truth.items())
                truth_has = int(any(ci.get(c)["has_counter"] for c in truth))
                known_counter = sum(ci.get(c)["counter"] * k for c, k in known.items())
                rowtab["g"].append(r["g"])
                rowtab["me"].append(r["me"])
                rowtab["turn"].append(r["turn"])
                rowtab["hold"].append(hold)
                rowtab["h"].append(h)
                rowtab["N"].append(N)
                rowtab["x"].append(rvec)
                rowtab["exp_counter"].append(exp_counter / 1000.0)
                rowtab["known_counter"].append(known_counter / 1000.0)
                rowtab["p_has_counter"].append(p_any(N, n_has_counter, h))
                rowtab["truth_counter"].append(truth_counter / 1000.0)
                rowtab["truth_has"].append(truth_has)
                rowtab["p_has_2k"].append(p_any(N, n_2k, h))
                rowtab["truth_has_2k"].append(int(any(ci.get(c)["counter"] >= 2000 for c in truth)))
                rowtab["p_has_cev"].append(p_any(N, n_cev, h))
                rowtab["truth_has_cev"].append(int(any(ci.get(c)["counter_event"] for c in truth)))
    P = {k: np.frombuffer(v, dtype=np.float32).astype(np.float64 if k in ("p", "p2") else np.float32)
         for k, v in pairs.items()}
    P["row"] = P["row"].astype(np.int64)
    P["role"] = np.array(["counter", "blocker", "removal", "other"])[P["role"].astype(np.int64)]
    R = {k: np.asarray(v) for k, v in rowtab.items()}
    return P, R, names, n_games


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def design(P, R, names):
    """(行, カード種) の入力行列（名前つき）。"""
    X = np.asarray(R["x"], dtype=np.float32)[P["row"]]
    cols, mats = [], []

    def add(name, v):
        cols.append(name)
        mats.append(np.asarray(v, dtype=np.float32))

    lp = _logit(P["p"])
    add("logit_p_hyp", lp)
    add("log_n_unseen", np.log(P["n"]))
    add("known_k", P["known"])
    add("known_k_pos", (P["known"] > 0).astype(float))
    for k in ("counter", "counter_event", "cost", "char", "event", "stage", "blocker", "removal", "trigger",
              "afford_te", "afford_te_any"):
        add(f"card_{k}", P[k])
    rf = {n: X[:, i] for i, n in enumerate(names)}
    for n in names:
        add(f"row_{n}", rf[n])
    # 交差項（打たなかった手 × カードの属性）
    has_counter = ((P["counter"] > 0) | (P["counter_event"] > 0)).astype(float)
    for n in ("cur_nocounter", "cur_nc_save1k", "cur_nc_save2k", "cur_countered", "cur_counter_value",
              "prev_nocounter", "prev_nc_save1k", "prev_nc_save2k", "prev_countered", "prev_counter_value",
              "cur_nc_char", "prev_nc_char", "opp_counter_used", "cur_nc_min_margin", "prev_nc_min_margin"):
        add(f"hasctr_x_{n}", has_counter * rf[n])
        add(f"ctrval_x_{n}", P["counter"] * rf[n])
    add("cevent_x_te_unused_don", P["counter_event"] * rf["te_unused_don"])
    add("cevent_x_opp_don_active", P["counter_event"] * rf["opp_don_active"])
    add("afford_x_unused", P["afford_te"] * rf["te_unused_don"])
    add("afford_x_plays", P["afford_te"] * rf["te_plays"])
    add("cost_x_te_don_total", P["cost"] * rf["te_don_total"])
    add("char_x_te_unused", P["char"] * rf["te_unused_don"])
    add("event_x_te_events_played", P["event"] * rf["te_events_played"])
    add("blocker_x_te_unused", P["blocker"] * rf["te_unused_don"])
    add("lp_x_turn", lp * rf["turn"])
    add("lp_x_opp_hand", lp * rf["opp_hand"])
    add("lp_x_mulligan", lp * rf["opp_mulligan"])
    out = np.empty((len(mats[0]), len(mats)), dtype=np.float32)
    for i, m in enumerate(mats):
        out[:, i] = m
    return out, cols


def fit_logreg(X, y, l2=1.0, iters=30, rows=None, cols=None, chunk=200000):
    """L2 つきロジスティック回帰（Newton・切片つき・入力は標準化して渡す）。

    `rows`（行の添字）・`cols`（列の添字）で部分を指す＝大きな表を写さずに塊ごとに勾配とヘッセ行列を足す。
    """
    rows = np.arange(X.shape[0]) if rows is None else rows
    cols = np.arange(X.shape[1]) if cols is None else np.asarray(cols)
    d = len(cols)
    w = np.zeros(d + 1)
    reg = np.full(d + 1, l2)
    reg[-1] = 0.0
    for _ in range(iters):
        g = reg * w
        H = np.diag(reg + 1e-9)
        for s0 in range(0, len(rows), chunk):
            ri = rows[s0:s0 + chunk]
            xb = np.empty((len(ri), d + 1))
            xb[:, :d] = X[ri][:, cols]
            xb[:, d] = 1.0
            p = 1.0 / (1.0 + np.exp(-(xb @ w)))
            g += xb.T @ (p - y[ri])
            H += (xb * (p * (1 - p))[:, None]).T @ xb
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return w


def pred_logreg(w, X, cols=None, chunk=500000):
    cols = np.arange(X.shape[1]) if cols is None else np.asarray(cols)
    out = np.empty(X.shape[0])
    for s0 in range(0, X.shape[0], chunk):
        z = X[s0:s0 + chunk][:, cols].astype(np.float64) @ w[:-1] + w[-1]
        out[s0:s0 + chunk] = 1.0 / (1.0 + np.exp(-z))
    return out


def fit_mlp(X, y, rows, base, hidden=32, epochs=6, lr=2e-3, batch=4096, seed=0, l2=1e-5):
    """1 層の小さな MLP（numpy・Adam・出口に logit_p_hyp を足す＝超幾何からの補正だけを学ぶ）。"""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W1 = rng.normal(0, 1 / math.sqrt(d), (d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1 / math.sqrt(hidden), hidden)
    b2 = np.zeros(1)
    params = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    for _ in range(epochs):
        idx = rows[rng.permutation(len(rows))]
        for s in range(0, len(idx), batch):
            bi = idx[s:s + batch]
            xb, yb = X[bi], y[bi]
            hpre = xb @ W1 + b1
            hh = np.maximum(hpre, 0)
            z = hh @ W2 + b2[0] + base[bi]
            p = 1 / (1 + np.exp(-z))
            dz = (p - yb) / len(bi)
            gW2 = hh.T @ dz + l2 * W2
            gb2 = np.array([dz.sum()])
            dh = np.outer(dz, W2) * (hpre > 0)
            gW1 = xb.T @ dh + l2 * W1
            gb1 = dh.sum(0)
            t += 1
            for i, gr in enumerate([gW1, gb1, gW2, gb2]):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                mh = m[i] / (1 - 0.9 ** t)
                vh = v[i] / (1 - 0.999 ** t)
                params[i] -= lr * mh / (np.sqrt(vh) + 1e-8)
    return params


def pred_mlp(params, X, base):
    W1, b1, W2, b2 = params
    out = np.empty(X.shape[0])
    for s0 in range(0, X.shape[0], 500000):
        z = np.maximum(X[s0:s0 + 500000] @ W1 + b1, 0) @ W2 + b2[0] + base[s0:s0 + 500000]
        out[s0:s0 + 500000] = 1 / (1 + np.exp(-z))
    return out


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _auc(y, s):
    y = np.asarray(y)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = np.asarray(s)[order]
    i = 0
    r = np.arange(1, len(s) + 1, dtype=float)
    # 同値は平均順位
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    ranks[order] = r
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def _scores(y, preds, mask):
    out = {"pairs": int(mask.sum()), "pos_rate": round(float(y[mask].mean()), 4)}
    base_ll = base_br = None
    for k, p in preds.items():
        ll = float(_ll(y[mask], p[mask]).mean())
        br = float(((p[mask] - y[mask]) ** 2).mean())
        out[k] = {"logloss": round(ll, 5), "brier": round(br, 5)}
        if k == "hyp":
            base_ll, base_br = ll, br
        else:
            out[k]["logloss_gain_pct"] = round(100 * (base_ll - ll) / base_ll, 2)
            out[k]["brier_gain_pct"] = round(100 * (base_br - br) / base_br, 2)
    return out


def cmd_analyze(a):
    db = D.load_db()
    ci = CardInfo(db)
    t0 = time.time()
    P, R, names, n_games = build(a.rows, ci)
    print(f"built {len(P['y'])} pairs / {len(R['g'])} rows / {n_games} games {time.time() - t0:.0f}s", flush=True)
    X, cols = design(P, R, names)
    hold_pair = R["hold"][P["row"]].astype(bool)
    tr = ~hold_pair
    tri = np.nonzero(tr)[0]
    mu = X[tri].mean(0, dtype=np.float64)
    sd = X[tri].std(0, dtype=np.float64) + 1e-9
    X -= mu.astype(np.float32)
    X /= sd.astype(np.float32)
    Xs = X
    y = P["y"].astype(float)
    res = {"games": n_games, "rows": int(len(R["g"])), "pairs": int(len(y)),
           "holdout_games": int(len(set(R["g"][R["hold"] == 1].tolist()))),
           "holdout_rows": int(R["hold"].sum()), "features": len(cols)}
    w = fit_logreg(Xs, y, l2=a.l2, rows=tri)
    preds = {"hyp": P["p"], "hyp_known": P["p2"], "logreg": pred_logreg(w, Xs)}
    # 公開情報の効き目の切り分け: 超幾何の logit だけ（較正）／打たなかった手を抜いたもの
    idx_lp = [cols.index("logit_p_hyp")]
    w0 = fit_logreg(Xs, y, l2=a.l2, rows=tri, cols=idx_lp)
    preds["hyp_calibrated"] = pred_logreg(w0, Xs, idx_lp)
    nogap = [i for i, c in enumerate(cols) if not any(s in c for s in (
        "nocounter", "nc_", "countered", "counter_value", "te_", "afford", "blocked", "counter_used",
        "counter_events", "mulligan"))]
    w1 = fit_logreg(Xs, y, l2=a.l2, rows=tri, cols=nogap)
    preds["logreg_no_unplayed"] = pred_logreg(w1, Xs, nogap)
    noknown = [i for i, c in enumerate(cols) if "known" not in c]
    w2 = fit_logreg(Xs, y, l2=a.l2, rows=tri, cols=noknown)
    preds["logreg_no_known"] = pred_logreg(w2, Xs, noknown)
    if not a.no_mlp:
        prm = fit_mlp(Xs, y, tri, _logit(P["p"]), hidden=a.mlp_hidden, epochs=a.mlp_epochs)
        preds["mlp"] = pred_mlp(prm, Xs, _logit(P["p"]))
    print(f"fitted {time.time() - t0:.0f}s", flush=True)
    turn = R["turn"][P["row"]]
    me = R["me"][P["row"]]
    role = P["role"]
    blocks = {}
    for seat_tag, smask in (("seatA_p1", me == "p1"), ("both", np.ones_like(hold_pair))):
        b = {"all": _scores(y, preds, hold_pair & smask)}
        b["train_all"] = _scores(y, preds, tr & smask)
        for tb, tm in (("turn1-2", turn <= 2), ("turn3-4", (turn >= 3) & (turn <= 4)), ("turn5+", turn >= 5)):
            b[tb] = _scores(y, preds, hold_pair & smask & tm)
        for rl in ("counter", "blocker", "removal", "other"):
            b[f"role_{rl}"] = _scores(y, preds, hold_pair & smask & (role == rl))
        blocks[seat_tag] = b
    res["card_presence"] = blocks
    # 係数（標準化した入力に対して）
    order = np.argsort(-np.abs(w[:-1]))
    res["top_coef"] = [{"feature": cols[i], "coef_std": round(float(w[i]), 4)} for i in order[:10]]
    res["all_coef"] = {cols[i]: round(float(w[i]), 4) for i in range(len(cols))}

    # --- 探索に効く量（行単位）
    rh = R["hold"].astype(bool)
    rtr = ~rh
    Xr = np.asarray(R["x"], dtype=float)
    rcols = list(names)
    extra = np.stack([R["exp_counter"], R["known_counter"], R["h"], R["N"]], 1)
    Xr = np.concatenate([Xr, extra], 1)
    rcols += ["exp_counter_hyp", "known_counter", "h", "N"]
    rmu, rsd = Xr[rtr].mean(0), Xr[rtr].std(0) + 1e-9
    Xrs = (Xr - rmu) / rsd
    # カウンター値の合計（1000 単位）: 超幾何の期待値 vs リッジ
    yt = R["truth_counter"]
    Xb = np.concatenate([Xrs, np.ones((len(Xrs), 1))], 1)
    lam = np.eye(Xb.shape[1]) * a.l2
    lam[-1, -1] = 0
    beta = np.linalg.solve(Xb[rtr].T @ Xb[rtr] + lam, Xb[rtr].T @ yt[rtr])
    pred_lin = Xb @ beta
    # 合計を (b) の確率から: Σ_k P̂(k≥1)・counter_k・E[枚数 | ≥1]（超幾何の比で近似）
    lr = preds["logreg"]
    hyp_cnt_given = np.where(P["p"] > 0, (P["n"] * R["h"][P["row"]] / R["N"][P["row"]]) / np.maximum(P["p"], 1e-9), 1)
    sum_from_lr = np.bincount(P["row"], weights=lr * P["counter"] * hyp_cnt_given, minlength=len(R["g"]))

    def err(mask, pred):
        e = pred[mask] - yt[mask]
        return {"mae": round(float(np.abs(e).mean()), 4), "rmse": round(float(np.sqrt((e ** 2).mean())), 4),
                "bias": round(float(e.mean()), 4)}

    cs = {}
    for seat_tag, smask in (("seatA_p1", R["me"] == "p1"), ("both", np.ones(len(rh), bool))):
        blk = {}
        for tb, tm in (("all", np.ones(len(rh), bool)), ("turn1-2", R["turn"] <= 2),
                       ("turn3-4", (R["turn"] >= 3) & (R["turn"] <= 4)), ("turn5+", R["turn"] >= 5)):
            m = rh & smask & tm
            blk[tb] = {"rows": int(m.sum()), "truth_mean": round(float(yt[m].mean()), 4),
                       "hyp": err(m, R["exp_counter"]), "ridge": err(m, pred_lin),
                       "from_logreg": err(m, sum_from_lr)}
        cs[seat_tag] = blk
    res["counter_sum_1000s"] = cs
    # カウンター 1 枚以上の AUC（指示の量）＋ 2000 のカウンター／カウンターイベントの 1 枚以上（補足）
    is_ctr = ((P["counter"] > 0) | (P["counter_event"] > 0)).astype(float)
    targets = (("has_counter", "truth_has", "p_has_counter", is_ctr),
               ("has_2k_counter", "truth_has_2k", "p_has_2k", (P["counter"] >= 2).astype(float)),
               ("has_counter_event", "truth_has_cev", "p_has_cev", P["counter_event"].astype(float)))
    for tname, tkey, pkey, sel in targets:
        yh = R[tkey].astype(float)
        ph_hyp = R[pkey]
        Xh = np.concatenate([Xrs, ((_logit(ph_hyp) - _logit(ph_hyp)[rtr].mean()) /
                                   (_logit(ph_hyp)[rtr].std() + 1e-9))[:, None]], 1)
        wh = fit_logreg(Xh[rtr], yh[rtr], l2=a.l2)
        ph_lr = pred_logreg(wh, Xh)
        any_from_lr = 1 - np.exp(np.bincount(P["row"], weights=np.log(np.clip(1 - lr, 1e-9, 1)) * sel,
                                             minlength=len(R["g"])))
        au = {}
        for seat_tag, smask in (("seatA_p1", R["me"] == "p1"), ("both", np.ones(len(rh), bool))):
            blk = {}
            for tb, tm in (("all", np.ones(len(rh), bool)), ("turn1-2", R["turn"] <= 2),
                           ("turn3-4", (R["turn"] >= 3) & (R["turn"] <= 4)), ("turn5+", R["turn"] >= 5)):
                m = rh & smask & tm
                blk[tb] = {"rows": int(m.sum()), "pos_rate": round(float(yh[m].mean()), 4),
                           "auc_hyp": _auc(yh[m], ph_hyp[m]),
                           "auc_logreg": _auc(yh[m], ph_lr[m]),
                           "auc_from_card_logreg": _auc(yh[m], any_from_lr[m]),
                           "logloss_hyp": round(float(_ll(yh[m], ph_hyp[m]).mean()), 5),
                           "logloss_logreg": round(float(_ll(yh[m], ph_lr[m]).mean()), 5)}
            au[seat_tag] = blk
        res[tname] = au
        hcols = rcols + [f"logit_{pkey}_hyp"]
        ho = np.argsort(-np.abs(wh[:-1]))
        res[f"{tname}_top_coef"] = [{"feature": hcols[i], "coef_std": round(float(wh[i]), 4)} for i in ho[:10]]
    res["sec"] = round(time.time() - t0, 1)
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({"card_presence_seatA": res["card_presence"]["seatA_p1"]["all"],
                      "has_counter_seatA": res["has_counter"]["seatA_p1"]["all"],
                      "counter_sum_seatA": res["counter_sum_1000s"]["seatA_p1"]["all"]}, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--games", type=int, default=2000)
    r.add_argument("--seed-base", type=int, default=3001000)
    r.add_argument("--sims", type=int, default=E.SERVE_SIMS)
    r.add_argument("--net", default=None)
    r.add_argument("--decks", default="synth_roles")
    r.add_argument("--jobs", type=int, default=4)
    r.add_argument("--no-pin", action="store_true")
    r.add_argument("--out", required=True)
    z = sub.add_parser("analyze")
    z.add_argument("--rows", required=True)
    z.add_argument("--out", required=True)
    z.add_argument("--l2", type=float, default=1.0)
    z.add_argument("--no-mlp", action="store_true")
    z.add_argument("--mlp-hidden", type=int, default=32)
    z.add_argument("--mlp-epochs", type=int, default=6)
    a = ap.parse_args(argv)
    if a.cmd == "record":
        E.engine()
        cmd_record(a)
    else:
        cmd_analyze(a)


if __name__ == "__main__":
    main()
