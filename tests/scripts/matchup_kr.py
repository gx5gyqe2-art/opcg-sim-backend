"""強豪の対面記事を物差しに、紫カイドウ対青ロックスで CPU の進行を突き合わせる（WP `matchup-kr`・
`docs/search_efficiency_plan.md` §7.63・台帳 2+466）。

読み取りだけの計測（エンジン・ネット・既定の挙動は変えない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。
土台は `blunder_audit.py`（局の記録の形・`rollout_ship`）と `hold_bias.py`（審判）。

デッキは固定: `tests/fixtures/decks/official_decks_20261022.json` の deck_113（紫カイドウ OP17-058）を **p1**、
deck_112（青ロックス OP17-039）を **p2**。局の番号 i（0〜199）の seed は 4241000＋i、i < 100 は紫カイドウ先攻・
i ≥ 100 は後攻。両席とも r27・出荷の設定（`blunder_audit.ship_spec`）。

1. ``play``: 局を打ち、全ての判断点で打った手・決め方・根の候補の上位 8（訪問・評価・事前分布）を記録する（`games*.jsonl`）。
2. ``trace``: 全局を探索なしで当て直し、手ごとの盤面の要約（手札・場・ドン・ライフ・トラッシュ）と打った手のカードを書く（`trace.jsonl`）。
3. ``items``: 記事の 10 項目の突き合わせと、比べる点（記事の手と CPU の手）の洗い出し（`items.json`・`points_pool.jsonl`）。
4. ``judge``: 比べる点を 1 点 16 通りで打ち比べる（`judge_points*.jsonl`・`--shard i/n`）。
5. ``summary``: 対面の分析と判定の集計（`summary.json`）。

    python tests/scripts/matchup_kr.py play --workers 4 [--shard 0/4]
    python tests/scripts/matchup_kr.py trace
    python tests/scripts/matchup_kr.py items
    python tests/scripts/matchup_kr.py judge --workers 4 [--shard 0/4]
    python tests/scripts/matchup_kr.py summary

再開できる（終えた局・点は飛ばす）。
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blunder_audit as BA  # noqa: E402
import move_judge as MJ  # noqa: E402

from opcg_sim.loop import driver as DR, engine as E  # noqa: E402

OUT_DIR = os.path.join("n1_results", "matchup_kr")
NET = "opcg_sim/data/learned/nrel_r27.npz"
DECKS = os.path.join("tests", "fixtures", "decks", "official_decks_20261022.json")
KAIDO, ROCKS = "deck_113", "deck_112"
SEED0 = 4241000
N_GAMES = 200
#: 判定の配り方の番号のずらし（blunder_audit 0〜2000 番台・hold_bias 3000・small_play 4000・small_whiff と重ねない）。
OFFSET = 7000
_G = {}


def deck(name):
    x = json.load(open(DECKS, encoding="utf-8"))[name]
    cs = []
    for cid, n in x["cards"].items():
        cs += [cid] * int(n)
    return x["leader"], cs


def game_setup(idx):
    """局の番号 → (seed, first_player)。紫カイドウは常に p1。"""
    return SEED0 + idx, ("p1" if idx < N_GAMES // 2 else "p2")


def new_game(idx):
    seed, first = game_setup(idx)
    return DR.new_game(seed, deck(KAIDO), deck(ROCKS), first)


def replay(g, step):
    game = new_game(g["idx"])
    for s, mv in enumerate(g["moves"]):
        if s == step:
            break
        name = json.loads(game.pending_json())["player_id"]
        BA.apply(game, name, mv)
    return game


def read_jsonl(path):
    return BA.read_jsonl(path)


def read_glob(pattern, keyf):
    rows = {}
    for path in sorted(glob.glob(pattern)):
        for x in read_jsonl(path):
            rows[keyf(x)] = x
    return rows


def load_games(out):
    return read_glob(os.path.join(out, "games*.jsonl"), lambda g: g["idx"])


# ---------------------------------------------------------------- 1. play

def top_rows(out, k=8):
    """根の候補（訪問の多い順）に事前分布を添える。"""
    rows = BA.ranked(out)
    st = out.get("stats") or {}
    P = st.get("P") or []
    if out.get("kind") == "main" and out.get("groups"):
        pr = {int(g["rep"]): sum(float(P[i]) for i in g.get("idxs") or [] if i < len(P)) for g in out["groups"]}
    else:
        pr = {i: float(P[i]) for i in range(len(P))}
    return rows, [{"mv": r["mv"], "n": r["n"], "q": round(r["q"], 4), "p": round(pr.get(r["rep"], 0.0), 4)} for r in rows[:k]]


def play_one(idx):
    E.engine()
    seed, first = game_setup(idx)
    spec = BA.ship_spec(NET)
    pts, moves = [], []

    def obs(game, name, turn, step, out, move):
        pend = json.loads(game.pending_json())
        own = game.turn_player() == name
        rows, top = top_rows(out)
        ch = BA.chosen(out, rows) if rows else None
        pts.append({"step": step, "seat": name, "turn": turn, "own": own, "act": pend.get("action"), "kind": out.get("kind"),
                    "dtype": BA.dtype(pend, own, move), "n_cand": len(rows), "move": move,
                    "a_box": ch["mv"] if ch else None, "a_rank": rows.index(ch) if ch else None, "top": top,
                    "commit_left": len(out.get("commit") or [])})
        moves.append(move)

    t0 = time.perf_counter()
    try:
        r = DR.run_game(seed, {"p1": spec, "p2": spec}, deck(KAIDO), deck(ROCKS), first_player=first, observer=obs)
        winner, turns, void = r["winner"], r["turns"], r["winner"] is None
    except DR.GameAborted as exc:
        winner, turns, void = None, None, str(exc)
    return {"idx": idx, "seed": seed, "first": first, "kaido_first": first == "p1", "winner": winner,
            "kaido_won": None if winner is None else winner == "p1", "turns": turns, "void": void,
            "sec": round(time.perf_counter() - t0, 1), "points": pts, "moves": moves}


def shard_of(a):
    return (int(x) for x in a.shard.split("/")) if a.shard else (0, 1)


def cmd_play(a):
    os.makedirs(a.out, exist_ok=True)
    si, sn = shard_of(a)
    gpath = os.path.join(a.out, f"games_shard{si}.jsonl" if a.shard else "games.jsonl")
    done = set(load_games(a.out))
    # 先攻と後攻が交互に進むように並べる（途中で止めても両方の帯がそろう）
    order = [j for pair in zip(range(N_GAMES // 2), range(N_GAMES // 2, N_GAMES)) for j in pair]
    idxs = [i for n, i in enumerate(order) if n % sn == si and i not in done][:a.games or None]
    print(f"PLAY shard={a.shard} todo {len(idxs)} (done {len(done)}) -> {gpath}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers) as pool:
        for i, g in enumerate(pool.imap_unordered(play_one, idxs, chunksize=1)):
            with open(gpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(idxs)}] idx={g['idx']} first={g['first']} turns={g['turns']} kaido_won={g['kaido_won']} "
                  f"sec={g['sec']} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- 2. trace

def uuid_map(h):
    out = {}

    def walk(o):
        if isinstance(o, dict):
            if "uuid" in o and "card_id" in o:
                out[o["uuid"]] = o["card_id"]
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(h)
    return out


def snap(h):
    """盤面の要約（両席）。"""
    out = {}
    for seat in ("p1", "p2"):
        pl = h["players"][seat]
        don = pl.get("don") or {}
        ld = pl["leader"]
        out[seat] = {
            "hand": [c["card_id"] for c in pl["hand"]],
            "field": [[c["card_id"], int(bool(c.get("is_rest"))), int(c.get("attached_don") or 0),
                       int(bool(c.get("is_newly_played"))), c["uuid"]] for c in pl["field"]],
            "leader": [int(bool(ld.get("is_rest"))), int(ld.get("attached_don") or 0)],
            "life": len(pl["life"]), "deck": len(pl["deck"]), "trash": [c.get("card_id") for c in pl.get("trash") or []],
            "stage": [c.get("card_id") for c in (pl.get("stage") if isinstance(pl.get("stage"), list) else
                                                   ([pl["stage"]] if pl.get("stage") else []))],
            "don": [len(don.get("active") or []), len(don.get("rested") or []), len(don.get("attached") or []),
                    len(don.get("deck") or [])]}
    return out


def mv_sum(mv, umap):
    """手の要約（カード番号つき）。"""
    if mv is None:
        return None
    p = mv.get("payload") or {}
    s = {"at": mv.get("action_type")}
    u = p.get("uuid") or mv.get("card_uuid")
    if u:
        s["card"] = umap.get(u, "?")
        s["uuid"] = u
    if p.get("target_ids"):
        s["tgt"] = [umap.get(t, "?") for t in p["target_ids"]]
    if p.get("selected_uuids"):
        s["sel"] = [umap.get(t, "?") for t in p["selected_uuids"]]
    if "accepted" in p:
        s["acc"] = bool(p["accepted"])
    if mv.get("action_type") in ("RESOLVE_EFFECT_SELECTION",) and p.get("index"):
        s["idx"] = p["index"]
    if p.get("don_k"):
        s["k"] = p["don_k"]
    if mv.get("action_type") == "SETUP_BOX" and p.get("base"):
        s["base"] = mv_sum(p["base"], umap)
    return s


def trace_one(g):
    E.engine()
    game = new_game(g["idx"])
    umap = {}
    rows = []
    for s, mv in enumerate(g["moves"]):
        h = json.loads(game.hidden_json())
        umap.update(uuid_map(h))
        pend = json.loads(game.pending_json())
        name = pend["player_id"]
        man = h.get("manager") or {}
        r = {"step": s, "seat": name, "turn": int(game.turn_count or 0), "tp": game.turn_player(), "act": pend.get("action"),
             "msg": (pend.get("message") or "")[:60], "battle": bool(man.get("active_battle")), "st": snap(h),
             "mv": mv_sum(mv, umap)}
        if pend.get("action") in ("MAIN_ACTION",) and name == "p1":
            legal = json.loads(game.legal_json(name))
            r["legal"] = [mv_sum(m, umap) for m in legal if m.get("action_type") in ("PLAY", "ACTIVATE_MAIN", "ATTACK")]
        rows.append(r)
        BA.apply(game, name, mv)
        if game.winner() is not None:
            break
    return {"idx": g["idx"], "kaido_first": g["kaido_first"], "kaido_won": g["kaido_won"], "turns": g["turns"],
            "rows": rows, "umap": umap}


def cmd_trace(a):
    games = load_games(a.out)
    gs = [games[i] for i in sorted(games)]
    path = os.path.join(a.trace_dir or a.out, "trace.jsonl")
    with get_context("fork").Pool(a.workers) as pool, open(path, "w", encoding="utf-8") as f:
        for t in pool.imap(trace_one, gs, chunksize=2):
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print("TRACE games", len(gs), "->", path, flush=True)


# ---------------------------------------------------------------- 3. items

#: 記事の愛称 → カード番号（カード DB の名前とコストで引き当て・報告の表と同じ）。deck_113 に無い札は None。
K6, Q1, Y3, H3 = "EB04-031", "EB04-032", "OP17-074", "OP17-073"
BK, RK, OK, Q9, J9 = "OP17-062", "OP17-063", "OP17-061", "OP17-065", "OP17-069"
D2, UP, S3 = "OP17-075", "OP17-060", "OP17-068"
L10, HIHO, BANRAI = "ST34-004", "OP07-077", "OP15-078"
SHIKI, JOHN, ROCKS_EV, ROCKS10 = "OP17-048", "OP17-044", "OP17-056", "OP17-118"
ACCEL = [Q1, Y3, H3]
BIG = {BK: 10, RK: 10, OK: 9, Q9: 9, J9: 9}
#: 記事の理想の展開（自分の何ターン目 → 出す札の優先順。10リンリンは deck_113 に無いので抜く）。
IDEAL = {1: ACCEL, 2: ACCEL, 3: [K6], 4: [BK], 5: [BK, OK, RK]}
IDEAL_LATE = [OK, RK]
DECIDED = 0.9
NAMES = {}


def cname(cid):
    if not NAMES:
        for x in json.load(open(os.path.join("opcg_sim", "data", "opcg_cards.json"), encoding="utf-8")):
            NAMES[x["number"]] = (x.get("name"), x.get("コスト"), x.get("種類"))
    return NAMES.get(cid, (cid, None, None))


def own_turn(t, seat, kaido_first):
    first = (seat == "p1") == kaido_first
    return (t + 1) // 2 if first else t // 2


def ideal_for(ot):
    return IDEAL.get(ot, IDEAL_LATE if ot >= 6 else [])


def kv(p):
    """記録の点の、紫カイドウから見た評価（打った手の根の値）。"""
    if not p.get("top"):
        return None
    r = p["top"][p["a_rank"]] if p.get("a_rank") is not None and p["a_rank"] < len(p["top"]) else p["top"][0]
    return r["q"] if p["seat"] == "p1" else -r["q"]


def is_bk_attack(mv, uuid):
    """手（原始手か箱）が、その uuid のキャラのアタック（の始まり）か。"""
    if not mv:
        return False
    p = mv.get("payload") or {}
    if mv.get("action_type") == "ATTACK":
        return p.get("uuid") == uuid
    if mv.get("action_type") == "DON_BOX":
        return p.get("uuid") == uuid and bool(p.get("target_ids"))
    if mv.get("action_type") == "SETUP_BOX":
        return is_bk_attack(p.get("base"), uuid)
    return False


def analyse_game(g, t):
    """1 局の項目ごとの場面と、比べる点の候補。"""
    rows = t["rows"]
    pts = g["points"]
    kf = g["kaido_first"]
    won = g["kaido_won"]
    res = {"idx": g["idx"], "kaido_first": kf, "won": won, "turns": g["turns"], "void": bool(g["void"])}
    cands = []

    def cand(item, s, b, why, extra=None):
        p = pts[s]
        cands.append(dict({"idx": g["idx"], "step": s, "item": item, "seat": "p1", "turn": p["turn"],
                           "a": p["a_box"] or p["move"], "b": b, "why": why, "qa": kv(p)}, **(extra or {})))

    # 自分のターンごとの行
    by_turn = {}
    for r in rows:
        if r["tp"] == "p1":
            by_turn.setdefault(r["turn"], []).append(r)
    # (2) マリガン
    for r in rows:
        if r["act"] == "MULLIGAN" and r["seat"] == "p1":
            hand = r["st"]["p1"]["hand"]
            nq = hand.count(Q1)
            if kf:
                crit = (K6 in hand and (Q1 in hand or Y3 in hand)) or (HIHO in hand and any(x in hand for x in ACCEL)) or nq >= 2
            else:
                crit = K6 in hand or nq >= 2 or HIHO in hand
            keep = r["mv"]["at"] == "KEEP_HAND"
            res["mull"] = {"hand": hand, "crit": bool(crit), "keep": keep, "bk_only": BK in hand and not crit}
            if keep != bool(crit):
                b = {"kind": "game", "action_type": "KEEP_HAND" if crit else "MULLIGAN", "payload": {}}
                cand("2", r["step"], b, "keep" if crit else "mulligan")
            break
    # (3) 理想の展開
    res["turns_play"] = {}
    res["ideal"] = {}
    for tn in sorted(by_turn):
        rs = by_turn[tn]
        ot = own_turn(tn, "p1", kf)
        plays = [r["mv"]["card"] for r in rs if r["seat"] == "p1" and r["mv"] and r["mv"]["at"] == "PLAY"]
        res["turns_play"][ot] = plays
        want = ideal_for(ot)
        mains = [r for r in rs if r["seat"] == "p1" and r["act"] == "MAIN_ACTION"]
        legal_cards = {}
        for r in mains:
            for m in r.get("legal") or []:
                if m["at"] == "PLAY" and m.get("card") in want and m["card"] not in legal_cards:
                    legal_cards[m["card"]] = (r["step"], m["uuid"])
        if want and legal_cards:
            top = next(c for c in want if c in legal_cards)
            agree_top = top in plays
            agree_any = any(c in plays for c in legal_cards)
            res["ideal"][ot] = {"want": top, "agree": agree_top, "agree_any": agree_any, "legal": sorted(legal_cards)}
            if not agree_top:
                # 記事の札が合法になった最初の点で比べる
                s0, uu = legal_cards[top]
                b = {"kind": "game", "action_type": "PLAY", "payload": {"uuid": uu}}
                if pts[s0]["kind"] == "main" and BA.key(pts[s0]["a_box"] or pts[s0]["move"]) != BA.key(b):
                    cand("3", s0, b, f"T{ot}:{top}", {"own_turn": ot})
    # (5) リーダー効果
    res["leader"] = []
    k6_played = False
    for r in rows:
        if r["seat"] == "p1" and r["mv"] and r["mv"]["at"] == "PLAY" and r["mv"].get("card") == K6:
            k6_played = True
        if not (r["seat"] == "p1" and r["act"] == "CONFIRM_OPTIONAL" and r["battle"] and r["msg"].startswith("「カイドウ」の効果を使用")):
            continue
        st = r["st"]["p1"]
        bk_on = any(c[0] == BK for c in st["field"])
        total = st["don"][0] + st["don"][1] + st["don"][2]
        nxt_keep, nxt_use = min(10, total + 2), min(10, total + 1)
        big_block = any(nxt_use < BIG[c] <= nxt_keep for c in st["hand"] if c in BIG)
        ot = own_turn(r["turn"], "p1", kf)
        mine = r["tp"] == "p1"
        # 記事の目安を機械の決まりに直したもの（序盤＝自分の 4 ターン目まで。ブロッカーカイドウの「ドンが戻された時に
        # 1 枚追加」は自分のターン中だけ働くので、場にいるから使う、は自分のアタックの時だけ）
        if big_block:
            art, rule = False, "big_next"
        elif not k6_played and not bk_on and ot <= 4:
            art, rule = False, "early"
        elif bk_on and mine:
            art, rule = True, "bk_on"
        else:
            art, rule = None, "free"
        used = bool(r["mv"].get("acc"))
        res["leader"].append({"step": r["step"], "own_turn": ot, "mine": mine, "used": used, "rule": rule,
                              "article": art, "k6_played": k6_played, "bk_on": bk_on, "opp_chars": len(r["st"]["p2"]["field"])})
        if art is not None and art != used:
            p = pts[r["step"]]
            alt = [x["mv"] for x in p["top"] if bool((x["mv"].get("payload") or {}).get("accepted")) == art]
            if alt:
                cand("5", r["step"], alt[0], rule, {"own_turn": ot})
            else:
                cand("5", r["step"], None, rule + ":no_b", {"own_turn": ot})
    # (6) 6キングが手札に戻された後
    res["k6_bounce"] = []
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        if cur["tp"] != "p2":
            continue
        pf = {c[4] for c in prev["st"]["p1"]["field"] if c[0] == K6}
        cf = {c[4] for c in cur["st"]["p1"]["field"] if c[0] == K6}
        gone = pf - cf
        if not gone or cur["st"]["p1"]["hand"].count(K6) <= prev["st"]["p1"]["hand"].count(K6):
            continue
        # 戻した札＝直前の相手の PLAY（イベント・登場時の効果）
        by = next((x["mv"].get("card") for x in reversed(rows[max(0, i - 8):i])
                   if x["seat"] == "p2" and x["mv"] and x["mv"]["at"] in ("PLAY", "ACTIVATE_MAIN")), None)
        # 次の自分のターン
        nt = next((tn for tn in sorted(by_turn) if tn > cur["turn"]), None)
        if nt is None:
            res["k6_bounce"].append({"turn": cur["turn"], "by": by, "next": None})
            continue
        rs = by_turn[nt]
        order = [r["mv"]["card"] for r in rs if r["seat"] == "p1" and r["mv"] and r["mv"]["at"] == "PLAY"]
        rec = {"turn": cur["turn"], "by": by, "next_own_turn": own_turn(nt, "p1", kf), "plays": order,
               "replay_k6": K6 in order}
        first_big = next((c for c in order if c in (K6, BK, OK, RK, Q9, J9)), None)
        bk_ok_legal = False
        for r in rs:
            if r["seat"] != "p1" or r["act"] != "MAIN_ACTION":
                continue
            lg = {m.get("card"): m["uuid"] for m in r.get("legal") or [] if m["at"] == "PLAY"}
            if BK in lg or OK in lg:
                bk_ok_legal = True
                # 記事の手＝ブロッカーカイドウ（無ければ大看板）を先に出す。CPU が先に別の大型を出した・出さなかった点
                if first_big not in (BK, OK) and pts[r["step"]]["kind"] == "main":
                    c = BK if BK in lg else OK
                    cand("6", r["step"], {"kind": "game", "action_type": "PLAY", "payload": {"uuid": lg[c]}},
                         f"first_{c}_not_{first_big}", {"own_turn": own_turn(nt, "p1", kf)})
                break
        rec["bk_ok_legal"] = bk_ok_legal
        rec["agree"] = (first_big in (BK, OK)) if bk_ok_legal else (not rec["replay_k6"])
        res["k6_bounce"].append(rec)
    # (9) シキがいる局面でブロッカーカイドウを殴りに使うか
    res["bk_vs_shiki"] = []
    for tn in sorted(by_turn):
        rs = by_turn[tn]
        mains = [r for r in rs if r["seat"] == "p1" and r["act"] == "MAIN_ACTION"]
        sc = None
        for r in mains:
            if not any(c[0] == SHIKI for c in r["st"]["p2"]["field"]):
                continue
            bks = [m["uuid"] for m in r.get("legal") or [] if m["at"] == "ATTACK" and m.get("card") == BK]
            if bks:
                sc = (r, bks[0])
                break
        if not sc:
            continue
        r0, bu = sc
        att = [r for r in rs if r["seat"] == "p1" and r["mv"] and r["mv"]["at"] == "ATTACK" and r["mv"].get("uuid") == bu]
        ot = own_turn(tn, "p1", kf)
        res["bk_vs_shiki"].append({"own_turn": ot, "attacked": bool(att), "opp_life": r0["st"]["p2"]["life"],
                                   "my_life": r0["st"]["p1"]["life"]})
        if att:
            s_att = att[0]["step"]
            # 箱で始まった攻撃は、その箱を選んだ点に戻る
            sp = next((r["step"] for r in mains if r["step"] <= s_att and pts[r["step"]]["kind"] == "main"
                       and is_bk_attack(pts[r["step"]]["a_box"] or pts[r["step"]]["move"], bu)), None)
            if sp is None:
                continue
            p = pts[sp]
            alt = [x["mv"] for x in p["top"] if not is_bk_attack(x["mv"], bu)]
            b = alt[0] if alt else {"kind": "game", "action_type": "TURN_END", "payload": {}}
            cand("9", sp, b, "bk_attack" + ("" if alt else ":turn_end_fill"), {"own_turn": ot})
    # 相手のシキのアタックをブロッカーカイドウが止めたか
    res["shiki_attacks"] = []
    for i, r in enumerate(rows):
        if r["seat"] == "p1" and r["act"] == "SELECT_BLOCKER":
            # 直前の相手のアタック
            j = i - 1
            while j >= 0 and not (rows[j]["seat"] == "p2" and rows[j]["mv"] and rows[j]["mv"]["at"] == "ATTACK"):
                j -= 1
            if j >= 0 and rows[j]["mv"].get("card") == SHIKI:
                res["shiki_attacks"].append({"blocked_by": r["mv"].get("card"), "bk_avail": any(
                    c[0] == BK and not c[1] for c in r["st"]["p1"]["field"])})
    # (10) 終盤
    res["rk"] = []
    for i, r in enumerate(rows):
        if r["seat"] == "p1" and r["mv"] and r["mv"]["at"] == "PLAY" and r["mv"].get("card") == RK:
            u = r["mv"]["uuid"]
            opp = [c[0] for c in rows[i + 1]["st"]["p2"]["field"]] if i + 1 < len(rows) else []
            tn = r["turn"]
            act = [x for x in rows[i:] if x["turn"] == tn and x["seat"] == "p1" and x["mv"] and x["mv"]["at"] == "ACTIVATE_MAIN"
                   and x["mv"].get("uuid") == u]
            tgt = None
            if act:
                k = rows.index(act[0])
                sel = next((x for x in rows[k:k + 4] if x["act"] == "SEARCH_AND_SELECT"), None)
                tgt = (sel["mv"].get("sel") or [None])[0] if sel else None
                # 記事の手＝キャプテン・ジョンを対象にする（いるのに別の札を選んだ点）
                if sel and JOHN in opp and tgt != JOHN and pts[sel["step"]]["kind"] == "window":
                    um = t["umap"]
                    alt = [x["mv"] for x in pts[sel["step"]]["top"]
                           if [um.get(u) for u in (x["mv"].get("payload") or {}).get("selected_uuids") or []] == [JOHN]]
                    cand("10", sel["step"], alt[0] if alt else None, "rk_john" + ("" if alt else ":no_b"),
                         {"own_turn": own_turn(tn, "p1", kf)})
            res["rk"].append({"own_turn": own_turn(tn, "p1", kf), "used": bool(act), "target": tgt,
                              "john": JOHN in opp, "opp_low": sum(1 for c in opp if int(cname(c)[1] or 99) <= 6)})
    res["d2"] = []
    for r in rows:
        if r["seat"] == "p1" and r["mv"] and r["mv"]["at"] == "PLAY" and r["mv"].get("card") == D2:
            res["d2"].append({"own_turn": own_turn(r["turn"], "p1", kf), "opp_hand": len(r["st"]["p2"]["hand"]),
                              "mine": r["tp"] == "p1"})
    # 相手の手札が少ない（2 枚以下）自分のターンで 2ドレークを出せたのに出さなかった
    res["d2_chance"] = []
    for tn in sorted(by_turn):
        rs = by_turn[tn]
        m0 = next((r for r in rs if r["seat"] == "p1" and r["act"] == "MAIN_ACTION"
                   and any(m["at"] == "PLAY" and m.get("card") == D2 for m in r.get("legal") or [])), None)
        if m0 and len(m0["st"]["p2"]["hand"]) <= 2:
            res["d2_chance"].append({"opp_hand": len(m0["st"]["p2"]["hand"]),
                                     "played": any(r["mv"] and r["mv"]["at"] == "PLAY" and r["mv"].get("card") == D2
                                                   for r in rs if r["seat"] == "p1")})
    # 青ロックス側の進行
    res["rocks_play"] = {}
    for r in rows:
        if r["tp"] == "p2" and r["seat"] == "p2" and r["mv"] and r["mv"]["at"] == "PLAY":
            ot = own_turn(r["turn"], "p2", kf)
            res["rocks_play"].setdefault(ot, []).append(r["mv"]["card"])
    # ロックス海賊団（メイン）で何を戻したか
    res["rocks_bounce"] = []
    for i, r in enumerate(rows):
        if r["seat"] == "p2" and r["mv"] and r["mv"]["at"] == "PLAY" and r["mv"].get("card") == ROCKS_EV and r["tp"] == "p2":
            sel = next((x for x in rows[i + 1:i + 5] if x["seat"] == "p2" and x["act"] == "SEARCH_AND_SELECT"), None)
            res["rocks_bounce"].append({"own_turn": own_turn(r["turn"], "p2", kf),
                                        "target": (sel["mv"].get("sel") or [None])[0] if sel else None})
    # 評価の推移（紫カイドウから見た・判断点ごと）と相手の札の登場の前後
    vals = [(p["step"], p["turn"], kv(p)) for p in pts if p["kind"] in ("main", "window") and kv(p) is not None]
    res["values"] = vals
    # 相手のターンの前後（紫カイドウ自身の評価だけで比べる＝読む側ごとの評価の癖を入れない）:
    # 前＝その直前の自分のターンの最後のメインの判断、後＝次の自分のターンの最初のメインの判断
    mains = [(p["step"], p["turn"], kv(p)) for p in pts if p["seat"] == "p1" and p["kind"] == "main"
             and p["dtype"] != "mulligan" and kv(p) is not None]
    res["opp_turns"] = []
    for tn in sorted({r["turn"] for r in rows if r["tp"] == "p2"}):
        bef = [v for s_, t_, v in mains if t_ < tn]
        aft = [v for s_, t_, v in mains if t_ > tn]
        if not bef or not aft:
            continue
        cards = sorted({r["mv"]["card"] for r in rows if r["turn"] == tn and r["seat"] == "p2" and r["mv"]
                        and r["mv"]["at"] == "PLAY"})
        res["opp_turns"].append({"turn": tn, "own_turn": own_turn(tn, "p2", kf), "before": bef[-1], "after": aft[0],
                                 "cards": cards})
    return res, cands


def _rank(idx, step):
    return E.search_seed(int(SEED0 + idx) ^ 0x3A7C_4D00, int(step), "p1")


def cmd_items(a):
    games = load_games(a.out)
    path = os.path.join(a.trace_dir or a.out, "trace.jsonl")
    traces = {t["idx"]: t for t in read_jsonl(path)}
    per, cands = [], []
    for i in sorted(games):
        if i not in traces:
            continue
        r, c = analyse_game(games[i], traces[i])
        per.append(r)
        cands += c
    with open(os.path.join(a.out, "items_games.jsonl"), "w", encoding="utf-8") as f:
        for r in per:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 比べる点: 項目ごとに局の記録だけで決まる順位で並べ、項目を順に回して上限まで
    by = {}
    for c in cands:
        # 勝負がほぼ決まった局面（CPU の根の値の絶対値 ≥ 0.9）は打ち比べても差が出ないので外す（数は items に書く）
        if c["b"] is not None and (c["qa"] is None or abs(c["qa"]) < DECIDED):
            by.setdefault(c["item"], []).append(c)
    for k in by:
        by[k].sort(key=lambda c: _rank(c["idx"], c["step"]))
    quota = {"2": 8}
    order = ["3", "5", "6", "9", "10", "2"]
    pool, used = [], {k: 0 for k in order}
    while len(pool) < a.cap and any(used[k] < min(len(by.get(k, [])), quota.get(k, 10 ** 6)) for k in order):
        for k in order:
            if len(pool) >= a.cap:
                break
            if used[k] < min(len(by.get(k, [])), quota.get(k, 10 ** 6)):
                pool.append(by[k][used[k]])
                used[k] += 1
    # 点の表は判定のシャードに配った後は固定する（175 局の時点で選んだ・後から届いた局は項目の数にだけ足す）
    ppath = os.path.join(a.out, "points_pool.jsonl")
    if os.path.exists(ppath) and not a.repool:
        ppath = os.path.join(a.out, "points_pool_latest.jsonl")
    with open(ppath, "w", encoding="utf-8") as f:
        for c in pool:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("ITEMS games", len(per), "cands", {k: len(v) for k, v in by.items()},
          "no_b", sum(1 for c in cands if c["b"] is None),
          "decided", sum(1 for c in cands if c["b"] is not None and c["qa"] is not None and abs(c["qa"]) >= DECIDED),
          "pool", used, flush=True)
    with open(os.path.join(a.out, "cands_all.jsonl"), "w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps({k: v for k, v in c.items() if k not in ("a", "b")}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 4. judge

def _init_judge(k):
    E.engine()
    BA._G["ship"] = BA.ship_spec(NET)
    _G["ref"] = E.SeatSpec(NET, sims=MJ.REFEREE_SIMS)
    _G["k"] = int(k)


def judge_point(job):
    g, c = job
    t0 = time.time()
    row = dict(c, ok=False, fail=None)
    try:
        game = replay(g, c["step"])
    except Exception as exc:  # noqa: BLE001
        row["fail"] = f"replay: {type(exc).__name__}"
        return row
    pend = json.loads(game.pending_json())
    if pend.get("player_id") != "p1" or int(game.turn_count or 0) != c["turn"]:
        row["fail"] = "seat/turn"
        return row
    a, b = c["a"], c["b"]
    row["a_desc"], row["b_desc"] = BA.desc(game, a), BA.desc(game, b)
    p = g["points"][c["step"]]
    row["top"] = [{"d": BA.desc(game, t["mv"]), "n": t["n"], "q": t["q"], "p": t.get("p")} for t in p["top"]]
    # 根の数字（CPU の手と記事の手）
    tot = sum(t["n"] for t in p["top"]) or 1.0

    def root(mv):
        hit = [t for t in p["top"] if BA.key(t["mv"]) == BA.key(mv)] or [t for t in p["top"] if BA.pkey(t["mv"]) == BA.pkey(mv)]
        return {"n": hit[0]["n"], "share": round(hit[0]["n"] / tot, 4), "q": hit[0]["q"], "p": hit[0].get("p")} if hit else None
    row["root_a"], row["root_b"] = root(a), root(b)
    wa, wb, errs = [], [], {}
    for i in range(_G["k"]):
        ws = MJ.world_seed(SEED0 + g["idx"], c["step"], OFFSET + i)
        ra, ea = BA.rollout_ship(game, "p1", a, ws, _G["ref"], BA._G["ship"], both=True)
        rb, eb = BA.rollout_ship(game, "p1", b, ws, _G["ref"], BA._G["ship"], both=True)
        wa.append(ra)
        wb.append(rb)
        for e in (ea, eb):
            if e:
                errs[e] = errs.get(e, 0) + 1
    # paired は b − a ＝「記事の手 − CPU の手」
    row["ref"] = dict(MJ.paired(wa, wb), a=wa, b=wb, errs=errs or None)
    row["ok"] = row["ref"]["diff"] is not None
    if not row["ok"]:
        row["fail"] = "no_pairs"
    row["sec"] = round(time.time() - t0, 1)
    return row


def cmd_judge(a):
    games = load_games(a.out)
    pool = read_jsonl(os.path.join(a.out, "points_pool.jsonl"))
    si, sn = shard_of(a)
    jobs = [c for n, c in enumerate(pool) if n % sn == si]
    if a.limit:
        jobs = jobs[:a.limit]
    rpath = os.path.join(a.out, f"judge_points_shard{si}.jsonl" if a.shard else "judge_points.jsonl")
    done = set(read_glob(os.path.join(a.out, "judge_points*.jsonl"), lambda x: (x["idx"], x["step"], x["item"])))
    jobs = [(games[c["idx"]], c) for c in jobs if (c["idx"], c["step"], c["item"]) not in done]
    print(f"JUDGE shard={a.shard} todo {len(jobs)} (done {len(done)}) k={a.k} -> {rpath}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init_judge, initargs=(a.k,)) as pool_:
        for i, r in enumerate(pool_.imap_unordered(judge_point, jobs, chunksize=1)):
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            ref = r.get("ref") or {}
            print(f"[{i + 1}/{len(jobs)}] idx={r['idx']} step={r['step']} item={r['item']} ok={r['ok']} "
                  f"diff={ref.get('diff')} ci={ref.get('ci')} sec={r.get('sec')} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- 5. summary

def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(c - h, 3), round(c + h, 3)]


def rate(xs):
    xs = [bool(x) for x in xs if x is not None]
    return {"n": len(xs), "k": sum(xs), "rate": round(sum(xs) / len(xs), 3) if xs else None, "ci": wilson(sum(xs), len(xs))}


def mean_ci(ds):
    n = len(ds)
    if n == 0:
        return {"n": 0, "mean": None, "ci": [None, None]}
    m = sum(ds) / n
    v = sum((x - m) ** 2 for x in ds) / (n - 1) if n > 1 else 0.0
    h = 1.96 * (v / n) ** 0.5 if n > 1 else None
    return {"n": n, "mean": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)] if h is not None else [None, None]}


def top_counts(lists, k=6):
    c = {}
    for xs in lists:
        for x in xs:
            c[x] = c.get(x, 0) + 1
    return [[x, cname(x)[0], n] for x, n in sorted(c.items(), key=lambda kv: -kv[1])[:k]]


def cmd_summary(a):
    per = [r for r in read_jsonl(os.path.join(a.out, "items_games.jsonl"))]
    ok = [r for r in per if not r["void"] and r["won"] is not None]
    S = {"games": len(per), "void": len(per) - len(ok)}
    # (1) 先後
    S["winrate"] = {"all": rate([r["won"] for r in ok]),
                    "first": rate([r["won"] for r in ok if r["kaido_first"]]),
                    "second": rate([r["won"] for r in ok if not r["kaido_first"]]),
                    "turns_mean": round(sum(r["turns"] for r in ok) / max(1, len(ok)), 2)}
    # (2) マリガン
    mm = [r for r in ok if r.get("mull")]
    S["mulligan"] = {}
    for tag, sub in (("first", [r for r in mm if r["kaido_first"]]), ("second", [r for r in mm if not r["kaido_first"]])):
        S["mulligan"][tag] = {
            "agree": rate([r["mull"]["keep"] == r["mull"]["crit"] for r in sub]),
            "crit_in": rate([r["mull"]["crit"] for r in sub]),
            "cpu_keep": rate([r["mull"]["keep"] for r in sub]),
            "keep_when_in": rate([r["mull"]["keep"] for r in sub if r["mull"]["crit"]]),
            "keep_when_out": rate([r["mull"]["keep"] for r in sub if not r["mull"]["crit"]]),
            "win_in": rate([r["won"] for r in sub if r["mull"]["crit"]]),
            "win_out": rate([r["won"] for r in sub if not r["mull"]["crit"]]),
            "win_keep": rate([r["won"] for r in sub if r["mull"]["keep"]]),
            "win_mull": rate([r["won"] for r in sub if not r["mull"]["keep"]]),
            "bk_only": rate([r["mull"]["bk_only"] for r in sub]),
            "keep_bk_only": rate([r["mull"]["keep"] for r in sub if r["mull"]["bk_only"]])}
    # (3) 理想の展開
    S["ideal"] = {}
    for tag, sub in (("first", [r for r in ok if r["kaido_first"]]), ("second", [r for r in ok if not r["kaido_first"]])):
        rows = {}
        for ot in range(1, 9):
            k = str(ot)
            app = [r["ideal"][k] for r in sub if k in r["ideal"]]
            reached = [r for r in sub if k in r["turns_play"]]
            rows[k] = {"reached": len(reached), "applicable": len(app), "agree": rate([x["agree"] for x in app]),
                       "agree_any": rate([x["agree_any"] for x in app]),
                       "want": top_counts([[x["want"]] for x in app], 3),
                       "plays_all": top_counts([r["turns_play"][k] for r in reached]),
                       "plays_won": top_counts([r["turns_play"][k] for r in reached if r["won"]]),
                       "none_played": rate([not r["turns_play"][k] for r in reached])}
        on = [r for r in sub if all(r["ideal"][k]["agree"] for k in r["ideal"] if int(k) <= 5) and
              any(int(k) <= 5 for k in r["ideal"])]
        off = [r for r in sub if r not in on]
        S["ideal"][tag] = {"by_turn": rows, "on_track": rate([r in on for r in sub]),
                           "win_on": rate([r["won"] for r in on]), "win_off": rate([r["won"] for r in off]),
                           "k6_by_t3": rate([K6 in sum((r["turns_play"].get(str(t), []) for t in (1, 2, 3)), []) for r in sub]),
                           "win_k6_by_t3": rate([r["won"] for r in sub if K6 in sum((r["turns_play"].get(str(t), []) for t in (1, 2, 3)), [])]),
                           "win_no_k6_by_t3": rate([r["won"] for r in sub if K6 not in sum((r["turns_play"].get(str(t), []) for t in (1, 2, 3)), [])])}
    # (5) リーダー効果
    L = [x for r in ok for x in r["leader"]]
    S["leader"] = {"all": len(L), "used": rate([x["used"] for x in L]),
                   "used_mine": rate([x["used"] for x in L if x["mine"]]),
                   "used_opp": rate([x["used"] for x in L if not x["mine"]]),
                   "by_rule": {rl: {"n": len([x for x in L if x["rule"] == rl]),
                                    "used": rate([x["used"] for x in L if x["rule"] == rl]),
                                    "agree": rate([x["used"] == x["article"] for x in L if x["rule"] == rl and x["article"] is not None])}
                               for rl in ("early", "big_next", "bk_on", "free")},
                   "used_by_own_turn": {str(t): rate([x["used"] for x in L if x["own_turn"] == t]) for t in range(1, 9)},
                   "first_use_turn": top_counts([[str(min(x["own_turn"] for x in r["leader"] if x["used"]))]
                                                 for r in ok if any(x["used"] for x in r["leader"])], 8),
                   "games_used_before_k6": rate([any(x["used"] and not x["k6_played"] and x["own_turn"] <= 4 for x in r["leader"]) for r in ok])}
    # (6) 6キングが返された後
    B = [x for r in ok for x in r["k6_bounce"] if x.get("next_own_turn")]
    S["k6_bounce"] = {"events": sum(len(r["k6_bounce"]) for r in ok), "with_next": len(B),
                      "by": top_counts([[x["by"]] for x in B if x["by"]], 5),
                      "games": rate([bool(r["k6_bounce"]) for r in ok]),
                      "win_bounced": rate([r["won"] for r in ok if r["k6_bounce"]]),
                      "win_not_bounced": rate([r["won"] for r in ok if not r["k6_bounce"]]),
                      "agree": rate([x["agree"] for x in B]), "replay_k6": rate([x["replay_k6"] for x in B]),
                      "bk_ok_legal": rate([x["bk_ok_legal"] for x in B]),
                      "agree_when_bk_ok_legal": rate([x["agree"] for x in B if x["bk_ok_legal"]]),
                      "next_plays": top_counts([x["plays"] for x in B])}
    # (9) シキとブロッカーカイドウ
    V = [x for r in ok for x in r["bk_vs_shiki"]]
    SA = [x for r in ok for x in r["shiki_attacks"]]
    S["bk_vs_shiki"] = {"scenes": len(V), "kept": rate([not x["attacked"] for x in V]),
                        "shiki_attacks_seen": len(SA), "blocked_by_bk_when_avail": rate([x["blocked_by"] == BK for x in SA if x["bk_avail"]]),
                        "bk_avail": rate([x["bk_avail"] for x in SA])}
    # (10) 終盤
    R = [x for r in ok for x in r["rk"]]
    D = [x for r in ok for x in r["d2"]]
    DC = [x for r in ok for x in r["d2_chance"]]
    S["late"] = {"rk_played": len(R), "rk_used": rate([x["used"] for x in R]),
                 "rk_used_when_target": rate([x["used"] for x in R if x["opp_low"] > 0]),
                 "rk_john_present": len([x for x in R if x["john"]]),
                 "rk_hit_john_when_present": rate([x["target"] == JOHN for x in R if x["john"] and x["used"]]),
                 "rk_targets": top_counts([[x["target"]] for x in R if x["target"]], 6),
                 "d2_played": len(D), "d2_opp_hand": top_counts([[str(x["opp_hand"])] for x in D], 8),
                 "d2_chance_small_hand": rate([x["played"] for x in DC])}
    # 青ロックス側
    S["rocks"] = {str(t): top_counts([r["rocks_play"].get(str(t), []) for r in ok], 5) for t in range(1, 8)}
    RB = [x for r in ok for x in r["rocks_bounce"]]
    S["rocks_bounce"] = {"n": len(RB), "targets": top_counts([[x["target"]] for x in RB if x["target"]], 6),
                         "by_turn": top_counts([[str(x["own_turn"])] for x in RB], 8)}
    # 厄介な札（相手の札の登場の前後で動いた紫カイドウの評価）
    # 勝負がほぼ決まった局面（前の値の絶対値 ≥ 0.9）は外す。札ごとに「その札を出した相手のターン」の下がり幅の平均から、
    # 全ての相手のターンの平均を引いた値（負＝その札が出たターンは紫カイドウの評価がふだんより下がる）
    OT = [x for r in ok for x in r["opp_turns"] if abs(x["before"]) < DECIDED]
    # 基準は相手の同じ「何ターン目」の平均（序盤の札と終盤の札を同じ物差しで比べるため）
    bt = {}
    for x in OT:
        bt.setdefault(min(x["own_turn"], 8), []).append(x["after"] - x["before"])
    base = {k: sum(v) / len(v) for k, v in bt.items()}
    oe = {}
    for x in OT:
        for c in x["cards"]:
            oe.setdefault(c, []).append(x["after"] - x["before"] - base[min(x["own_turn"], 8)])
    S["opp_turn_base"] = {"n": len(OT), "by_own_turn": {str(k): [len(bt[k]), round(base[k], 4)] for k in sorted(bt)}}
    S["opp_cards"] = sorted([[c, cname(c)[0], len(v)] + [mean_ci(v)["mean"], mean_ci(v)["ci"]] for c, v in oe.items() if len(v) >= 8],
                            key=lambda x: x[3])
    S["opp_turn_none"] = mean_ci([x["after"] - x["before"] - base[min(x["own_turn"], 8)] for x in OT if not x["cards"]])
    # 判定
    J = [x for x in read_glob(os.path.join(a.out, "judge_points*.jsonl"), lambda x: (x["idx"], x["step"], x["item"])).values() if x["ok"]]
    S["judge"] = {"points": len(J), "by_item": {}, "all": mean_ci([x["ref"]["diff"] for x in J])}
    for it in sorted({x["item"] for x in J}):
        sub = [x for x in J if x["item"] == it]
        S["judge"]["by_item"][it] = dict(mean_ci([x["ref"]["diff"] for x in sub]),
                                        article_better=sum(1 for x in sub if x["ref"]["ci"][0] is not None and x["ref"]["ci"][0] > 0),
                                        cpu_better=sum(1 for x in sub if x["ref"]["ci"][1] is not None and x["ref"]["ci"][1] < 0),
                                        by_why={w: mean_ci([x["ref"]["diff"] for x in sub if x["why"].split(":")[0] == w])
                                                for w in sorted({x["why"].split(":")[0] for x in sub})})
    S["judge"]["rows"] = sorted([{"idx": x["idx"], "step": x["step"], "item": x["item"], "why": x["why"], "own_turn": x.get("own_turn"),
                                  "a": x["a_desc"], "b": x["b_desc"], "diff": x["ref"]["diff"], "ci": x["ref"]["ci"],
                                  "wa": x["ref"]["wa"], "wb": x["ref"]["wb"], "root_a": x.get("root_a"), "root_b": x.get("root_b"),
                                  "qa": x.get("qa")} for x in J], key=lambda x: (x["item"], -(x["diff"] or 0)))
    cands = read_jsonl(os.path.join(a.out, "cands_all.jsonl"))
    S["cands"] = {it: {"all": len([c for c in cands if c["item"] == it]),
                       "no_b": len([c for c in cands if c["item"] == it and c["why"].endswith("no_b")]),
                       "decided": len([c for c in cands if c["item"] == it and c["qa"] is not None and abs(c["qa"]) >= DECIDED])}
                  for it in sorted({c["item"] for c in cands})}
    json.dump(S, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: S[k] for k in ("games", "void", "winrate")}, ensure_ascii=False), flush=True)

# ---------------------------------------------------------------- 6. story（代表局の振り返り）

def story_one(g, t, ig):
    rows, pts = t["rows"], g["points"]
    kf = g["kaido_first"]
    turns = {}
    for r in rows:
        tn = r["turn"]
        d = turns.setdefault(tn, {"turn": tn, "who": "紫" if r["tp"] == "p1" else "青", "own_turn": own_turn(tn, r["tp"], kf),
                                   "plays": [], "attacks": 0, "life": None})
        if r["seat"] == r["tp"] and r["mv"] and r["mv"]["at"] in ("PLAY", "ACTIVATE_MAIN"):
            d["plays"].append(("起動:" if r["mv"]["at"] == "ACTIVATE_MAIN" else "") + (cname(r["mv"].get("card"))[0] or "?"))
        if r["seat"] == r["tp"] and r["mv"] and r["mv"]["at"] == "ATTACK":
            d["attacks"] += 1
        d["life"] = [r["st"]["p1"]["life"], r["st"]["p2"]["life"]]
        d["hand"] = [len(r["st"]["p1"]["hand"]), len(r["st"]["p2"]["hand"])]
        d["field"] = [[cname(c[0])[0] for c in r["st"]["p1"]["field"]], [cname(c[0])[0] for c in r["st"]["p2"]["field"]]]
    vals = ig["values"]
    for tn, d in turns.items():
        vs = [v for s_, t_, v in vals if t_ == tn]
        d["value_end"] = round(vs[-1], 3) if vs else None
    # 紫カイドウの判断で、打った手の値と次の自分の判断の値の差が最も大きく下がった点
    mine = [(p["step"], kv(p)) for p in pts if p["seat"] == "p1" and p["kind"] == "main" and p["dtype"] != "mulligan"
            and kv(p) is not None]
    drops = []
    for (s0, v0), (s1, v1) in zip(mine, mine[1:]):
        if pts[s0]["turn"] != pts[s1]["turn"]:
            drops.append((v1 - v0, s0))
    worst = sorted(drops)[:2]
    keys = []
    for dv, s0 in worst:
        p = pts[s0]
        game = replay(g, s0)
        keys.append({"step": s0, "turn": p["turn"], "own_turn": own_turn(p["turn"], "p1", kf), "drop_to_next_turn": round(dv, 3),
                     "chosen": BA.desc(game, p["a_box"] or p["move"]),
                     "top": [{"d": BA.desc(game, x["mv"]), "n": x["n"], "q": x["q"], "p": x.get("p")} for x in p["top"][:5]],
                     "hand": [cname(c)[0] for c in rows[s0]["st"]["p1"]["hand"]],
                     "field": [cname(c[0])[0] for c in rows[s0]["st"]["p1"]["field"]],
                     "opp_field": [cname(c[0])[0] for c in rows[s0]["st"]["p2"]["field"]],
                     "don": rows[s0]["st"]["p1"]["don"], "life": [rows[s0]["st"]["p1"]["life"], rows[s0]["st"]["p2"]["life"]],
                     "opp_hand": len(rows[s0]["st"]["p2"]["hand"]),
                     # そのターンの紫カイドウの判断（打った手・評価・次点）
                     "turn_decisions": [{"step": q["step"], "chosen": BA.desc(replay(g, q["step"]), q["a_box"] or q["move"]),
                                         "q": kv(q), "share": round(q["top"][q["a_rank"] or 0]["n"] / (sum(x["n"] for x in q["top"]) or 1), 3),
                                         "alt": [{"d": BA.desc(replay(g, q["step"]), x["mv"]), "q": x["q"], "n": x["n"]}
                                                 for x in q["top"] if BA.key(x["mv"]) != BA.key(q["a_box"] or q["move"])][:2]}
                                        for q in pts if q["seat"] == "p1" and q["kind"] == "main" and q["turn"] == p["turn"]
                                        and len(q["top"]) > 1]})
    return {"idx": g["idx"], "kaido_first": kf, "won": g["kaido_won"], "turns": [turns[k] for k in sorted(turns)], "key_points": keys}


def cmd_story(a):
    E.engine()
    games = load_games(a.out)
    traces = {t["idx"]: t for t in read_jsonl(os.path.join(a.trace_dir or a.out, "trace.jsonl"))}
    per = {r["idx"]: r for r in read_jsonl(os.path.join(a.out, "items_games.jsonl"))}
    ok = [r for r in per.values() if not r["void"] and r["won"] is not None]

    def swing(r):
        vs = [v for _, _, v in r["values"]]
        return max(vs) - vs[-1] if vs else 0

    pick = []
    # 1. 記事どおりに 3 ターン目に 6キングを出して勝った先攻の局（局の長さが中ほど）
    w = sorted([r for r in ok if r["won"] and r["kaido_first"] and K6 in r["turns_play"].get("3", [])],
               key=lambda r: abs(r["turns"] - 13))
    if w:
        pick.append(("ideal_win", w[0]["idx"]))
    # 2. 一度は優勢（評価 +0.3 以上）だったのに負けた局のうち、落ち方の最も大きい局
    l = sorted([r for r in ok if not r["won"]], key=lambda r: -swing(r))
    if l:
        pick.append(("comeback_loss", l[0]["idx"]))
    # 3. 6キングをロックス海賊団で返された局
    b = [r for r in ok if r["k6_bounce"] and r["idx"] not in [x for _, x in pick]]
    if b:
        pick.append(("k6_bounced", sorted(b, key=lambda r: r["idx"])[0]["idx"]))
    out = [dict(story_one(games[i], traces[i], per[i]), tag=tag) for tag, i in pick]
    json.dump(out, open(os.path.join(a.out, "stories.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("STORY", pick, flush=True)


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["play", "trace", "items", "judge", "summary", "story"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--cap", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--repool", action="store_true")
    a = ap.parse_args(argv)
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
