"""「出す前から効果が働かないと分かっていた」25 回は手札の状況から仕方なかったか（WP `small-whiff`・
`docs/search_efficiency_plan.md` §7.62・台帳 2+465）。

読み取りだけの計測（エンジン・ネット・既定の挙動は変えない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。
対象は §7.61（`small-use`）の「登場時が見えていた空振り」25 回（`n1_results/small_use/labels.jsonl`）。局は §7.59 の 150 局。

1. ``cases``: 各回の局面を当て直し、自分の手札・場・ドン・ライフ・相手の場とライフ・このターンに先に打った手、
   **根の候補**（出荷の設定で同じ局面を読み直した上位 8・訪問数と評価つき）と記録の上位 3 を残す。代わりの手を機械的に挙げる:
   (a) 根の次点・(b) このキャラ（同じカード）を出さない最上位の手・(c) ターンの最初の判断点で先にこのキャラを出していれば
   効果が働いたか（出す順番）・(d) 手札の他のキャラ（今出せるもの）を出したら登場時の効果が働くか。
   出力 `cases.jsonl`・`cases.md`（人が読む形）。
2. ``judge``: 打った手と (b)（無ければ (a)）を対に、§7.60 と同じ判定（公平な配り直し・今のターンの残りは両席とも出荷の設定・
   その先は審判〔r27・160 回・世界 1 本〕）で 1 回 16 通り。§7.60 で同じ b を判定済みの点は 8 通りを足して 16 にする
   （配り方の番号は §7.60 と同じ並びの続き）。``--runner`` で (a) が (b) と違う回の打った手と (a) の対も足す。
3. ``summary``: 「打った手 − 代わりの手」（`move_judge.paired` は b − a なので符号を反転）で回ごとの差と 95% 区間、
   3 分類（仕方ない／違和感あり・損／違和感あり・損は見えない）の機械の下書きを `summary.json` に書く。

    python tests/scripts/small_whiff.py cases
    python tests/scripts/small_whiff.py judge --workers 4
    python tests/scripts/small_whiff.py summary
"""
import argparse
import json
import os
import sys
import time
import unicodedata
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blunder_audit as BA  # noqa: E402
import hold_bias as HB  # noqa: E402
import move_judge as MJ  # noqa: E402
import small_play as SP  # noqa: E402
import small_use as SU  # noqa: E402

from opcg_sim.loop import decks as D, engine as E  # noqa: E402

GAMES = SP.GAMES
OUT_DIR = os.path.join("n1_results", "small_whiff")
NET = HB.NET
K = 16
_G = {}

WHY_JA = {"cost_unpayable": "出す代金を払うと効果のコストが払えない", "no_target": "対象になるキャラがいない",
          "cond_unmet": "条件を満たさない", "negated": "相手のリーダーの効果で登場時が無効", "leader_cond": "リーダーの条件"}


# ---------------------------------------------------------------- 共通

def raw_db():
    if "raw" not in _G:
        _G["raw"] = {x["number"]: x for x in json.load(open(os.path.join("opcg_sim", "data", "opcg_cards.json"), encoding="utf-8"))}
    return _G["raw"]


def cdesc(cid, full=False):
    x = raw_db().get(cid) or {}
    t = (x.get("効果(テキスト)") or "").strip()
    if not full and len(t) > 90:
        t = t[:90] + "…"
    kind = {"キャラクター": "キャラ", "イベント": "イベント", "ステージ": "ステージ"}.get(x.get("種類"), x.get("種類"))
    pw = f"・パワー{x.get('パワー')}" if x.get("パワー") else ""
    ct = f"・カウンター{x.get('カウンター')}" if x.get("カウンター") and x.get("カウンター") != "-" else ""
    co = f"・コスト{x.get('コスト')}" if x.get("コスト") not in (None, "", "-") else ""
    return f"{x.get('name')}（{cid}・{kind}{co}{pw}{ct}）" + (f"「{t}」" if t else "「効果なし」")


def whiffs():
    rows = BA.read_jsonl(os.path.join("n1_results", "small_use", "labels.jsonl"))
    out = []
    for x in rows:
        c = [c for c in x["clauses"] if c["type"] == "on_play" and c["out"] == "whiff_visible"]
        if c:
            out.append(dict(x, why_vis=c[0]["why"], op_kind=c[0]["kind"]))
    return out


def _init(k=K):
    E.engine()
    _G["db"] = D.load_db()
    BA._G["db"] = _G["db"]
    BA._G["net"] = E.load_net(NET)
    BA._G["ship"] = BA.ship_spec(NET)
    HB._G["ref"] = E.SeatSpec(NET, sims=MJ.REFEREE_SIMS)
    _G["k"] = int(k)


def first_play_uuid(mv):
    m = BA.first_primitive(mv) if mv else None
    if m and m.get("action_type") == "PLAY":
        return (m.get("payload") or {}).get("uuid")
    return None


def effect_window(game, name, mv, uuid, cname):
    """mv を当てた後、uuid のカードの登場時が**対象に効くか**。確認・選択の窓が出たら「使う・選べるだけ選ぶ」側の合法手で
    進め（8 回まで）、本体の EFFECT（対象が空でない・対象を取らない動作）が積まれたかを見る。戻り値 (働いたか, 何が起きたか)。"""
    f = game.fork()
    try:
        f.apply_search_move(name, json.dumps(mv, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        return None, f"illegal: {type(exc).__name__}"
    evs, seen = [], []
    for _ in range(8):
        evs += [e for e in json.loads(f.events_json()) if e.get("type") == "EFFECT" and e.get("source_uuid") == uuid
                and e.get("trigger") == "ON_PLAY"]
        pend = json.loads(f.pending_json()) or {}
        # カード DB とエンジンの窓で全角・半角が違う名がある（「Ｘ・ドレーク」と「X・ドレーク」）ので揃えて比べる
        if not (cname and unicodedata.normalize("NFKC", cname) in unicodedata.normalize("NFKC", pend.get("message") or "")
                and pend.get("player_id") == name):
            break
        seen.append(pend.get("action"))
        legal = json.loads(f.legal_json(name))
        legal = [m for m in legal if (m.get("payload") or {}).get("accepted") is not False]
        if not legal:
            break
        legal.sort(key=lambda m: -len((m.get("payload") or {}).get("selected_uuids") or []))
        try:
            BA.apply(f, name, legal[0])
        except Exception:  # noqa: BLE001
            break
    # 対象の空の EFFECT（「1 枚まで」で何にも効かなかった）は働いたと数えない（`small_use.productive` と同じ線）
    body = [e for e in evs if e.get("action") not in SU.COST_ACTS and (e.get("targets") or e.get("action") in SU.SELF_ACTS)]
    win = f"（窓: {'・'.join(seen)}）" if seen else ""
    if body:
        return True, "効果が対象に効いた（" + ",".join(sorted({e.get("action") for e in body})) + "）" + win
    if seen:
        return False, "窓は出たが効く対象が無い" + win
    return False, "効果が動かない（窓も解決も無い）"


def board(game, name):
    h = json.loads(game.hidden_json())
    opp = "p2" if name == "p1" else "p1"

    def side(s, hand):
        pl = h["players"][s]
        don = pl.get("don") or {}
        d = {k: len(v) for k, v in don.items()} if isinstance(don, dict) else don
        out = {"leader": (pl.get("leader") or {}).get("card_id"), "life": len(pl["life"]), "hand_n": len(pl["hand"]),
               "deck": len(pl["deck"]), "trash": len(pl["trash"]), "don": d,
               "leader_don": len((pl.get("leader") or {}).get("attached_don") or []) if isinstance((pl.get("leader") or {}).get("attached_don"), list)
               else (pl.get("leader") or {}).get("attached_don"),
               "field": [{"card": c["card_id"], "rest": c.get("is_rest"),
                          "don": len(c["attached_don"]) if isinstance(c.get("attached_don"), list) else c.get("attached_don")}
                         for c in pl["field"]]}
        if hand:
            out["hand"] = [{"uuid": c["uuid"], "card": c["card_id"]} for c in pl["hand"]]
        return out
    return {"me": side(name, True), "opp": side(opp, False)}


# ---------------------------------------------------------------- 1. cases

def case_one(x):
    games = _G["games"]
    g = games[x["seed"]]
    p = g["points"][x["step"]]
    name, turn, seed, step = x["seat"], x["turn"], x["seed"], x["step"]
    db = _G["db"]
    p1, p2 = D.build_pair(db, g["leaders"][0], g["leaders"][1], seed, "real")
    game = BA.DR.new_game(seed, p1, p2)
    umap = SP.uuid_map(game)
    s0 = None
    before = []
    for s, mv in enumerate(g["moves"]):
        if s == step:
            break
        nm = json.loads(game.pending_json())["player_id"]
        q = g["points"][s]
        if q["seat"] == name and q["turn"] == turn:
            if s0 is None and q["kind"] == "main" and q["act"] == "MAIN_ACTION":
                s0 = s
            if s0 is not None:
                before.append({"step": s, "kind": q["kind"], "d": BA.desc(game, mv)})
        BA.apply(game, nm, mv)
        umap.update(SP.uuid_map(game))
    a = p["move"] if p["kind"] in ("lethal", "commit") else (p["a_box"] or p["move"])
    u = first_play_uuid(a) or x.get("uuid")
    cname = (raw_db().get(x["card"]) or {}).get("name")
    row = {"seed": seed, "step": step, "seat": name, "turn": turn, "kind": p["kind"], "card": x["card"], "name": cname,
           "uuid": u, "why_vis": x["why_vis"], "op_kind": x["op_kind"], "attacks": x["attacks"], "blocks": x["blocks"],
           "body": x["body"], "a": a, "a_desc": BA.desc(game, a), "before": before, "board": board(game, name),
           "rec_top": [{"d": BA.desc(game, t["mv"]), "n": t["n"], "q": t["q"], "mv": t["mv"]} for t in p["top"]],
           "a_rank": p["a_rank"]}
    # 根の読み直し（出荷の設定・同じ探索の種。証明の点は証明を切って根の並びを見る）
    opts = json.loads(BA._G["ship"].decide_opts(seed, turn, name, None))
    if p["kind"] == "lethal":
        opts["lethal_proof"] = "off"
    out = json.loads(game.fork().decide(name, json.dumps(opts)))
    rows = BA.ranked(out)
    ch = BA.chosen(out, rows) if rows else None
    row["fresh_kind"] = out.get("kind")
    row["fresh_same"] = bool(ch and BA.key(ch["mv"]) == BA.key(a))
    # ネットの方策（根の事前の見込み・束は束に入る合法手の和）
    P = (out.get("stats") or {}).get("P") or []
    pri = {}
    for gr in out.get("groups") or []:
        pri[int(gr["rep"])] = sum(P[i] for i in gr["idxs"] if i < len(P))
    if not pri:
        pri = {i: x for i, x in enumerate(P)}
    row["fresh_top"] = [{"d": BA.desc(game, r["mv"]), "n": r["n"], "q": round(r["q"], 4), "mv": r["mv"],
                         "p": round(pri.get(r["rep"], 0.0), 4),
                         "same_card": bool(first_play_uuid(r["mv"]) and umap.get(first_play_uuid(r["mv"])) == x["card"])}
                        for r in rows[:8]]
    row["n_cand"] = len(rows)
    # (a) 次点・(b) このキャラを出さない最上位（記録の上位 → 読み直しの並び → ターン終了）
    pool = [t["mv"] for t in p["top"]] + [r["mv"] for r in rows]
    others = [m for m in pool if BA.key(m) != BA.key(a)]
    row["alt_a"] = others[0] if others else None
    nb = [m for m in others if not (first_play_uuid(m) and umap.get(first_play_uuid(m)) == x["card"])]
    row["alt_b"], row["alt_b_src"] = (nb[0], "root") if nb else (SP.TURN_END, "turn_end_fill")
    row["alt_a_desc"], row["alt_b_desc"] = BA.desc(game, row["alt_a"]), BA.desc(game, row["alt_b"])
    q_of = {BA.key(r["mv"]): (r["n"], round(r["q"], 4)) for r in rows}
    q_of.update({BA.key(t["mv"]): (t["n"], t["q"]) for t in p["top"]})
    row["alt_a_nq"], row["alt_b_nq"], row["a_nq"] = q_of.get(BA.key(row["alt_a"])), q_of.get(BA.key(row["alt_b"])), q_of.get(BA.key(a))
    # 打った局面で効果が動かないことの確かめ
    play_mv = {"kind": "game", "action_type": "PLAY", "payload": {"uuid": u}}
    row["here"] = effect_window(game, name, play_mv, u, cname)
    # (c) ターンの最初の判断点で先に出していれば
    row["turn_start_step"] = s0
    if s0 is not None and s0 != step:
        g0 = BA.replay(g, s0)
        h0 = json.loads(g0.hidden_json())
        hand0 = [c["uuid"] for c in h0["players"][name]["hand"] if c["card_id"] == x["card"]]
        if hand0:
            u0 = u if u in hand0 else hand0[0]
            row["first"] = effect_window(g0, name, {"kind": "game", "action_type": "PLAY", "payload": {"uuid": u0}}, u0, cname)
            row["first_uuid"] = u0
        else:
            row["first"] = (None, "ターンの最初には手札に無かった（このターンに引いた・手札に加えた）")
    else:
        row["first"] = (None, "このターンの最初の手だった")
    # (d) 手札の他のキャラ（今出せるもの）
    alts = []
    seen = set()
    root_nq = {first_play_uuid(r["mv"]): (r["n"], round(r["q"], 4)) for r in rows if first_play_uuid(r["mv"])}
    for c in row["board"]["me"]["hand"]:
        cid = c["card"]
        info = raw_db().get(cid) or {}
        if info.get("種類") != "キャラクター" or cid == x["card"] or cid in seen:
            continue
        seen.add(cid)
        ok, what = effect_window(game, name, {"kind": "game", "action_type": "PLAY", "payload": {"uuid": c["uuid"]}},
                                 c["uuid"], info.get("name"))
        if ok is None:
            continue
        has_op = "【登場時】" in (info.get("効果(テキスト)") or "")
        alts.append({"card": cid, "uuid": c["uuid"], "name": info.get("name"), "cost": int(info.get("コスト") or 0),
                     "power": int(info.get("パワー") or 0), "has_on_play": has_op,
                     "works": bool(ok) if has_op else None, "what": what if has_op else "登場時なし",
                     "same_band": abs(int(info.get("コスト") or 0) - SP.card_table()[x["card"]]["cost"]) <= 1,
                     "root": root_nq.get(c["uuid"])})
    row["other_chars"] = alts
    row["d_exists"] = any(y["same_band"] and y["works"] for y in alts)
    return row


def _init_cases(path, k=K):
    _init(k)
    _G["games"] = {g["seed"]: g for g in BA.read_jsonl(path)}


def cmd_cases(a):
    os.makedirs(a.out, exist_ok=True)
    xs = whiffs()
    print("WHIFFS", len(xs), flush=True)
    with get_context("fork").Pool(a.workers, initializer=_init_cases, initargs=(a.games,)) as pool:
        rows = pool.map(case_one, xs, chunksize=1)
    rows.sort(key=lambda r: (r["seed"], r["step"]))
    with open(os.path.join(a.out, "cases.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in rows:
        print(r["seed"], r["step"], r["name"], "fresh_same", r["fresh_same"], "here", r["here"], "first", r["first"],
              "d", r["d_exists"], "b", r["alt_b_desc"], flush=True)


# ---------------------------------------------------------------- 2. judge

def _sp_rows():
    return {(x["seed"], x["step"]): x for x in SP.read_points(SP.OUT_DIR) if x["ok"]}


def judge_one(job):
    c, alt, prev = job
    t0 = time.time()
    g = _G["games"][c["seed"]]
    seed, step, name = c["seed"], c["step"], c["seat"]
    a, b = c["a"], c.get("alt_" + alt)
    if alt == "c":
        # 出す順番: ターンの最初の判断点で、記録の手 と 先にこのキャラを出す手
        step = c["turn_start_step"]
        a = g["points"][step]["a_box"] or g["points"][step]["move"]
        b = {"kind": "game", "action_type": "PLAY", "payload": {"uuid": c["first_uuid"]}}
    elif alt == "d":
        b = {"kind": "game", "action_type": "PLAY", "payload": {"uuid": c["d_uuid"]}}
    row = {"seed": seed, "step": step, "case_step": c["step"], "alt": alt, "a": a, "b": b, "ok": False, "why": None}
    game = BA.replay(g, step)
    wa, wb = list(prev["a"]) if prev else [], list(prev["b"]) if prev else []
    row["reused"] = len(wa)
    errs = {}
    for i in range(len(wa), _G["k"]):
        ws = MJ.world_seed(seed, step, SP.OFFSET + i)
        ra, ea = BA.rollout_ship(game, name, a, ws, HB._G["ref"], BA._G["ship"], both=True)
        rb, eb = BA.rollout_ship(game, name, b, ws, HB._G["ref"], BA._G["ship"], both=True)
        wa.append(ra)
        wb.append(rb)
        for e in (ea, eb):
            if e:
                errs[e] = errs.get(e, 0) + 1
    row["ref"] = dict(MJ.paired(wa, wb), a=wa, b=wb, errs=errs or None)
    row["ok"] = row["ref"]["diff"] is not None
    row["sec"] = round(time.time() - t0, 1)
    return row


def cmd_judge(a):
    cases = BA.read_jsonl(os.path.join(a.out, "cases.jsonl"))
    rpath = os.path.join(a.out, "points.jsonl")
    done = {(x["seed"], x.get("case_step", x["step"]), x["alt"]) for x in BA.read_jsonl(rpath) if x.get("ok")}
    sp = _sp_rows()
    jobs = []
    for c in cases:
        alt = "b" if c["alt_b"] is not None else "a"
        prev = sp.get((c["seed"], c["step"]))
        if prev and not (BA.key(prev["a"]) == BA.key(c["a"]) and BA.key(prev["b"]) == BA.key(c["alt_b"]) and alt == "b"):
            prev = None
        jobs.append((c, alt, {"a": prev["ref"]["a"], "b": prev["ref"]["b"]} if prev else None))
        if a.runner and c["alt_a"] is not None and BA.key(c["alt_a"]) != BA.key(c["alt_b"]):
            jobs.append((c, "a", None))
        if a.extra:
            if c["first"][0] and c.get("first_uuid"):
                jobs.append((c, "c", None))
            dd = [y for y in c["other_chars"] if y["same_band"] and y["works"]]
            if dd:
                jobs.append((dict(c, d_uuid=dd[0]["uuid"], d_name=dd[0]["name"]), "d", None))
    if a.only:
        # 指定の回だけ、判定済みの通りを引き継いで k 通りまで足す（配り方の番号は続き）
        want = set(a.only.split(","))
        have = {}
        for x in BA.read_jsonl(rpath):
            if x.get("ok"):
                have[(x["seed"], x.get("case_step", x["step"]), x["alt"])] = x
        jobs = [(c, alt, {"a": have[(c["seed"], c["step"], alt)]["ref"]["a"], "b": have[(c["seed"], c["step"], alt)]["ref"]["b"]})
                for c, alt, _ in jobs if f"{c['seed']}:{c['step']}" in want and (c["seed"], c["step"], alt) in have
                and len(have[(c["seed"], c["step"], alt)]["ref"]["a"]) < a.k]
    else:
        jobs = [j for j in jobs if (j[0]["seed"], j[0]["step"], j[1]) not in done]
    print(f"JUDGE todo {len(jobs)} k={a.k} -> {rpath}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init_cases, initargs=(a.games, a.k)) as pool:
        for i, row in enumerate(pool.imap_unordered(judge_one, jobs, chunksize=1)):
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            r = row["ref"]
            print(f"JUDGE seed={row['seed']} step={row['step']} alt={row['alt']} reused={row['reused']} "
                  f"a-b={-r['diff'] if r['diff'] is not None else None} ci={r['ci']} n={r['n']} sec={row['sec']} "
                  f"({i + 1}/{len(jobs)} {time.time() - t0:.0f}s)", flush=True)
    print("JUDGE_DONE", flush=True)


# ---------------------------------------------------------------- 3. summary

def played_minus_alt(r):
    """打った手 − 代わりの手（paired は b − a）。"""
    d, ci = r["ref"]["diff"], r["ref"]["ci"]
    return -d, [None if ci[1] is None else -ci[1], None if ci[0] is None else -ci[0]]


def cmd_summary(a):
    cases = BA.read_jsonl(os.path.join(a.out, "cases.jsonl"))
    pts = {}
    for x in BA.read_jsonl(os.path.join(a.out, "points.jsonl")):
        if x.get("ok"):
            pts[(x["seed"], x["step"], x["alt"])] = x
    out = {"cases": len(cases), "rows": []}
    gains = []
    for c in cases:
        alt = "b" if c["alt_b"] is not None else "a"
        r = pts.get((c["seed"], c["step"], alt))
        ra = pts.get((c["seed"], c["step"], "a"))
        rc = pts.get((c["seed"], c.get("turn_start_step"), "c"))
        rd = pts.get((c["seed"], c["step"], "d"))
        dd = [y for y in c["other_chars"] if y["same_band"] and y["works"]]

        def extra(x, desc):
            if not x:
                return None
            m2, ci2 = played_minus_alt(x)
            return {"desc": desc, "played_minus_alt": round(m2, 4), "ci": ci2, "pairs": x["ref"]["n"],
                    "wins_played": x["ref"]["wa"], "wins_alt": x["ref"]["wb"]}
        m, ci = played_minus_alt(r) if r else (None, [None, None])
        if m is not None:
            gains.append(m)
        if m is None:
            auto = "未判定"
        elif ci[1] is not None and ci[1] < 0:
            auto = "違和感あり・損"
        else:
            auto = "差なし"
        out["rows"].append({"seed": c["seed"], "step": c["step"], "turn": c["turn"], "name": c["name"], "card": c["card"],
                            "why": c["why_vis"], "alt": alt, "alt_desc": c["alt_" + alt + "_desc"],
                            "alt_src": c["alt_b_src"] if alt == "b" else "runner_up",
                            "played_minus_alt": None if m is None else round(m, 4), "ci": ci,
                            "wins_played": r["ref"]["wa"] if r else None, "wins_alt": r["ref"]["wb"] if r else None,
                            "pairs": r["ref"]["n"] if r else None, "reused": r["reused"] if r else None,
                            "runner": None if not ra else {"desc": c["alt_a_desc"], "played_minus_alt": round(-ra["ref"]["diff"], 4),
                                                           "ci": played_minus_alt(ra)[1]},
                            "order": extra(rc, "ターンの最初の判断点で、記録の手 − 先にこのキャラを出す手"),
                            "other_char": extra(rd, f"打った手 − {dd[0]['name']}を出す手" if dd else None),
                            "first_works": c["first"][0], "first_what": c["first"][1], "here": c["here"],
                            "d_exists": c["d_exists"], "body": c["body"], "auto": auto,
                            "a_nq": c["a_nq"], "alt_nq": c["alt_" + alt + "_nq"], "fresh_same": c["fresh_same"]})
    vp = os.path.join(a.out, "verdicts.json")
    if os.path.exists(vp):
        vd = json.load(open(vp, encoding="utf-8"))
        for r in out["rows"]:
            v = vd.get(f"{r['seed']}:{r['step']}") or {}
            r.update({k: v[k] for k in ("verdict", "type", "reason") if k in v})
        out["verdict_counts"], out["type_counts"] = {}, {}
        for r in out["rows"]:
            for k, f in (("verdict_counts", "verdict"), ("type_counts", "type")):
                if r.get(f):
                    out[k][r[f]] = out[k].get(r[f], 0) + 1
    out["mean_played_minus_alt"] = HB.mean_ci(gains)
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    for r in out["rows"]:
        print(r["seed"], r["step"], r["name"], r["why"], r["played_minus_alt"], r["ci"], r["auto"], "first", r["first_works"],
              "d", r["d_exists"], "| order", (r["order"] or {}).get("played_minus_alt"), (r["order"] or {}).get("ci"),
              "| other", (r["other_char"] or {}).get("played_minus_alt"), (r["other_char"] or {}).get("ci"))
    print("MEAN", out["mean_played_minus_alt"])


# ---------------------------------------------------------------- 4. md（人が読む形）

def nm(cid):
    return (raw_db().get(cid) or {}).get("name") or cid


def mv_ja(d):
    """BA.desc の JSON → 言葉。"""
    if d is None:
        return "—"
    j = json.loads(d) if isinstance(d, str) else d
    at, c = j.get("action_type"), j.get("card")
    tg = "・".join(nm(t) for t in j.get("targets") or [])
    if at == "TURN_END":
        return "ターン終了"
    if at == "PLAY":
        return f"{nm(c)}を出す"
    if at == "ACTIVATE_MAIN":
        return f"{nm(c)}の起動メイン"
    if at in ("DON_BOX", "ATTACK", "ATTACH_DON"):
        k = j.get("don_k")
        don = f"ドン{k}枚を付けて" if k else ""
        if tg:
            return f"{nm(c)}に{don}{tg}へアタック" if don else f"{nm(c)}で{tg}へアタック"
        return f"{nm(c)}にドン{k or 1}枚を付ける"
    if at == "RESOLVE_EFFECT_SELECTION":
        if j.get("accepted") is False:
            return "効果を使わない"
        sel = j.get("selected")
        return "効果の選択: " + ("・".join(nm(x) for x in sel) if sel else "選ばない／はい")
    return json.dumps(j, ensure_ascii=False)


def don_ja(d):
    if not isinstance(d, dict):
        return str(d)
    return f"アクティブ {d.get('active', 0)}・レスト {d.get('rested', 0)}・付与 {d.get('attached', 0)}・ドンデッキ {d.get('deck', 0)}"


def cmd_md(a):
    cases = BA.read_jsonl(os.path.join(a.out, "cases.jsonl"))
    sm = {}
    sp = os.path.join(a.out, "summary.json")
    if os.path.exists(sp):
        sm = {(r["seed"], r["step"]): r for r in json.load(open(sp, encoding="utf-8"))["rows"]}
    L = ["# 「見えていた空振り」25 回の局面（WP `small-whiff`・§7.62）", "",
         "道具 `tests/scripts/small_whiff.py`。各回: 出した時点の盤面・このターンに先に打った手・根の候補（出荷の設定で同じ局面を読み直した上位 8・"
         "訪問数と評価〔自分から見た勝ちやすさ・−1〜+1〕）・代わりの手・判定（16 通り・打った手 − 代わりの手の勝率の差）。", ""]
    for i, c in enumerate(cases, 1):
        me, op = c["board"]["me"], c["board"]["opp"]
        L.append(f"## {i}. {c['name']}（{c['card']}）— 局 {c['seed']}・手 {c['step']}・ターン {c['turn']}・{c['seat']}")
        L.append("")
        L.append(f"- **出したキャラ**: {cdesc(c['card'], True)}")
        L.append(f"- **効果が働かなかった理由**: {WHY_JA.get(c['why_vis'], c['why_vis'])}（同じ局面で出し直して確かめた: {c['here'][1]}）")
        L.append(f"- **体として**: アタック {c['attacks']} 回・ブロック {c['blocks']} 回")
        L.append(f"- **自分**: リーダー {nm(me['leader'])}（付与ドン {me['leader_don']}）・ライフ {me['life']}・ドン {don_ja(me['don'])}・山 {me['deck']}・トラッシュ {me['trash']}")
        L.append("- **自分の場**: " + ("、".join(f"{nm(x['card'])}{'（レスト）' if x['rest'] else ''}{'・ドン' + str(x['don']) if x['don'] else ''}" for x in me["field"]) or "なし"))
        L.append(f"- **自分の手札**（{len(me['hand'])} 枚・出したキャラを含む）:")
        for h in me["hand"]:
            L.append(f"  - {cdesc(h['card'])}")
        L.append(f"- **相手**: リーダー {nm(op['leader'])}・ライフ {op['life']}・手札 {op['hand_n']}・ドン {don_ja(op['don'])}")
        L.append("- **相手の場**: " + ("、".join(f"{nm(x['card'])}（コスト{(raw_db().get(x['card']) or {}).get('コスト')}・パワー{(raw_db().get(x['card']) or {}).get('パワー')}{'・レスト' if x['rest'] else ''}）" for x in op["field"]) or "なし"))
        L.append("- **このターンに先に打った手**: " + ("、".join(mv_ja(b["d"]) for b in c["before"]) or "なし（ターンの最初の手）"))
        L.append(f"- **打った手**: {mv_ja(c['a_desc'])}（根の順位 {c['a_rank'] + 1 if c['a_rank'] is not None else '—'}・候補 {c['n_cand']}）"
                 + ("（詰みの証明が選んだ手。下の根は証明を切って読み直した並び）" if c["kind"] == "lethal" else ""))
        L.append("- **根の候補**（上位 8・訪問数／評価／ネットの事前の見込み）:")
        L.append("")
        L.append("  | 順 | 手 | 訪問数 | 評価 | 事前の見込み |")
        L.append("  |---:|---|---:|---:|---:|")
        for j, t in enumerate(c["fresh_top"], 1):
            L.append(f"  | {j} | {mv_ja(t['d'])}{'（このキャラ）' if t['same_card'] else ''} | {int(t['n'])} | {t['q']:+.3f} | {t.get('p', 0):.2f} |")
        L.append("")
        L.append(f"- **代わりの手**: (a) 根の次点・(b) このキャラを出さない最上位 = **{mv_ja(c['alt_b_desc'])}**"
                 + ("（(a) と同じ）" if BA.key(c["alt_a"]) == BA.key(c["alt_b"]) else f"・(a) {mv_ja(c['alt_a_desc'])}"))
        L.append(f"  - (c) 出す順番: {c['first'][1]}" + ("" if c["turn_start_step"] in (None, c["step"]) else "（ターンの最初の判断点で先にこのキャラを出した場合）"))
        oc = [y for y in c["other_chars"]]
        if oc:
            L.append("  - (d) 手札の他のキャラ（今出せるもの）: " + "、".join(
                f"{y['name']}（コスト{y['cost']}・{'登場時が働く' if y['works'] else ('登場時が働かない' if y['has_on_play'] else '登場時なし')}"
                f"{'・根 ' + str(int(y['root'][0])) + '回' if y.get('root') else ''}）" for y in oc))
        else:
            L.append("  - (d) 手札の他のキャラ（今出せるもの）: なし")
        r = sm.get((c["seed"], c["step"]))
        if r and r["played_minus_alt"] is not None:
            L.append(f"- **判定**（{r['pairs']} 通り・打った手 − 代わりの手）: {r['played_minus_alt']:+.3f}（95% 区間 {r['ci'][0]:+.3f}〜{r['ci'][1]:+.3f}）"
                     f"・勝率 打った手 {r['wins_played']:.3f}／代わりの手 {r['wins_alt']:.3f}")
        for kk in ("order", "other_char"):
            e = (r or {}).get(kk)
            if e:
                L.append(f"- **判定・{'出す順番' if kk == 'order' else '別のキャラ'}**（{e['pairs']} 通り・{e['desc']}）: {e['played_minus_alt']:+.3f}"
                         f"（95% 区間 {e['ci'][0]:+.3f}〜{e['ci'][1]:+.3f}）")
        if r and r.get("verdict"):
            L.append(f"- **分類**: {r['verdict']} — {r.get('reason', '')}")
        L.append("")
    with open(os.path.join(a.out, "cases.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("MD", len(cases))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cases")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--games", default=GAMES)
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("judge")
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--runner", action="store_true", help="(a) 根の次点が (b) と違う回の対も足す")
    p.add_argument("--only", default=None, help="seed:step,…: その回の判定済みの対を k 通りまで足す（--k 32 と使う）")
    p.add_argument("--extra", action="store_true",
                   help="(c) ターンの最初に先にこのキャラを出す手（効果が働く回）・(d) 効果の働く同じコスト帯の別のキャラを出す手の対も足す")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--games", default=GAMES)
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("summary")
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("md")
    p.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)
    {"cases": cmd_cases, "judge": cmd_judge, "summary": cmd_summary, "md": cmd_md}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
