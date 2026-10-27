"""出した低パワーのキャラは効果を実際に使えたか（WP `small-use`・`docs/search_efficiency_plan.md` §7.61・台帳 2+463）。

読み取りだけの計測（エンジン・ネット・既定の挙動は変えない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。
判定の打ち直しはしない。§7.59 の 150 局（`n1_results/hold_bias/games.jsonl`）を探索なしで当て直し、
§7.60（`small-play`）の「パワー 4000 以下のキャラを出した」737 回それぞれについて、出した後の進行を追う。

1. ``collect``: 全局を当て直し、各手の後にエンジンが積むイベント（``EFFECT``＝出どころの個体番号・発動のきっかけ・
   動作・対象・成否）・アタック・ブロック・起動メイン・選択の窓（候補・選んだもの）を、小さいキャラの個体番号ごとに集める。
   出した時点の盤面（両席の場・手札・トラッシュ・リーダー・ドン）も残す（判断の時点で見えていた情報で決まる空振りの判定用）。
   出力 `n1_results/small_use/trace.jsonl`（1 行＝出した 1 回）。
2. ``summary``: カードの本文を型に分け（登場時サーチ・ドロー・ドン追加・対象を取る効果・任意のコスト・ブロッカー・起動メイン・
   アタック時・KO 時・常時など）、型ごとに「働いた／空振り／判定外」を数える。空振りは「出した時点で見えていた情報で
   決まる」（対象になる相手のキャラがいない・リーダーの条件を満たさない など）と「運・その後の進行で決まる」に分ける。
   §7.60 の判定 400 点（`n1_results/small_play/points_shard*.jsonl`）を働いた／空振りで分け、「出した − 出さなかった」で集計する。

    python tests/scripts/small_use.py collect
    python tests/scripts/small_use.py summary
"""
import argparse
import json
import os
import re
import sys
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blunder_audit as BA  # noqa: E402
import hold_bias as HB  # noqa: E402
import small_play as SP  # noqa: E402

from opcg_sim.loop import decks as D, engine as E  # noqa: E402

GAMES = SP.GAMES
SP_DIR = SP.OUT_DIR
OUT_DIR = os.path.join("n1_results", "small_use")
_G = {}


# ---------------------------------------------------------------- 1. collect

def _snap(h, seat):
    """出した時点の盤面（両席・公開の範囲＋自分の手札）。"""
    out = {}
    for s2 in ("p1", "p2"):
        pl = h["players"][s2]
        out[s2] = {
            "leader": (pl.get("leader") or {}).get("card_id"),
            "field": [{"uuid": c["uuid"], "card": c["card_id"], "rest": c.get("is_rest"),
                       "cost_buff": (c.get("cost_buff") or 0) + (c.get("timed_cost") or 0)} for c in pl["field"]],
            "trash": [c["card_id"] for c in pl["trash"]],
            "hand_n": len(pl["hand"]), "life_n": len(pl["life"]), "deck_n": len(pl["deck"]),
            "don": {k: len(v) for k, v in (pl.get("don") or {}).items()} if isinstance(pl.get("don"), dict) else pl.get("don"),
            "negate_onplay_until": pl.get("negate_onplay_until"),
        }
        if s2 == seat:
            out[s2]["hand"] = [c["card_id"] for c in pl["hand"]]
    return out


def collect_one(g):
    E.engine()
    db = _G.setdefault("db", D.load_db())
    p1, p2 = D.build_pair(db, g["leaders"][0], g["leaders"][1], g["seed"], "real")
    game = BA.DR.new_game(g["seed"], p1, p2)
    umap = SP.uuid_map(game)
    tr = {}       # uuid → 今追っている 1 回（出してから次に出すまで）
    out = []
    for s, mv in enumerate(g["moves"]):
        pend = json.loads(game.pending_json())
        name = pend["player_id"]
        t = int(game.turn_count or 0)
        at = mv.get("action_type")
        p = mv.get("payload") or {}
        u_mv = p.get("uuid") if mv.get("kind") != "battle" else mv.get("card_uuid")
        new = None
        if at == "PLAY" and mv.get("kind") != "battle":
            c = SP.card_table().get(umap.get(u_mv)) or {}
            if c.get("type") == "キャラクター" and c["power"] <= SP.SMALL:
                h = json.loads(game.hidden_json())
                new = {"seed": g["seed"], "step": s, "seat": name, "turn": t, "uuid": u_mv, "card": umap.get(u_mv),
                       "kind": g["points"][s]["kind"], "snap": _snap(h, name), "ev": [], "dialogs": [],
                       "attacks": [], "blocks": [], "activates": [], "chances": {}, "left": None, "left_turn": None, "left_step": None}
        # 選択の窓（直前に積まれた EFFECT の出どころに結び付けるため、手と候補を残す）
        dlg = None
        if at == "RESOLVE_EFFECT_SELECTION":
            dlg = {"step": s, "turn": t, "who": name, "action": pend.get("action"), "msg": pend.get("message"),
                   "cands": [x.get("uuid") for x in pend.get("candidates") or []] or pend.get("selectable_uuids") or [],
                   "can_skip": pend.get("can_skip"), "sel": p.get("selected_uuids") or [], "accepted": p.get("accepted")}
        # 機会: 候補にこの個体が出た窓（ブロッカーの窓・メインの行動など）の数
        sel = set(pend.get("selectable_uuids") or []) | {x.get("uuid") for x in pend.get("candidates") or []}
        for u in sel & set(tr):
            r = tr[u]
            if r["left"] is None:
                k = pend.get("action") or "?"
                r["chances"][k] = r["chances"].get(k, 0) + 1
        BA.apply(game, name, mv)
        umap.update(SP.uuid_map(game))
        evs = json.loads(game.events_json())
        if new is not None:
            prev = tr.pop(u_mv, None)
            if prev is not None:
                out.append(prev)
            tr[u_mv] = new
        # 起動・アタック・ブロック（手そのもの）
        if u_mv in tr:
            r = tr[u_mv]
            if at == "ATTACK":
                r["attacks"].append({"step": s, "turn": t, "targets": p.get("target_ids")})
            elif at == "SELECT_BLOCKER":
                r["blocks"].append({"step": s, "turn": t})
            elif at == "ACTIVATE_MAIN":
                r["activates"].append({"step": s, "turn": t})
        for e in evs:
            su = e.get("source_uuid")
            if e.get("type") == "EFFECT" and su in tr:
                tr[su]["ev"].append(dict(e, step=s, turn=t))
                if dlg is not None and dlg not in tr[su]["dialogs"]:
                    tr[su]["dialogs"].append(dlg)
        # 選択の窓を、その窓で積まれた EFFECT が無くても、直前の出どころへ結ぶ（空の選択は何も積まない）
        if dlg is not None and "の効果" in (dlg["msg"] or ""):
            nm = dlg["msg"].split("」の効果")[0].lstrip("「")
            for r in tr.values():
                if r["left"] is None and (SP.card_table().get(r["card"]) or {}).get("name") == nm and dlg not in r["dialogs"] \
                        and r["seat"] == name:
                    r["dialogs"].append(dict(dlg, by_name=True))
        # 場を離れたか
        h = None
        for u, r in tr.items():
            if r["left"] is None:
                h = h or json.loads(game.hidden_json())
                if not any(c["uuid"] == u for c in h["players"][r["seat"]]["field"]):
                    r["left"] = SP.where(game, r["seat"], u)
                    r["left_turn"] = int(game.turn_count or 0)
                    r["left_step"] = s
        if game.winner() is not None:
            break
    out.extend(tr.values())
    for r in out:
        r["game_end_turn"] = int(game.turn_count or 0)
        r["winner"] = game.winner()
    return out


def cmd_collect(a):
    os.makedirs(a.out, exist_ok=True)
    games = BA.read_jsonl(a.games)
    with get_context("fork").Pool(a.workers) as pool:
        res = pool.map(collect_one, games, chunksize=2)
    rows = sorted((r for rs in res for r in rs), key=lambda r: (r["seed"], r["step"]))
    with open(os.path.join(a.out, "trace.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("COLLECT games", len(games), "small plays", len(rows), flush=True)


# ---------------------------------------------------------------- 2. summary（本文の型・働いたか）

#: 払うコスト・見るだけ・残りの片付けの動作（これだけでは「効果が働いた」と数えない）。
COST_ACTS = {"LOOK", "REVEAL", "LOOK_LIFE", "DISCARD", "REST_DON", "RETURN_DON", "TRASH", "DECK_BOTTOM", "SELECT"}
#: 対象を取らずに働く動作。
SELF_ACTS = {"DRAW", "RAMP_DON", "TRASH_FROM_DECK", "ACTIVE_DON", "RULE_PROCESSING", "HEAL"}
OPT_RE = re.compile(r"(できる|ドン[!‼]+-\d+(\([^)]*\))?)\s*[:：]")


def segments(text):
    """本文を「 / 」で区切った節（括弧書きの注釈・選択肢の「・」行は前の節に含める）。"""
    out = []
    for seg in (text or "").split(" / "):
        seg = seg.strip()
        if out and (seg.startswith("(") or seg.startswith("・")):
            out[-1] += " " + seg
        elif seg:
            out.append(seg)
    return out


def on_play_kind(seg):
    """登場時の節の主な中身（コストの「：」の後を見る）。"""
    body = OPT_RE.split(seg)[-1] if OPT_RE.search(seg) else seg
    if re.search(r"デッキの上から\d+枚(まで)?を見て", body) and "手札に加える" in body:
        return "search"
    if re.search(r"デッキの上から\d+枚(まで)?を見て", body):
        return "arrange"
    if re.search(r"カード\d枚を引く", body):
        return "draw"
    if re.search(r"ドン[!‼]+デッキからドン[!‼]+", body):
        return "don"
    if "相手の" in body and re.search(r"(キャラ|リーダー).*(KO|レスト|パワー-|コスト-|手札に戻す|デッキの下|アタックできない|捨てる)", body):
        return "target"
    if "相手の手札" in body:
        return "target"
    if "登場させる" in body:
        return "play"
    if "トラッシュ" in body and "手札に加える" in body:
        return "recover"
    return "other"


def card_clauses(text):
    """本文 → 節の型の集合（on_play:<kind>・blocker・activate・on_attack・on_ko・trigger_other・passive）。"""
    out = []
    for seg in segments(text):
        if "【登場時】" in seg:
            out.append({"type": "on_play", "kind": on_play_kind(seg), "optional": bool(OPT_RE.search(seg)),
                        "cond": "場合" in seg, "seg": seg})
        if "【起動メイン】" in seg:
            out.append({"type": "activate", "seg": seg})
        if "【アタック時】" in seg and "相手のアタック時" not in seg:
            out.append({"type": "on_attack", "seg": seg})
        if "【KO時】" in seg or "KOされた時" in seg:
            out.append({"type": "on_ko", "seg": seg})
        if "ブロッカー" in seg and not any(t in seg for t in ("【登場時】", "【起動メイン】", "【アタック時】")) and "ブロッカー" not in [c.get("type") for c in out]:
            out.append({"type": "blocker", "seg": seg, "cond": "場合" in seg or "【ドン" in seg})
        if not any(t in seg for t in ("【登場時】", "【起動メイン】", "【アタック時】", "【KO時】", "KOされた時", "ブロッカー")):
            if re.search(r"【[^】]*(時|ターン終了)[^】]*】|した時", seg):
                out.append({"type": "trigger_other", "seg": seg})
            elif not seg.startswith("("):
                out.append({"type": "passive", "seg": seg})
    return out


def productive(evs, trig, kind=None):
    """trig の EFFECT のうち、効果の本体が働いたもの。"""
    good = []
    looked = {e["step"] for e in evs if e.get("trigger") == trig and e.get("action") == "LOOK"}
    for e in evs:
        if e.get("trigger") != trig:
            continue
        a, tg = e.get("action"), e.get("targets") or []
        if kind in ("search", "recover"):
            ok = a == "MOVE_CARD" and e.get("dest") == "HAND" and tg
        elif kind == "draw":
            ok = a == "DRAW"
        elif kind == "don":
            ok = a == "RAMP_DON" and (e.get("value") or 0) > 0
        elif kind == "play":
            ok = a == "PLAY_CARD" and tg
        elif kind in ("arrange", "attack"):
            # 並べ替え（見るだけの効果）・アタック時は、見る動作も効果の本体
            ok = a == "LOOK" or (a not in COST_ACTS and (tg or a in SELF_ACTS))
        else:
            # デッキの下・トラッシュへ置く動作は、見た札の残りの片付けでなければ本体（相手のキャラを除く型）
            ok = (a not in COST_ACTS and (tg or a in SELF_ACTS)) or \
                (a in ("TRASH", "DECK_BOTTOM") and tg and kind in ("target", "other") and e["step"] not in looked)
        if ok:
            good.append(e)
    return good


def leader_ok(seg, leader):
    """「自分のリーダーが…場合」の条件（読めなければ None）。"""
    m = re.search(r"自分のリーダーが(.+?)(の|を持つ|を含む特徴を持つ)場合", seg)
    if not m:
        return None
    c = SP.card_table().get(leader) or {}
    raw = _G.setdefault("raw", {x["number"]: x for x in json.load(open(os.path.join("opcg_sim", "data", "opcg_cards.json"), encoding="utf-8"))})
    x = raw.get(leader) or {}
    traits = x.get("特徴") or ""
    if isinstance(traits, list):
        traits = "/".join(traits)
    cond = m.group(1)
    names = re.findall(r"「([^」]+)」", cond)
    feats = re.findall(r"《([^》]+)》", cond)
    parts = re.findall(r"『([^』]+)』", cond)
    if names:
        return any(n == c.get("name") for n in names)
    if feats:
        return any(f in traits.split("/") for f in feats)
    if parts:
        return any(p in traits for p in parts)
    return None


def judge_on_play(r, cl):
    """登場時の節 1 つ → (結果, 理由)。結果は worked／whiff_visible／whiff_luck／whiff_choice。"""
    evs = [e for e in r["ev"] if e.get("trigger") == "ON_PLAY" and e["turn"] == r["turn"] and e["step"] <= r["step"] + 40]
    if productive(evs, "ON_PLAY", cl["kind"]):
        return "worked", cl["kind"]
    me = r["snap"][r["seat"]]
    neg = me.get("negate_onplay_until") or 0
    if me["leader"] == "OP09-081" or (neg and neg >= r["turn"]):
        return "whiff_visible", "negated"
    lk = leader_ok(cl["seg"], me["leader"])
    if lk is False:
        return "whiff_visible", "leader_cond"
    dl = [d for d in r["dialogs"] if d["turn"] == r["turn"] and d["step"] <= r["step"] + 40]
    if cl["optional"] and any(d["action"] == "CONFIRM_OPTIONAL" and d.get("accepted") is False for d in dl):
        return "whiff_choice", "declined"
    if not evs:
        if cl["optional"] and not any(d["action"] == "CONFIRM_OPTIONAL" for d in dl):
            return "whiff_visible", "cost_unpayable"
        if cl["cond"]:
            return "whiff_visible", "cond_unmet"
        if cl["kind"] in ("target", "play", "recover"):
            # 対象を選ぶ窓が無い＝対象が無かった（相手の場・自分の手札・トラッシュは見えている）
            return "whiff_visible", "no_target"
        return "whiff_luck", "no_event"
    if cl["kind"] in ("search",):
        return "whiff_luck", "search_miss"
    if cl["kind"] in ("target", "play", "recover", "other"):
        sel_dl = [d for d in dl if d["action"] in ("SEARCH_AND_SELECT", "SELECT_TARGET") or "対象" in (d["msg"] or "")]
        if any(d["cands"] and not d["sel"] for d in sel_dl):
            return "whiff_choice", "chose_none"
        return "whiff_visible", "no_target"
    return "whiff_luck", cl["kind"] + "_miss"


def judge_play(r):
    """出した 1 回 → 節ごとの結果と、全体の結果。"""
    t = SP.card_table()[r["card"]]["text"]
    cls = card_clauses(t)
    res = []
    left_trash_opp = r["left"] == "trash" and r["left_turn"] is not None
    for cl in cls:
        ty = cl["type"]
        if ty == "on_play":
            o, why = judge_on_play(r, cl)
        elif ty == "blocker":
            if r["blocks"]:
                o, why = "worked", "blocked"
            elif r["chances"].get("SELECT_BLOCKER"):
                o, why = "whiff_choice", "had_chance_no_block"
            elif r["left"] is None and r.get("left_turn") is None:
                o, why = "na", "no_attack_faced"
            else:
                o, why = "na", "left_before_chance"
        elif ty == "activate":
            if productive(r["ev"], "ACTIVATE_MAIN") or r["activates"]:
                o, why = "worked", "activated"
            else:
                o, why = "whiff_choice", "never_activated"
        elif ty == "on_attack":
            if productive(r["ev"], "ON_ATTACK", "attack"):
                o, why = "worked", "on_attack"
            elif r["attacks"]:
                # ドンの付与・コストの支払い・場の条件を満たさずにアタックした（自分の選択）
                o, why = "whiff_choice", "attacked_cond_unmet"
            else:
                o, why = "whiff_choice", "never_attacked"
        elif ty == "on_ko":
            if [e for e in r["ev"] if e.get("trigger") == "ON_KO"]:
                o, why = "worked", "ko_resolved"
            elif left_trash_opp:
                o, why = "whiff_luck", "left_without_ko_effect"
            else:
                o, why = "na", "not_koed"
        elif ty == "trigger_other":
            trig = [e for e in r["ev"] if e.get("trigger") not in ("ON_PLAY", "ACTIVATE_MAIN", "ON_ATTACK", "ON_KO")]
            o, why = ("worked", "triggered") if trig else ("na", "not_judged")
        else:
            o, why = "na", "passive"
        res.append({"type": ty, "kind": cl.get("kind"), "out": o, "why": why})
    eff = [x for x in res if x["out"] != "na"]
    if not cls:
        overall = "vanilla"
    elif any(x["out"] == "worked" for x in res):
        overall = "worked"
    elif not eff:
        overall = "na"
    elif any(x["out"] == "whiff_visible" for x in eff):
        overall = "whiff_visible"
    elif any(x["out"] == "whiff_luck" for x in eff):
        overall = "whiff_luck"
    else:
        overall = "whiff_choice"
    body = bool(r["attacks"] or r["blocks"])
    return {"clauses": res, "overall": overall, "body": body}


def _rate(c, n):
    return round(c / n, 3) if n else None


def cmd_summary(a):
    rows = BA.read_jsonl(os.path.join(a.out, "trace.jsonl"))
    pts = {(x["seed"], x["step"]): x for x in SP.read_points(SP_DIR) if x["ok"]}
    lab = []
    for r in rows:
        j = judge_play(r)
        lab.append({"seed": r["seed"], "step": r["step"], "seat": r["seat"], "turn": r["turn"], "card": r["card"],
                    "name": SP.card_table()[r["card"]]["name"], "kind": r["kind"], "left": r["left"],
                    "attacks": len(r["attacks"]), "blocks": len(r["blocks"]), "activates": len(r["activates"]),
                    "block_chances": r["chances"].get("SELECT_BLOCKER", 0), **j,
                    "judged": (r["seed"], r["step"]) in pts})
    with open(os.path.join(a.out, "labels.jsonl"), "w", encoding="utf-8") as f:
        for x in lab:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    n = len(lab)
    s = {"plays": n, "judged_points": len(pts), "overall": {}, "by_clause": {}, "why": {}, "body": {}, "only_played": {}}
    for x in lab:
        s["overall"][x["overall"]] = s["overall"].get(x["overall"], 0) + 1
        for c in x["clauses"]:
            k = c["type"] + (":" + c["kind"] if c["kind"] else "")
            d = s["by_clause"].setdefault(k, {"n": 0})
            d["n"] += 1
            d[c["out"]] = d.get(c["out"], 0) + 1
            w = s["why"].setdefault(k, {})
            w[c["out"] + ":" + c["why"]] = w.get(c["out"] + ":" + c["why"], 0) + 1
    for d in s["by_clause"].values():
        for o in ("worked", "whiff_visible", "whiff_luck", "whiff_choice", "na"):
            d[o + "_rate"] = _rate(d.get(o, 0), d["n"])
    s["body"] = {"used": sum(x["body"] for x in lab), "rate": _rate(sum(x["body"] for x in lab), n),
                 "attacked": sum(1 for x in lab if x["attacks"]), "blocked": sum(1 for x in lab if x["blocks"])}
    # 効果も体も働かなかった（本当に「出しただけ」の候補）
    s["only_played"] = {"n": sum(1 for x in lab if x["overall"] not in ("worked",) and not x["body"]),
                        "by_overall": {}}
    for x in lab:
        if x["overall"] != "worked" and not x["body"]:
            s["only_played"]["by_overall"][x["overall"]] = s["only_played"]["by_overall"].get(x["overall"], 0) + 1
    # 登場時を持つ回だけの結果
    op = [x for x in lab if any(c["type"] == "on_play" for c in x["clauses"])]
    s["on_play"] = {"n": len(op)}
    for x in op:
        o = [c for c in x["clauses"] if c["type"] == "on_play"][0]["out"]
        s["on_play"][o] = s["on_play"].get(o, 0) + 1
    # 判定 400 点との突き合わせ（出した − 出さなかった）
    by = {}
    for x in lab:
        p = pts.get((x["seed"], x["step"]))
        if not p:
            continue
        g = SP.gain(p)
        opc = [c for c in x["clauses"] if c["type"] == "on_play"]
        keys = {"overall:" + x["overall"], "body:" + ("used" if x["body"] else "unused"),
                "on_play:" + (opc[0]["out"] if opc else "none"),
                "only_played:" + ("yes" if x["overall"] != "worked" and not x["body"] else "no"),
                "worked_any:" + ("yes" if x["overall"] == "worked" else "no")}
        if opc:
            keys.add("on_play_kind:" + opc[0]["kind"] + "|" + opc[0]["out"])
        for k in keys:
            by.setdefault(k, []).append(g)
    s["gain"] = {k: HB.mean_ci(v) for k, v in sorted(by.items())}
    s["gain"]["all"] = HB.mean_ci([SP.gain(p) for p in pts.values()])
    txt = json.dumps(s, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--games", default=GAMES)
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("summary")
    p.add_argument("--games", default=GAMES)
    p.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)
    {"collect": cmd_collect, "summary": cmd_summary}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
