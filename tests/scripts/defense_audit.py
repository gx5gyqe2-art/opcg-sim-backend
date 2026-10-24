"""守りの見逃しの監査（`docs/search_efficiency_plan.md` §7.24・WP `defense-audit`・台帳 2+343）。

§7.13（`lethal_audit.py`・自分の詰みの見逃し）の裏返し。**相手に詰みがある局面で、CPU は防げる応手が
在るのに防がないことがどれだけあるか**を数える。判定器（AND-OR の総当たり・置換表・時間の上限）は
`lethal_audit.py` のものをそのまま使い、相手（手番の側）を攻め手にして回す。

主（応手の点）: 相手の手番の中で自分が答える判断点（カウンター・ブロッカー・確認・選択肢・対象）ごとに、
  各合法な応手を当てた後の盤面で相手の甲（本当の手札に対する詰み）を判定する。
  - **脅威が在った点** = 応手によって相手が勝ち切れる（甲 True）ものと勝ち切れない（甲 False）ものの両方が在る点。
  - **見逃し（甲）** = 脅威が在った点で CPU が「相手が勝ち切れる応手」を選んだ。
  - **乙** = 相手の見えない札（手札・山札・裏向きのライフ）を配り直した全部（本当＋攻めの強い配り方＋
    攻めの弱い配り方＋無作為 k）で甲 False の応手が在る点。**見逃し（乙）** = CPU の応手は本当の配り方で
    甲 True なのに、どの配り方でも守れる応手が在った（＝公平な CPU が原理的に防げたもの）。
副（終わり方）: 自分のメインの最後の 1〜2 判断点で、実際の手の代わりに各合法手 1 つを当て、既定解決
  （自分の対話は先頭の合法手・相手の応手は素通し・メインに戻ったらターン終了）で相手の次の手番の頭まで進め、
  そこで相手の詰みを判定する。実際の終わり方（実対局の相手の手番の頭）で相手に乙の詰み（全部の配り方で甲）が
  在り、別の 1 手なら無い（乙 False）点＝見逃し。厳密版（別の 1 手がどの配り方でも甲 False）も並べる。

使い方:
  python tests/scripts/defense_audit.py run --games 300 --workers 4      # 打って判定（JSONL）
  python tests/scripts/defense_audit.py summarize                        # 数える（summary.json）
  python tests/scripts/defense_audit.py verify --workers 4               # 見逃しの点を打ち直し、攻め手の手順を
                                                                         # 全部（キャラへのアタック・付与の並べ替え）
                                                                         # 読む判定器で確かめる（verify.json）
本線のコードは触らない（計器だけ・テストは足さない）。
"""
import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPCG_LOG_SILENT", "1")

import lethal_audit as LA  # noqa: E402  判定器・配り直し・盤面の要約を再利用
from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402

OUT_DIR = os.path.join("n1_results", "defense_audit")
MAIN = LA.MAIN
#: 自分が答える判断点として数える要求（相手の手番の中）。
RESPONSE_ACTIONS = ("SELECT_COUNTER", "SELECT_BLOCKER", "CONFIRM_OPTIONAL", "CONFIRM_TRIGGER", "CHOICE",
                    "SEARCH_AND_SELECT", "SELECT_TARGET", "SELECT_RESOURCE", "ARRANGE_DECK")


# --- 攻めの強さ（乙の「攻めの強い／弱い配り方」を作るためだけ）-------------------------------

_OFF: Dict[str, int] = {}


def offense_value(cid: str) -> int:
    """手札に在るときの攻めの値（打点を増やす文言＋メインのパワー上げ＋パワー）。"""
    if not _OFF:
        path = os.path.join(E.DATA, "opcg_cards.json")
        for c in json.load(open(path, encoding="utf-8")):
            text = c.get("効果(テキスト)") or ""
            v = 0
            try:
                v = int(c.get("パワー") or 0) // 1000
            except ValueError:
                v = 0
            ups = [int(x) for x in re.findall(r"パワー\+(\d+)", text)]
            if ups and "【カウンター】" not in text:
                v += max(ups) // 1000
            if any(w in text for w in LA.DANGER_WORDS):
                v += 20
            if "【速攻】" in text:
                v += 10
            if c.get("種類") == "イベント" and "【カウンター】" in text and not ups:
                v -= 5
            _OFF[c["number"]] = max(_OFF.get(c["number"], -99), v)
    return _OFF.get(cid, 0)


def redeal_offense(hidden: Dict[str, Any], opp: str, mode: str, rng: random.Random) -> Dict[str, Any]:
    """相手（攻め手）の手札＋山札＋裏向きのライフを配り直した hidden。
    mode: "atk"（攻めの値の高い順に手札へ）／"weak"（低い順）／"rand"。"""
    h = json.loads(json.dumps(hidden))
    p = h["players"][opp]
    life = p["life"]
    down_idx = [i for i, c in enumerate(life) if not c.get("is_face_up")]
    pool = list(p["hand"]) + list(p["deck"]) + [life[i] for i in down_idx]
    nh, nd = len(p["hand"]), len(p["deck"])
    rng.shuffle(pool)
    if mode == "atk":
        pool.sort(key=lambda c: -offense_value(c["card_id"]))
    elif mode == "weak":
        pool.sort(key=lambda c: offense_value(c["card_id"]))
    hand, rest = pool[:nh], pool[nh:]
    rng.shuffle(rest)
    p["hand"] = hand
    p["deck"] = rest[:nd]
    tail = rest[nd:]
    for j, i in enumerate(down_idx):
        life[i] = tail[j]
    for c in hand + p["deck"]:
        c["is_face_up"] = False
    return h


def _restorable(game) -> Optional[str]:
    """`from_hidden` で失われるものが無いか（対話スタック・誘発待ち）。あれば理由を返す。"""
    h = json.loads(game.hidden_json())
    if game.has_interaction or h.get("interaction_depth"):
        return "interaction"
    if h.get("pending_triggers"):
        return "pending_triggers"
    return None


def judge_deals(game, attacker: str, seconds: float, max_nodes: int, n_rand: int, seed: int,
                want: str) -> Dict[str, Any]:
    """配り直した全部で相手の甲を判定する。

    want="safe": 全部で甲 False なら True（どの配り方でも守れている）・1 つでも True なら False・残りは None。
    want="lethal": 全部で甲 True なら True（どの配り方でも詰み）・1 つでも False なら False・残りは None。
    本当の配り方は呼び出し側で判定済み（`true_res`）とし、ここでは配り直しだけを回す。"""
    reason = _restorable(game)
    if reason:
        return {"res": None, "skip": reason, "per": []}
    hidden = json.loads(game.hidden_json())
    pend0 = json.loads(game.pending_json())
    rng = random.Random(seed)
    modes = ["atk", "weak"] + ["rand"] * n_rand
    res: Optional[bool] = True
    per = []
    for m in modes:
        h = redeal_offense(hidden, attacker, m, rng)
        try:
            g = E.engine().Game.from_hidden(json.dumps(h, ensure_ascii=False), "p1", "p2", None)
            pend1 = json.loads(g.pending_json())
            if (pend1.get("action"), pend1.get("player_id")) != (pend0.get("action"), pend0.get("player_id")):
                raise RuntimeError(f"pending mismatch {pend0.get('action')} -> {pend1.get('action')}")
        except Exception as exc:  # noqa: BLE001
            per.append({"mode": m, "res": None, "crash": str(exc)[:200]})
            res = None
            continue
        j = LA.judge(g, attacker, seconds, max_nodes, None, False)
        j.pop("line", None)
        j["mode"] = m
        per.append(j)
        bad = (j["res"] is True) if want == "safe" else (j["res"] is False)
        if bad:
            res = False
            break
        if j["res"] is None:
            res = None
    return {"res": res, "per": per}


# --- 応手の列挙と既定解決 ---------------------------------------------------------------------

def move_key(mv: Dict[str, Any]) -> str:
    return json.dumps({"a": mv.get("action_type"), "c": mv.get("card_uuid"), "p": mv.get("payload") or {}},
                      sort_keys=True, ensure_ascii=False)


def response_kind(pend: Dict[str, Any], mv: Dict[str, Any]) -> str:
    """応手の型（集計用）。"""
    act, at = pend.get("action"), mv.get("action_type")
    if act == "SELECT_BLOCKER":
        return "no_block" if at == "PASS" else "block"
    if act == "SELECT_COUNTER":
        return "no_counter" if at == "PASS" else "counter"
    if act and act.startswith("CONFIRM"):
        return "confirm_" + ("yes" if (mv.get("payload") or {}).get("accepted", True) else "no")
    if act == "CHOICE":
        return "choice"
    return "target"


def settle_to_opp_main(game, me: str, opp: str, turn: int, limit: int = 120) -> Optional[str]:
    """既定解決で相手の次の手番の頭（相手の MAIN・対話なし）まで進める。
    自分の対話＝先頭の合法手・相手の応手（カウンター／ブロッカー）＝素通し・自分のメインに戻ったらターン終了。
    着いたら None・着けなければ理由。"""
    for _ in range(limit):
        if game.winner() is not None:
            return "winner:" + game.winner()
        pend = json.loads(game.pending_json())
        if not pend:
            return "no_pending"
        actor, act = pend["player_id"], pend.get("action")
        tc = int(game.turn_count)
        if actor == opp and act == MAIN and not game.has_interaction and tc == turn + 1:
            return None
        if tc > turn + 1:
            return "overshoot"
        if actor == me and act == MAIN and not game.has_interaction:
            mv = {"kind": "game", "action_type": "TURN_END", "payload": {}}
        else:
            legal = json.loads(game.legal_json(actor))
            if not legal:
                return "no_legal"
            mv = legal[0]
            if act in ("SELECT_COUNTER", "SELECT_BLOCKER"):
                mv = next((m for m in legal if m.get("action_type") == "PASS"), legal[0])
        try:
            LA.apply(game, actor, mv)
        except Exception as exc:  # noqa: BLE001
            return f"apply_error:{type(exc).__name__}"
    return "limit"


def _alt_order(mv: Dict[str, Any]) -> int:
    return {"TURN_END": 0, "PLAY": 1, "ACTIVATE_MAIN": 2, "ATTACH_DON": 3, "ATTACK": 4}.get(mv.get("action_type"), 5)


# --- 対局と記録 -----------------------------------------------------------------------------

class Audit:
    """1 局ぶんの状態（置換表・自分のメインの最後の点の控え）。"""

    def __init__(self, seed: int, a):
        self.seed, self.a = seed, a
        self.points: List[Dict[str, Any]] = []
        self.memos: Dict[tuple, Dict[str, Any]] = {}
        #: 席ごとの「今のメインの最後の判断点」（最大 2 つ・`(step, fork, move, legal)`）
        self.last_main: Dict[str, List[Tuple[int, Any, Dict[str, Any], List[Dict[str, Any]]]]] = {"p1": [], "p2": []}
        self.last_turn: Dict[str, int] = {"p1": -1, "p2": -1}
        self.pending_end: Dict[str, Optional[Tuple[int, List]]] = {"p1": None, "p2": None}
        self.stats = {"resp_points_seen": 0, "resp_points_audited": 0, "end_turns_seen": 0, "end_turns_audited": 0}

    def memo(self, attacker: str, turn: int) -> Dict[str, Any]:
        for k in [k for k in self.memos if k != (attacker, turn)]:
            del self.memos[k]
        return self.memos.setdefault((attacker, turn), {})

    # -- 主: 応手の点 --
    def response_point(self, game, name: str, turn: int, step: int, move: Dict[str, Any], pend: Dict[str, Any]):
        a = self.a
        opp = game.turn_player()
        h = json.loads(game.hidden_json())
        my_life = len(h["players"][name]["life"])
        self.stats["resp_points_seen"] += 1
        if turn < a.min_turn or my_life > a.max_life:
            return
        self.stats["resp_points_audited"] += 1
        memo = self.memo(opp, turn)
        alts = LA.expand(game, name, pend)
        cands: List[Dict[str, Any]] = []
        seen = set()
        for mv in [move] + alts:
            k = move_key(mv)
            if k in seen:
                continue
            seen.add(k)
            cands.append(mv)
        rows = []
        t0 = time.perf_counter()
        for i, mv in enumerate(cands):
            child = game.fork()
            try:
                LA.apply(child, name, mv)
            except Exception as exc:  # noqa: BLE001
                rows.append({"i": i, "cpu": i == 0, "move": LA.describe(game, mv), "kind": response_kind(pend, mv),
                             "apply_error": f"{type(exc).__name__}"[:60]})
                continue
            j = LA.judge(child, opp, a.sec, a.max_nodes, memo, False)
            line = j.pop("line", None)
            row = {"i": i, "cpu": i == 0, "move": LA.describe(game, mv), "kind": response_kind(pend, mv),
                   "ko": j}
            if i == 0 and line:
                row["line"] = line
            rows.append(row)
        ok = [r for r in rows if "ko" in r]
        cpu = ok[0] if ok and ok[0]["cpu"] else None
        any_true = any(r["ko"]["res"] is True for r in ok)
        safe = [r for r in ok if r["ko"]["res"] is False]
        threat = any_true and bool(safe)
        miss_ko = bool(cpu and cpu["ko"]["res"] is True and safe)
        rec = {"kind": "resp", "seed": self.seed, "seat": name, "turn": turn, "step": step,
               "action": pend.get("action"), "my_life": my_life, "opp_life": len(h["players"][opp]["life"]),
               "my_hand": len(h["players"][name]["hand"]), "n_cands": len(cands),
               "threat": threat, "miss_ko": miss_ko,
               "unknown_cpu": bool(cpu is None or cpu["ko"]["res"] is None),
               "unknown_any": any(r["ko"]["res"] is None for r in ok),
               "cpu_kind": response_kind(pend, move), "rows": rows}
        # 乙: 脅威が在った点で、守れる応手のうち「どの配り方でも守れる」ものが在るか（見つかったら止める）
        if threat:
            otsu_rows = []
            found = None
            unknown = False
            for r in safe:
                child = game.fork()
                LA.apply(child, name, cands[r["i"]])
                d = judge_deals(child, opp, a.sec_otsu, a.max_nodes, a.rand_hands,
                                self.seed * 1000 + step * 10 + r["i"], "safe")
                otsu_rows.append({"i": r["i"], "res": d["res"], "skip": d.get("skip"),
                                  "per": [{"mode": p.get("mode"), "res": p.get("res"), "sec": p.get("sec")} for p in d["per"]]})
                if d["res"] is True:
                    found = r["i"]
                    break
                if d["res"] is None:
                    unknown = True
            rec["otsu"] = {"safe_i": found, "unknown": unknown and found is None, "rows": otsu_rows}
            rec["miss_otsu"] = bool(miss_ko and found is not None)
            rec["threat_otsu"] = found is not None
        else:
            rec["miss_otsu"] = False
            rec["threat_otsu"] = False
        rec["sec"] = round(time.perf_counter() - t0, 3)
        if miss_ko or rec.get("threat_otsu"):
            rec["board"] = LA.summarize_board(game, name)
        self.points.append(rec)

    # -- 副: 終わり方 --
    def main_point(self, game, name: str, turn: int, step: int, move: Dict[str, Any]):
        if self.last_turn[name] != turn:
            self.last_main[name] = []
            self.last_turn[name] = turn
        legal = json.loads(game.legal_json(name))
        self.last_main[name].append((step, game.fork(), move, legal))
        self.last_main[name] = self.last_main[name][-2:]

    def arm_end(self, name: str, turn: int):
        """自分のターンが終わった（相手の手番の頭を待つ）。"""
        if self.last_main[name] and self.last_turn[name] == turn:
            self.pending_end[name] = (turn, self.last_main[name])
            self.last_main[name] = []

    def end_point(self, game, opp: str, turn: int, step: int):
        """相手（opp）の手番の頭＝自分（me）の終わり方を判定する。"""
        a = self.a
        me = "p2" if opp == "p1" else "p1"
        pe = self.pending_end.get(me)
        if not pe or pe[0] != turn - 1:
            return
        self.pending_end[me] = None
        my_turn, pts = pe
        h = json.loads(game.hidden_json())
        my_life = len(h["players"][me]["life"])
        self.stats["end_turns_seen"] += 1
        if my_turn < a.min_turn or my_life > a.max_life:
            return
        self.stats["end_turns_audited"] += 1
        memo = self.memo(opp, turn)
        t0 = time.perf_counter()
        actual = LA.judge(game, opp, a.sec_end, a.max_nodes, memo, False)
        line = actual.pop("line", None)
        rec = {"kind": "end", "seed": self.seed, "seat": me, "turn": my_turn, "step": step,
               "my_life": my_life, "opp_life": len(h["players"][opp]["life"]),
               "my_hand": len(h["players"][me]["hand"]), "actual_ko": actual, "actual_line": line,
               "actual_moves": [LA.describe(g, m) for (_s, g, m, _l) in pts]}
        rec["unknown_actual"] = actual["res"] is None
        if actual["res"] is True:
            d = judge_deals(game, opp, a.sec_otsu, a.max_nodes, a.rand_hands, self.seed * 1000 + step, "lethal")
            rec["actual_otsu"] = {"res": d["res"], "skip": d.get("skip"),
                                  "per": [{"mode": p.get("mode"), "res": p.get("res"), "sec": p.get("sec")} for p in d["per"]]}
        threat = actual["res"] is True and rec.get("actual_otsu", {}).get("res") is True
        rec["threat_otsu"] = threat
        rec["threat_ko"] = actual["res"] is True
        rec["alts"] = []
        if threat:
            # 別の 1 手: 最後の 1〜2 判断点の各合法手（実際の手を除く）
            for (s, g, m, legal) in pts:
                actual_key = move_key(m)
                alts = [x for x in legal if move_key(x) != actual_key]
                alts.sort(key=_alt_order)
                if len(alts) > a.max_alts:
                    rng = random.Random(self.seed * 1000 + s)
                    head = [x for x in alts if x.get("action_type") in ("TURN_END",)]
                    rest = [x for x in alts if x not in head]
                    rng.shuffle(rest)
                    alts = head + rest[:a.max_alts - len(head)]
                for i, mv in enumerate(alts):
                    c = g.fork()
                    row = {"point_step": s, "i": i, "move": LA.describe(g, mv), "at": mv.get("action_type")}
                    try:
                        LA.apply(c, me, mv)
                    except Exception as exc:  # noqa: BLE001
                        row["apply_error"] = f"{type(exc).__name__}"[:60]
                        rec["alts"].append(row)
                        continue
                    why = settle_to_opp_main(c, me, opp, my_turn)
                    if why:
                        row["settle"] = why
                        if why.startswith("winner:"):
                            row["ko"] = {"res": False if why == f"winner:{me}" else True, "sec": 0}
                            row["otsu"] = {"res": row["ko"]["res"]}
                        rec["alts"].append(row)
                        continue
                    j = LA.judge(c, opp, a.sec, a.max_nodes, None, False)
                    j.pop("line", None)
                    row["ko"] = j
                    if j["res"] is True:
                        d = judge_deals(c, opp, a.sec_otsu, a.max_nodes, a.rand_hands,
                                        self.seed * 1000 + s * 10 + i, "lethal")
                        row["otsu"] = {"res": d["res"], "skip": d.get("skip"),
                                       "per": [{"mode": p.get("mode"), "res": p.get("res")} for p in d["per"]]}
                    elif j["res"] is False:
                        row["otsu"] = {"res": False}
                        d = judge_deals(c, opp, a.sec_otsu, a.max_nodes, a.rand_hands,
                                        self.seed * 1000 + s * 10 + i, "safe")
                        row["safe"] = {"res": d["res"], "skip": d.get("skip"),
                                       "per": [{"mode": p.get("mode"), "res": p.get("res")} for p in d["per"]]}
                    rec["alts"].append(row)
            alts = rec["alts"]
            rec["miss"] = any((r.get("otsu") or {}).get("res") is False for r in alts)
            rec["miss_strict"] = any((r.get("safe") or {}).get("res") is True for r in alts)
            rec["unknown_alts"] = any("ko" in r and r["ko"]["res"] is None for r in alts)
            rec["board"] = LA.summarize_board(game, me)
        else:
            rec["miss"] = rec["miss_strict"] = False
        rec["sec"] = round(time.perf_counter() - t0, 3)
        self.points.append(rec)


def play_one(args) -> List[Dict[str, Any]]:
    seed, a = args
    E.engine()
    db = D.load_db()
    kw = {"budget": a.budget}
    if a.worlds > 1:
        kw["worlds"] = a.worlds
    if a.lethal_proof and a.lethal_proof != "off":
        kw["lethal_proof"] = a.lethal_proof
    if getattr(a, "commit_sig_legacy", False):
        kw["commit_sig_legacy"] = True       # WP `rs-commit-sig` の前後比較（旧い署名＝修正前の打ち方）
    if getattr(a, "commit_hidden_legacy", False):
        kw["commit_hidden_legacy"] = True    # §7.39 の前後比較（見えていなかった札を選ぶ続きも積む＝修正前の打ち方）
    for _k in ("window_carry_legacy", "window_worlds_legacy", "window_tree_legacy"):
        if getattr(a, "window_legacy", False) or getattr(a, _k, False):
            kw[_k] = True                    # §7.42 の前後比較（窓・対話の決め方の旧い形）
    if getattr(a, "window_tree_counter_legacy", False):
        kw["window_tree_counter_legacy"] = True  # §7.45 の前後比較（カウンターの窓も短い木で読む旧い形）
    if getattr(a, "window_worlds", None):
        kw["window_worlds"] = a.window_worlds
    if getattr(a, "confirm_trigger_legacy", False):
        kw["confirm_trigger_legacy"] = True  # §7.44 の (B) の前後比較（トリガーを断らない＝修正前の打ち方）
    spec = E.SeatSpec(a.net, sims=a.sims, **kw)
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "synth_roles")
    au = Audit(seed, a)
    import hashlib
    deck_md5 = hashlib.md5(json.dumps([la, lb, p1, p2], sort_keys=True, default=str).encode()).hexdigest()[:12]

    def obs(game, name, turn, step, out, move):
        pend = json.loads(game.pending_json())
        act = pend.get("action")
        tp = game.turn_player()
        if tp == name:
            if act == MAIN and not game.has_interaction:
                # 相手の手番の頭（前の自分＝相手の終わり方の判定）と、自分のメインの判断点の控え
                au.end_point(game, name, turn, step)
                au.main_point(game, name, turn, step, move)
                if move.get("action_type") == "TURN_END":
                    au.arm_end(name, turn)
            return
        if act in RESPONSE_ACTIONS and act != MAIN:
            au.response_point(game, name, turn, step, move, pend)

    t0 = time.perf_counter()
    try:
        r = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, observer=obs)
        winner, turns, void = r["winner"], r["turns"], r["winner"] is None
    except DR.GameAborted as exc:
        winner, turns, void = None, None, str(exc)
    return [{"game": {"seed": seed, "winner": winner, "turns": turns, "void": void, "deck_md5": deck_md5,
                      "sec": round(time.perf_counter() - t0, 1), "points": len(au.points), "stats": au.stats}}] + au.points


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"points_{a.seed_base}_{a.games}.jsonl")
    if a.seeds:
        path = os.path.join(a.out, f"points_seeds_{os.path.basename(a.seeds).split('.')[0]}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            if "game" in d:
                done.add(d["game"]["seed"])
    planned = ([int(x) for x in open(a.seeds).read().split()] if a.seeds
               else [a.seed_base + i for i in range(a.games)])
    seeds = [s for s in planned if s not in done]
    print(f"todo {len(seeds)} (done {len(done)}) -> {path}", flush=True)
    import multiprocessing as mp
    with open(path, "a", encoding="utf-8") as f, mp.get_context("fork").Pool(a.workers) as pool:
        for i, recs in enumerate(pool.imap_unordered(play_one, [(s, a) for s in seeds])):
            for r in recs[1:] + recs[:1]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            g = recs[0]["game"]
            print(f"[{i + 1}/{len(seeds)}] seed={g['seed']} pts={g['points']} sec={g['sec']} winner={g['winner']} "
                  f"stats={g['stats']}", flush=True)


# --- 確かめ: 見逃しの点を打ち直し、攻め手の手順を全部読む判定器で判定し直す -----------------------

def judge_full(game, attacker: str, seconds: float, max_nodes: int) -> Dict[str, Any]:
    """`lethal_audit.judge` と同じだが、キャラへのアタックと付与の並べ替えも手順に入れる（近似を外す）。
    見つかる詰みは増えるだけ＝「守れる」と判定した応手が本当に守れるかを確かめるために使う。"""
    s = LA.Solver(attacker, int(game.turn_count), seconds, max_nodes, None, char_attacks=True, full_don=True)
    t0 = time.perf_counter()
    try:
        r: Optional[bool] = s.win(game.fork())
    except LA.Unknown:
        r = None
    except Exception as exc:  # noqa: BLE001
        r = None
        s.crash = f"{type(exc).__name__}: {exc}"[:200]
    out = {"res": r, "nodes": s.nodes, "sec": round(time.perf_counter() - t0, 3)}
    if s.crash:
        out["crash"] = s.crash
    return out


def judge_deals_full(game, attacker: str, seconds: float, max_nodes: int, n_rand: int, seed: int,
                     want: str) -> Dict[str, Any]:
    reason = _restorable(game)
    if reason:
        return {"res": None, "skip": reason, "per": []}
    hidden = json.loads(game.hidden_json())
    rng = random.Random(seed)
    res: Optional[bool] = True
    per = []
    for m in ["atk", "weak"] + ["rand"] * n_rand:
        h = redeal_offense(hidden, attacker, m, rng)
        try:
            g = E.engine().Game.from_hidden(json.dumps(h, ensure_ascii=False), "p1", "p2", None)
        except Exception as exc:  # noqa: BLE001
            per.append({"mode": m, "res": None, "crash": str(exc)[:200]})
            res = None
            continue
        j = judge_full(g, attacker, seconds, max_nodes)
        j["mode"] = m
        per.append(j)
        bad = (j["res"] is True) if want == "safe" else (j["res"] is False)
        if bad:
            res = False
            break
        if j["res"] is None:
            res = None
    return {"res": res, "per": per}


class _Stop(Exception):
    pass


def replay_to(seed: int, a, seat: str, step: int):
    """同じ seed・同じ読みで打ち直し、`step` の決める前の盤面（と決めた手）を返す（決定論の契約）。"""
    E.engine()
    db = D.load_db()
    kw = {"budget": a.budget}
    if a.worlds > 1:
        kw["worlds"] = a.worlds
    if a.lethal_proof and a.lethal_proof != "off":
        kw["lethal_proof"] = a.lethal_proof
    if getattr(a, "commit_sig_legacy", False):
        kw["commit_sig_legacy"] = True       # WP `rs-commit-sig` の前後比較（旧い署名＝修正前の打ち方）
    if getattr(a, "commit_hidden_legacy", False):
        kw["commit_hidden_legacy"] = True    # §7.39 の前後比較（見えていなかった札を選ぶ続きも積む＝修正前の打ち方）
    for _k in ("window_carry_legacy", "window_worlds_legacy", "window_tree_legacy"):
        if getattr(a, "window_legacy", False) or getattr(a, _k, False):
            kw[_k] = True                    # §7.42 の前後比較（窓・対話の決め方の旧い形）
    if getattr(a, "window_tree_counter_legacy", False):
        kw["window_tree_counter_legacy"] = True  # §7.45 の前後比較（カウンターの窓も短い木で読む旧い形）
    if getattr(a, "window_worlds", None):
        kw["window_worlds"] = a.window_worlds
    if getattr(a, "confirm_trigger_legacy", False):
        kw["confirm_trigger_legacy"] = True  # §7.44 の (B) の前後比較（トリガーを断らない＝修正前の打ち方）
    spec = E.SeatSpec(a.net, sims=a.sims, **kw)
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "synth_roles")
    got = {}

    def obs(game, name, turn, st, out, move):
        if st == step:
            if name != seat:
                raise RuntimeError(f"replay mismatch: step {step} is {name}'s, expected {seat}")
            got["game"], got["move"], got["out"], got["turn"] = game.fork(), move, out, turn
            raise _Stop()

    try:
        DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, observer=obs)
    except _Stop:
        pass
    if "game" not in got:
        raise RuntimeError(f"replay: step {step} not reached")
    got["spec"] = spec
    return got["game"], got["move"], got


def verify_one(args) -> Dict[str, Any]:
    rec, a = args
    t0 = time.perf_counter()
    out = {"kind": rec["kind"], "seed": rec["seed"], "seat": rec["seat"], "turn": rec["turn"], "step": rec["step"]}
    try:
        if rec["kind"] == "resp":
            game, move, got = replay_to(rec["seed"], a, rec["seat"], rec["step"])
            name = rec["seat"]
            # 実対局の決め方（木で読んだか・箱の commit の続きを打っただけか）と、持ち越し無しで読み直したときの手
            live = got["out"]
            out["live_kind"] = live.get("kind")
            out["live_commit_left"] = len(live.get("commit") or [])
            fresh = json.loads(game.decide(name, got["spec"].decide_opts(rec["seed"], got["turn"], name, None)))
            out["fresh_kind"] = fresh.get("kind")
            out["fresh_move"] = LA.describe(game, fresh["move"]) if fresh.get("move") else None
            opp = game.turn_player()
            pend = json.loads(game.pending_json())
            cands, seen = [], set()
            for mv in [move] + LA.expand(game, name, pend):
                k = move_key(mv)
                if k not in seen:
                    seen.add(k)
                    cands.append(mv)
            if LA.describe(game, cands[0]) != rec["rows"][0]["move"]:
                out["error"] = "cpu move differs on replay"
                return out
            rows = []
            still_safe = []
            for r in rec["rows"]:
                if "ko" not in r or r["ko"]["res"] is not False:
                    continue
                mv = cands[r["i"]] if r["i"] < len(cands) else None
                if mv is None or LA.describe(game, mv) != r["move"]:
                    rows.append({"i": r["i"], "error": "candidate differs on replay"})
                    continue
                child = game.fork()
                LA.apply(child, name, mv)
                j = judge_full(child, opp, a.sec_verify, a.max_nodes)
                row = {"i": r["i"], "move": r["move"], "ko_full": j}
                if j["res"] is False:
                    still_safe.append(r["i"])
                    if rec.get("miss_otsu") and (rec.get("otsu") or {}).get("safe_i") == r["i"]:
                        d = judge_deals_full(child, opp, a.sec_verify_otsu, a.max_nodes, a.rand_hands,
                                             rec["seed"] * 1000 + rec["step"] * 10 + r["i"], "safe")
                        row["otsu_full"] = {"res": d["res"], "skip": d.get("skip"),
                                            "per": [{"mode": p.get("mode"), "res": p.get("res")} for p in d["per"]]}
                rows.append(row)
            out["rows"] = rows
            safe_moves = {r["move"] for r in rows if r.get("ko_full", {}).get("res") is False}
            out["fresh_safe"] = out["fresh_move"] in safe_moves if out.get("fresh_move") else None
            out["fresh_same_as_live"] = out["fresh_move"] == rec["rows"][0]["move"]
            out["miss_ko_full"] = bool(still_safe)
            o = next((r for r in rows if r.get("otsu_full")), None)
            out["miss_otsu_full"] = bool(rec.get("miss_otsu") and o and o["otsu_full"]["res"] is True)
            out["otsu_full_unknown"] = bool(rec.get("miss_otsu") and o and o["otsu_full"]["res"] is None)
            out["ko_full_unknown"] = bool(not still_safe and any(r.get("ko_full", {}).get("res") is None for r in rows))
        else:
            # 終わり方: 乙の詰みが無いと判定した別の 1 手を、全部読む判定器で判定し直す
            me = rec["seat"]
            opp = "p2" if me == "p1" else "p1"
            rows = []
            still = []
            for r in rec["alts"]:
                if (r.get("otsu") or {}).get("res") is not False or r.get("settle") == f"winner:{me}":
                    continue
                game, _, _got = replay_to(rec["seed"], a, me, r["point_step"])
                legal = json.loads(game.legal_json(me))
                mv = next((x for x in legal if LA.describe(game, x) == r["move"]), None)
                if mv is None:
                    rows.append({"point_step": r["point_step"], "i": r["i"], "error": "alt differs on replay"})
                    continue
                c = game.fork()
                LA.apply(c, me, mv)
                why = settle_to_opp_main(c, me, opp, rec["turn"])
                row = {"point_step": r["point_step"], "i": r["i"], "move": r["move"]}
                if why:
                    row["settle"] = why
                    row["otsu_full"] = {"res": (why != f"winner:{me}")} if why.startswith("winner:") else {"res": None}
                else:
                    j = judge_full(c, opp, a.sec_verify, a.max_nodes)
                    row["ko_full"] = j
                    if j["res"] is False:
                        row["otsu_full"] = {"res": False}
                        d = judge_deals_full(c, opp, a.sec_verify_otsu, a.max_nodes, a.rand_hands,
                                             rec["seed"] * 1000 + r["point_step"] * 10 + r["i"], "safe")
                        row["safe_full"] = {"res": d["res"], "per": [{"mode": p.get("mode"), "res": p.get("res")} for p in d["per"]]}
                    elif j["res"] is True:
                        d = judge_deals_full(c, opp, a.sec_verify_otsu, a.max_nodes, a.rand_hands,
                                             rec["seed"] * 1000 + r["point_step"] * 10 + r["i"], "lethal")
                        row["otsu_full"] = {"res": d["res"], "per": [{"mode": p.get("mode"), "res": p.get("res")} for p in d["per"]]}
                    else:
                        row["otsu_full"] = {"res": None}
                if (row.get("otsu_full") or {}).get("res") is False:
                    still.append(r["i"])
                rows.append(row)
            out["rows"] = rows
            out["miss_full"] = bool(still)
            out["miss_strict_full"] = any((r.get("safe_full") or {}).get("res") is True for r in rows)
            out["unknown_full"] = bool(not still and any((r.get("otsu_full") or {}).get("res") is None for r in rows))
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
    out["sec"] = round(time.perf_counter() - t0, 1)
    return out


def cmd_verify(a):
    path = a.points or os.path.join(a.out, f"points_{a.seed_base}_{a.games}.jsonl")
    _games, pts = _load(path)
    todo = [p for p in pts if (p["kind"] == "resp" and p["miss_ko"])
            or (p["kind"] == "end" and any((r.get("otsu") or {}).get("res") is False
                                           and r.get("settle") != f"winner:{p['seat']}" for r in p["alts"]))]
    print(f"verify {len(todo)} miss points", flush=True)
    import multiprocessing as mp
    outp = os.path.join(a.out, "verify.jsonl")
    with open(outp, "w", encoding="utf-8") as f, mp.get_context("fork").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(verify_one, [(p, a) for p in todo])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(todo)}] {r['kind']} seed={r['seed']} step={r['step']} "
                  f"{ {k: v for k, v in r.items() if k.endswith('_full') or k == 'error'} } sec={r['sec']}", flush=True)


def _rate(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None}


def _load(path):
    games, pts = {}, []
    pending: List[Dict[str, Any]] = []
    import gzip
    fh = gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")
    for line in fh:
        d = json.loads(line)
        if "game" in d:
            games[d["game"]["seed"]] = d["game"]
            pts.extend(p for p in pending if p["seed"] == d["game"]["seed"])
            pending = [p for p in pending if p["seed"] != d["game"]["seed"]]
        else:
            pending.append(d)
    return games, pts


def _count(items, key):
    out: Dict[str, int] = {}
    for x in items:
        out[str(key(x))] = out.get(str(key(x)), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def cmd_summarize(a):
    path = a.points or os.path.join(a.out, f"points_{a.seed_base}_{a.games}.jsonl")
    games, pts = _load(path)
    resp = [p for p in pts if p["kind"] == "resp"]
    end = [p for p in pts if p["kind"] == "end"]
    n = len(resp)
    threat = [p for p in resp if p["threat"]]
    miss_ko = [p for p in resp if p["miss_ko"]]
    threat_ot = [p for p in resp if p.get("threat_otsu")]
    miss_ot = [p for p in resp if p.get("miss_otsu")]
    # 手番単位（同じ局・席・相手のターン）
    turns: Dict[tuple, List[Dict[str, Any]]] = {}
    for p in resp:
        turns.setdefault((p["seed"], p["seat"], p["turn"]), []).append(p)
    t_threat = [v for v in turns.values() if any(x["threat"] for x in v)]
    t_miss_ko = [v for v in turns.values() if any(x["miss_ko"] for x in v)]
    t_threat_ot = [v for v in turns.values() if any(x.get("threat_otsu") for x in v)]
    t_miss_ot = [v for v in turns.values() if any(x.get("miss_otsu") for x in v)]

    def lost(seat_set):
        return sum(1 for s, seat in seat_set if games[s]["winner"] not in (None, seat))

    def lost_this_turn(v):
        s = v[0]["seed"]
        return games[s]["winner"] not in (None, v[0]["seat"]) and games[s]["turns"] == v[0]["turn"]

    g_miss_ko = {(p["seed"], p["seat"]) for p in miss_ko}
    g_miss_ot = {(p["seed"], p["seat"]) for p in miss_ot}
    secs = sorted(p["sec"] for p in resp)

    def cpu_kind(p):
        return p["cpu_kind"]

    def miss_type(p):
        # 見逃した応手の型: CPU の応手と、守れた応手の型
        safe = [r["kind"] for r in p["rows"] if "ko" in r and r["ko"]["res"] is False]
        return f"{p['action']}:{p['cpu_kind']}->" + "/".join(sorted(set(safe)))

    summ = {
        "games": len(games), "void_games": sum(1 for g in games.values() if g["void"]),
        "resp": {
            "points_seen": sum(g["stats"]["resp_points_seen"] for g in games.values()),
            "points": n, "turns": len(turns),
            "by_action": _count(resp, lambda p: p["action"]),
            "threat": _rate(len(threat), n), "threat_otsu": _rate(len(threat_ot), n),
            "unknown_cpu": _rate(sum(1 for p in resp if p["unknown_cpu"]), n),
            "unknown_any": _rate(sum(1 for p in resp if p["unknown_any"]), n),
            "unknown_otsu_among_threat": _rate(sum(1 for p in threat if (p.get("otsu") or {}).get("unknown")), len(threat)),
            "miss_per_point": {"ko": _rate(len(miss_ko), n), "otsu": _rate(len(miss_ot), n)},
            "miss_given_threat": {"ko": _rate(len(miss_ko), len(threat)), "otsu": _rate(len(miss_ot), len(threat_ot))},
            "miss_per_turn": {"ko": _rate(len(t_miss_ko), len(turns)), "otsu": _rate(len(t_miss_ot), len(turns)),
                              "turns_with_threat": len(t_threat), "turns_with_threat_otsu": len(t_threat_ot),
                              "ko_given_threat": _rate(len(t_miss_ko), len(t_threat)),
                              "otsu_given_threat": _rate(len(t_miss_ot), len(t_threat_ot))},
            "miss_per_game": {"ko": _rate(len({s for s, _ in g_miss_ko}), len(games)),
                              "otsu": _rate(len({s for s, _ in g_miss_ot}), len(games)),
                              "ko_seats": len(g_miss_ko), "otsu_seats": len(g_miss_ot),
                              "ko_missed_then_lost": lost(g_miss_ko), "otsu_missed_then_lost": lost(g_miss_ot),
                              "otsu_missed_lost_this_turn": sum(1 for v in t_miss_ot if lost_this_turn(v)),
                              "ko_missed_lost_this_turn": sum(1 for v in t_miss_ko if lost_this_turn(v))},
            "miss_ko_by_action": _count(miss_ko, lambda p: p["action"]),
            "miss_ko_type": _count(miss_ko, miss_type),
            "miss_otsu_type": _count(miss_ot, miss_type),
            "threat_by_action": _count(threat, lambda p: p["action"]),
            "miss_otsu_my_life": _count(miss_ot, lambda p: p["my_life"]),
            "miss_otsu_opp_life": _count(miss_ot, lambda p: p["opp_life"]),
            "miss_ko_my_life": _count(miss_ko, lambda p: p["my_life"]),
            "threat_my_life": _count(threat, lambda p: p["my_life"]),
            "cands_mean": round(sum(p["n_cands"] for p in resp) / n, 2) if n else None,
            "sec": {"mean": round(sum(secs) / n, 3) if n else None, "p50": secs[n // 2] if n else None,
                    "p99": secs[int(n * 0.99)] if n else None, "max": secs[-1] if n else None},
        },
    }
    m = len(end)

    def _own_win(p, r):
        return (r.get("settle") or "").startswith("winner:") and r["settle"] == f"winner:{p['seat']}"

    for p in end:
        # 既定解決（相手は素通し）でそのまま勝ってしまう別の 1 手は「自分の詰み」（§7.13）の問いなので外す
        alts = [r for r in p["alts"] if not _own_win(p, r)]
        p["alts_own_win"] = sum(1 for r in p["alts"] if _own_win(p, r))
        p["alts"] = alts
        p["miss"] = any((r.get("otsu") or {}).get("res") is False for r in alts)
        p["miss_strict"] = any((r.get("safe") or {}).get("res") is True for r in alts)
    e_threat_ko = [p for p in end if p["threat_ko"]]
    e_threat = [p for p in end if p["threat_otsu"]]
    e_miss = [p for p in end if p["miss"]]
    e_strict = [p for p in end if p["miss_strict"]]
    e_secs = sorted(p["sec"] for p in end)

    def alt_type(p, strict):
        key = "safe" if strict else "otsu"
        ok = [r["at"] for r in p["alts"] if ((r.get(key) or {}).get("res") is (True if strict else False))]
        return "/".join(sorted(set(ok)))

    summ["end"] = {
        "turns_seen": sum(g["stats"]["end_turns_seen"] for g in games.values()),
        "turns": m,
        "threat_ko": _rate(len(e_threat_ko), m), "threat_otsu": _rate(len(e_threat), m),
        "unknown_actual": _rate(sum(1 for p in end if p["unknown_actual"]), m),
        "unknown_actual_otsu_among_ko": _rate(sum(1 for p in e_threat_ko if (p.get("actual_otsu") or {}).get("res") is None),
                                              len(e_threat_ko)),
        "miss_per_turn": _rate(len(e_miss), m), "miss_strict_per_turn": _rate(len(e_strict), m),
        "miss_given_threat": _rate(len(e_miss), len(e_threat)),
        "miss_strict_given_threat": _rate(len(e_strict), len(e_threat)),
        "unknown_alts_among_threat": _rate(sum(1 for p in e_threat if p.get("unknown_alts")), len(e_threat)),
        "miss_games": len({p["seed"] for p in e_miss}),
        "miss_then_lost": lost({(p["seed"], p["seat"]) for p in e_miss}),
        "miss_lost_next_turn": sum(1 for p in e_miss if games[p["seed"]]["winner"] not in (None, p["seat"])
                                   and games[p["seed"]]["turns"] == p["turn"] + 1),
        "miss_strict_then_lost": lost({(p["seed"], p["seat"]) for p in e_strict}),
        "miss_alt_type": _count(e_miss, lambda p: alt_type(p, False)),
        "miss_strict_alt_type": _count(e_strict, lambda p: alt_type(p, True)),
        "miss_my_life": _count(e_miss, lambda p: p["my_life"]),
        "threat_turns_with_own_win_alt": sum(1 for p in e_threat if p.get("alts_own_win")),
        "threat_my_life": _count(e_threat, lambda p: p["my_life"]),
        "alts_mean_among_threat": round(sum(len(p["alts"]) for p in e_threat) / len(e_threat), 1) if e_threat else None,
        "sec": {"mean": round(sum(e_secs) / m, 3) if m else None, "p50": e_secs[m // 2] if m else None,
                "p99": e_secs[int(m * 0.99)] if m else None, "max": e_secs[-1] if m else None},
    }
    examples = []
    for p in sorted(miss_ot, key=lambda p: (p["seed"], p["turn"]))[:7] + \
            sorted([p for p in miss_ko if not p.get("miss_otsu")], key=lambda p: (p["seed"], p["turn"]))[:3]:
        examples.append({"kind": "resp_otsu" if p.get("miss_otsu") else "resp_ko_only",
                         "seed": p["seed"], "seat": p["seat"], "turn": p["turn"], "step": p["step"],
                         "action": p["action"], "my_life": p["my_life"], "opp_life": p["opp_life"],
                         "cpu_move": p["rows"][0]["move"], "cpu_line": p["rows"][0].get("line"),
                         "safe_moves": [r["move"] for r in p["rows"] if "ko" in r and r["ko"]["res"] is False][:4],
                         "otsu_safe_i": (p.get("otsu") or {}).get("safe_i"),
                         "board": p.get("board"), "game_winner": games[p["seed"]]["winner"],
                         "game_turns": games[p["seed"]]["turns"]})
    for p in sorted(e_miss, key=lambda p: (p["seed"], p["turn"]))[:5]:
        examples.append({"kind": "end_strict" if p["miss_strict"] else "end",
                         "seed": p["seed"], "seat": p["seat"], "turn": p["turn"],
                         "my_life": p["my_life"], "opp_life": p["opp_life"],
                         "actual_moves": p["actual_moves"], "actual_line": p.get("actual_line"),
                         "alts_without_otsu": [r["move"] for r in p["alts"] if (r.get("otsu") or {}).get("res") is False][:5],
                         "alts_safe": [r["move"] for r in p["alts"] if (r.get("safe") or {}).get("res") is True][:5],
                         "board": p.get("board"), "game_winner": games[p["seed"]]["winner"],
                         "game_turns": games[p["seed"]]["turns"]})
    vpath = os.path.join(a.out, "verify.jsonl")
    if os.path.exists(vpath):
        vs = [json.loads(line) for line in open(vpath, encoding="utf-8")]
        vr = [v for v in vs if v["kind"] == "resp"]
        ve = [v for v in vs if v["kind"] == "end"]
        key = lambda v: (v["seed"], v["seat"], v["step"])  # noqa: E731
        vmap = {key(v): v for v in vs}
        ok_r = [v for v in vr if "error" not in v]
        ko_full = [v for v in ok_r if v["miss_ko_full"]]
        ot_full = [v for v in ok_r if v["miss_otsu_full"]]
        t_ot_full = {(v["seed"], v["seat"], v["turn"]) for v in ot_full}
        t_ko_full = {(v["seed"], v["seat"], v["turn"]) for v in ko_full}
        summ["verify"] = {
            "resp": {"verified": len(vr), "errors": sum(1 for v in vr if "error" in v),
                     "miss_ko_raw": len(miss_ko), "miss_ko_full": len(ko_full),
                     "miss_ko_full_unknown": sum(1 for v in ok_r if v.get("ko_full_unknown")),
                     "miss_otsu_raw": len(miss_ot), "miss_otsu_full": len(ot_full),
                     "miss_otsu_full_unknown": sum(1 for v in ok_r if v.get("otsu_full_unknown")),
                     "turns_miss_otsu_full": len(t_ot_full), "turns_miss_ko_full": len(t_ko_full),
                     "games_miss_otsu_full": len({s for s, _, _ in t_ot_full}),
                     "otsu_full_then_lost": lost({(s, seat) for s, seat, _ in t_ot_full}),
                     "otsu_full_lost_this_turn": sum(1 for v in t_miss_ot if (v[0]["seed"], v[0]["seat"], v[0]["turn"]) in t_ot_full
                                                     and lost_this_turn(v)),
                     "miss_given_threat_full": {"ko": _rate(len(ko_full), len(threat)), "otsu": _rate(len(ot_full), len(threat_ot))},
                     "live_kind_among_miss_ko_full": _count(ko_full, lambda v: v.get("live_kind")),
                     "live_kind_among_miss_otsu_full": _count(ot_full, lambda v: v.get("live_kind")),
                     "fresh_safe_among_miss_ko_full": _count(ko_full, lambda v: v.get("fresh_safe")),
                     "fresh_safe_among_miss_otsu_full": _count(ot_full, lambda v: v.get("fresh_safe")),
                     "fresh_safe_by_live_kind_otsu_full": _count(ot_full, lambda v: f"{v.get('live_kind')}:{v.get('fresh_safe')}"),
                     "miss_otsu_full_type": _count([p for p in miss_ot if vmap.get(key(p), {}).get("miss_otsu_full")], miss_type),
                     "miss_otsu_full_my_life": _count([p for p in miss_ot if vmap.get(key(p), {}).get("miss_otsu_full")],
                                                      lambda p: p["my_life"])},
            "end": {"verified": len(ve), "errors": sum(1 for v in ve if "error" in v),
                    "miss_raw": len(e_miss), "miss_full": sum(1 for v in ve if v.get("miss_full")),
                    "miss_strict_full": sum(1 for v in ve if v.get("miss_strict_full")),
                    "miss_full_unknown": sum(1 for v in ve if v.get("unknown_full")),
                    "miss_full_then_lost": lost({(v["seed"], v["seat"]) for v in ve if v.get("miss_full")})},
        }
        for ex in examples:
            v = vmap.get((ex["seed"], ex["seat"], ex.get("step")))
            if v:
                ex["verify"] = {k: v[k] for k in v if k.endswith("_full") or k == "error"}
    summ["examples"] = examples
    out = os.path.join(a.out, "summary.json")
    json.dump(summ, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "examples"}, ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "summarize", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--seed-base", type=int, default=3013000)
        p.add_argument("--games", type=int, default=300)
        p.add_argument("--out", default=OUT_DIR)
        if name in ("run", "verify"):
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--sims", type=int, default=E.PLAY_SIMS)
            p.add_argument("--budget", type=int, default=E.PLAY_BUDGET)
            p.add_argument("--worlds", type=int, default=1)
            p.add_argument("--net", default=None)
            p.add_argument("--min-turn", type=int, default=4)
            p.add_argument("--max-life", type=int, default=3, help="自分のライフがこれ以下の点だけ対象にする")
            p.add_argument("--sec", type=float, default=5.0, help="甲の 1 応手あたりの時間の上限")
            p.add_argument("--sec-otsu", type=float, default=3.0, help="乙の配り方 1 つあたりの時間の上限")
            p.add_argument("--sec-end", type=float, default=10.0,
                           help="副の『実際の終わり方』（相手の手番の頭）の甲の時間の上限（置換表は同じ手番の応手の点と共有）")
            p.add_argument("--max-nodes", type=int, default=200000)
            p.add_argument("--rand-hands", type=int, default=4)
            p.add_argument("--max-alts", type=int, default=16, help="副（終わり方）で試す別の 1 手の上限")
            p.add_argument("--lethal-proof", default=None, choices=("off", "root", "root_reply"),
                           help="両席の CPU の詰みの証明（§7.21）。既定は off＝§7.13 と同じ設定")
            p.add_argument("--commit-sig-legacy", action="store_true",
                           help="両席を旧い手の署名で打たせる（WP `rs-commit-sig` の前後比較・測定専用）")
            p.add_argument("--commit-hidden-legacy", action="store_true",
                           help="両席を、見えていなかった札を選ぶ commit の続きも積む旧い形で打たせる（§7.39 の前後比較・測定専用）")
            p.add_argument("--window-legacy", action="store_true",
                           help="両席を §7.42 の旧い形 3 つで打たせる（窓・対話の続きを積む・窓の世界 1 本・ライフが少ない戦闘の応手を静的な値で）")
            p.add_argument("--window-carry-legacy", action="store_true", help="§7.42 の 1 だけ旧い形")
            p.add_argument("--window-worlds-legacy", action="store_true", help="§7.42 の 2 だけ旧い形")
            p.add_argument("--window-tree-legacy", action="store_true", help="§7.42 の 3 だけ旧い形")
            p.add_argument("--window-tree-counter-legacy", action="store_true",
                           help="ライフが少ないカウンターの窓も短い木で読む旧い形（§7.45 の前後比較・測定専用）")
            p.add_argument("--window-worlds", type=int, default=None, help="窓の根畳みの世界の本数（省略＝Rust の既定）")
            p.add_argument("--confirm-trigger-legacy", action="store_true",
                           help="両席を、トリガーの発動確認を「発動する」の 1 手だけで打つ旧い形で打たせる（§7.44 の (B) の前後比較・測定専用）")
        if name == "run":
            p.add_argument("--seeds", default=None, help="seed の一覧のファイル（空白区切り）。指定すると --seed-base/--games の代わりにこれだけ回す")
        if name == "verify":
            p.add_argument("--sec-verify", type=float, default=30.0, help="確かめの甲の 1 応手あたりの時間の上限")
            p.add_argument("--sec-verify-otsu", type=float, default=10.0, help="確かめの乙の配り方 1 つあたりの時間の上限")
        if name != "run":
            p.add_argument("--points", default=None)
    a = ap.parse_args(argv)
    {"run": cmd_run, "summarize": cmd_summarize, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
