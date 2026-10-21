"""詰みの見逃しの監査（`docs/search_efficiency_plan.md` §7.13・WP `lethal-audit`・台帳 2+312）。

問い: CPU（出荷の読み＝1 本 × 1,280 回・戦闘箱の枝予算なし）が、**このターンに勝ち切れる局面で
勝ち切れていない**ことがどれだけあるか。

- **(甲) 本当の手札に対する詰み**: 本当の盤面（相手の手札・ライフの並び・両者の山札の並び）のまま、
  こちらの手番の残りを総当たりして、相手の最善応手（カウンター／ブロッカー／トリガー・効果の選択）に
  対しても相手を倒せる手順があるか。
- **(乙) どんな手札でも詰み**: 相手の手札（とライフ・山札の並び）が見えていない前提で、どの配り方でも
  勝てる手順があるか。全部の配り方は数えられないので「本当の配り方＋守りの強い配り方 2 つ＋無作為 k 個」の
  **全部で**甲が成り立つことで近似する（必要条件側の近似＝乙を多めに数えうる・notes 参照）。乙 ⊂ 甲。

判定器は `Game.fork()`（真の盤面の複製）に `legal_json`／`apply_*` を当てる AND-OR の総当たり
（自分の判断＝どれか 1 つ、相手の判断＝全部）。深さはこのターンの終わりまで。同じ盤面は置換表で
1 回だけ読む（ドン!! の uuid は捨てて「どこに何枚」に潰す）。時間か節点の上限を超えたら「判定不能」。

使い方:
  python tests/scripts/lethal_audit.py run --games 1000 --workers 4      # 打って判定（JSONL）
  python tests/scripts/lethal_audit.py summarize                         # 数える（summary.json）
本線のコードは触らない（計器だけ）。
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("OPCG_LOG_SILENT", "1")

from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402

OUT_DIR = os.path.join("n1_results", "lethal_audit")
MAIN = "MAIN_ACTION"


#: 1 本の手順の深さの上限（このターンの残りの手＋相手の応手＋対話。超えたら判定不能）。
MAX_DEPTH = 300


class Unknown(Exception):
    """時間・節点の上限を超えた（判定不能）。"""


# --- カードの守りの強さ（乙の「守りの強い配り方」を作るためだけ）-----------------------------

_DEF: Dict[str, int] = {}


def defense_value(cid: str) -> int:
    """手札に在るときの守りの値（カウンター値・カウンターイベントはパワー上げの値＋α）。"""
    if not _DEF:
        path = os.path.join(E.DATA, "opcg_cards.json")
        for c in json.load(open(path, encoding="utf-8")):
            v = 0
            try:
                v = int(c.get("カウンター") or 0)
            except ValueError:
                v = 0
            text = c.get("効果(テキスト)") or ""
            if c.get("種類") == "イベント" and "【カウンター】" in text:
                ups = [int(x) for x in re.findall(r"パワー\+(\d+)", text)]
                v = max(v, max(ups) if ups else 1500) + 500   # 除去・ブロッカー付与などの汎用の守り
            _DEF[c["number"]] = max(_DEF.get(c["number"], 0), v)
    return _DEF.get(cid, 0)


#: リーダーへの打点の回数を増やしうる文言（上界の枝刈りを止める）。打点が増える道は
#: 「このターンにアタックできるユニットが増える（速攻・アタックできる）」「アタックした後に立ち上がる
#: （アクティブに）」「1 回で 2 点（ダブルアタック）」「アタック以外でライフを削る（相手のライフ・ダメージ）」
#: 「もう一度」だけ。広めに取る＝取りこぼしは枝刈りが効かないだけ（判定は変わらない）。
DANGER_WORDS = ("速攻", "ダブルアタック", "アクティブに", "アクティブで", "ダメージ", "相手のライフ", "アタックできる",
                "もう一度", "レストにできない")
#: 他のカードを場に出す文言。出せる札（手札・トラッシュ・山札）に危ない札が 1 枚でもあれば危ない扱い。
SUMMON_WORDS = ("登場させる", "登場できる")
_DANGER: Dict[str, set] = {}


def danger_ids() -> set:
    if not _DANGER:
        path = os.path.join(E.DATA, "opcg_cards.json")
        _DANGER["d"], _DANGER["s"] = set(), set()
        for c in json.load(open(path, encoding="utf-8")):
            text = c.get("効果(テキスト)") or ""
            if any(w in text for w in DANGER_WORDS):
                _DANGER["d"].add(c["number"])
            if any(w in text for w in SUMMON_WORDS):
                _DANGER["s"].add(c["number"])
    return _DANGER["d"]


def summon_ids() -> set:
    danger_ids()
    return _DANGER["s"]


# --- 判定器 ---------------------------------------------------------------------------------

_DON_UUID = re.compile(r'\{"uuid":"[^"]*","owner_id"')


def state_key(game) -> str:
    """置換表の鍵（hidden＋要求。ドン!! の uuid は捨てる＝どのドン!! を付けたかは区別しない）。"""
    s = _DON_UUID.sub('{"owner_id"', game.hidden_json()) + "|" + game.pending_json()
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()


def _in_play(game) -> set:
    h = json.loads(game.hidden_json())
    out = set()
    for p in h["players"].values():
        out.add(p["leader"]["uuid"])
        out.update(c["uuid"] for c in p["field"])
        if p.get("stage"):
            out.add(p["stage"]["uuid"])
    return out


def expand(game, actor: str, pend: Dict[str, Any]) -> List[Dict[str, Any]]:
    """要求の手（合法手＋効果対話の分岐）。対話は既定解決に加えて、確認の受ける/受けない・
    選択肢の各番号・対象 1 枚ずつ（と、飛ばせるなら 0 枚）を足す（並べ替えは既定だけ）。"""
    legal = json.loads(game.legal_json(actor))
    act = pend.get("action")
    if act in (MAIN, "SELECT_COUNTER", "SELECT_BLOCKER", "MULLIGAN") or not legal:
        return legal
    base = legal[0]
    payload = dict(base.get("payload") or {})
    variants = [payload]
    if act.startswith("CONFIRM"):
        variants.append(dict(payload, accepted=False))
    elif act == "CHOICE":
        for i in range(len(pend.get("options") or [])):
            variants.append(dict(payload, index=i))
    elif act != "ARRANGE_DECK":
        uu = pend.get("selectable_uuids") or []
        # 対象を 1 枚ずつ分けるのは場のカード（KO・レスト・戻す・付与の対象）だけ。山札・手札・トラッシュから
        # 選ぶ対話（サーチ・捨てる）は既定解決だけ（近似・notes）
        if uu:
            live = _in_play(game)
            uu = [u for u in uu if u in live]
        for u in uu:
            variants.append(dict(payload, selected_uuids=[u]))
        if pend.get("can_skip"):
            variants.append(dict(payload, selected_uuids=[]))
    out, seen = [], set()
    for v in variants:
        k = json.dumps(v, sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append({"kind": "game", "action_type": base["action_type"], "payload": v})
    return out


def apply(game, actor: str, mv: Dict[str, Any]) -> None:
    if mv.get("kind") == "battle":
        game.apply_battle_action(actor, mv["action_type"], mv.get("card_uuid"))
    else:
        game.apply_game_action(actor, mv["action_type"], json.dumps(mv.get("payload") or {}, ensure_ascii=False))


def _target(mv: Dict[str, Any]) -> Optional[str]:
    """アタックの対象の uuid（payload は `{"uuid": 攻撃者, "target_ids": [対象]}`）。"""
    t = (mv.get("payload") or {}).get("target_ids") or [None]
    return t[0]


def _my_order(mv: Dict[str, Any], opp_leader: str) -> int:
    at = mv.get("action_type")
    if at == "ATTACK":
        return 0 if _target(mv) == opp_leader else 4
    return {"ATTACH_DON": 1, "PLAY": 2, "ACTIVATE_MAIN": 3, "TURN_END": 9}.get(at, 5)


def _opp_order(mv: Dict[str, Any]) -> int:
    at = mv.get("action_type")
    return {"SELECT_COUNTER": 0, "SELECT_BLOCKER": 0, "PASS": 2}.get(at, 1)


class Solver:
    """このターンに勝ち切れるか（AND-OR・置換表・上限つき）。"""

    def __init__(self, me: str, turn: int, seconds: float, max_nodes: int,
                 memo: Optional[Dict[str, Any]] = None, char_attacks: bool = False,
                 full_don: bool = False, use_bound: bool = True):
        self.me, self.turn = me, turn
        self.char_attacks = char_attacks
        self.full_don = full_don
        self.use_bound = use_bound
        self.bound_cuts = 0
        self.too_deep = False
        self.crash = None
        self.deadline = time.perf_counter() + seconds
        self.max_nodes = max_nodes
        self.nodes = 0
        self.errors = 0
        # 置換表は同じ (席, ターン) の対象点で使い回す（後の点は前の点の部分木）。判定不能で
        # 打ち切った節点は書かない＝書いた値は全部確定値。
        memo = memo if memo is not None else {}
        self.tt: Dict[str, bool] = memo.setdefault("tt", {})
        self.best: Dict[str, Dict[str, Any]] = memo.setdefault("best", {})
        self.opp_leader = ""

    def win(self, game) -> bool:
        h = json.loads(game.hidden_json())
        opp = "p2" if self.me == "p1" else "p1"
        self.opp_leader = h["players"][opp]["leader"]["uuid"]
        return self._solve(game, None)

    def _bound_fails(self, game, moves: List[Dict[str, Any]]) -> bool:
        """攻撃の回数の上界で詰みが無いと言えるか（自分の札に回数を増やしうる文言が 1 枚も無いときだけ）。

        必要な打点＝相手のライフ＋1。上界＝今リーダーへアタックできる自分のユニットの数＋
        手札の【速攻】持ち（文言の検査で弾くので実際は 0）。"""
        h = json.loads(game.hidden_json())
        opp = "p2" if self.me == "p1" else "p1"
        p = h["players"][self.me]
        cids = [c["card_id"] for z in ("hand", "field", "trash") for c in p[z]] + [p["leader"]["card_id"]]
        if p.get("stage"):
            cids.append(p["stage"]["card_id"])
        dz = danger_ids()
        if any(c in dz for c in cids):
            return False
        if any(c in summon_ids() for c in cids) and any(c["card_id"] in dz for c in p["deck"]):
            return False
        need = len(h["players"][opp]["life"]) + 1
        att = {(m.get("payload") or {}).get("uuid") for m in moves
               if m.get("action_type") == "ATTACK" and _target(m) == self.opp_leader}
        return len(att) < need

    def _solve(self, game, commit: Optional[str], depth: int = 0) -> bool:
        w = game.winner()
        if w is not None:
            return w == self.me
        if int(game.turn_count) != self.turn or game.turn_player() != self.me:
            return False
        pend = json.loads(game.pending_json())
        if not pend:
            return False
        actor = pend["player_id"]
        mine = actor == self.me
        # 置換表は自分の判断点だけ（相手の応手の節点は浅い＝鍵の計算〔hidden の文字列化〕の方が高い）
        key = (state_key(game) + (commit or "")) if mine else None
        if key is not None and key in self.tt:
            return self.tt[key]
        self.nodes += 1
        if depth > MAX_DEPTH:
            # 同じ対話が回り続ける等の深すぎる手順（2026-10-21 の本番で RecursionError が出た）＝判定不能
            self.too_deep = True
            raise Unknown()
        if self.nodes > self.max_nodes or time.perf_counter() > self.deadline:
            raise Unknown()
        moves = expand(game, actor, pend)
        if mine:
            if not self.char_attacks:
                # キャラへのアタックは相手のライフを減らさない＝詰みの手順から外す（近似・notes）
                moves = [m for m in moves if not (m.get("action_type") == "ATTACK"
                                                  and _target(m) != self.opp_leader)]
            if pend.get("action") == MAIN and not self.full_don:
                # ドン!! の付与は「付けたユニットでそのままアタックする（か起動メインを使う）」形だけ
                # （付与の並べ替えを潰す近似・notes）。レストのユニットへの付与は外す。
                att = {(m.get("payload") or {}).get("uuid") for m in moves if m.get("action_type") == "ATTACK"}
                if commit is not None:
                    moves = [m for m in moves if (m.get("payload") or {}).get("uuid") == commit
                             and m.get("action_type") in ("ATTACH_DON", "ATTACK", "ACTIVATE_MAIN")]
                else:
                    moves = [m for m in moves if m.get("action_type") != "ATTACH_DON"
                             or (m.get("payload") or {}).get("uuid") in att]
                if commit is None and self.use_bound and self._bound_fails(game, moves):
                    self.bound_cuts += 1
                    self.tt[key] = False
                    return False
            moves.sort(key=lambda m: _my_order(m, self.opp_leader))
        else:
            moves.sort(key=_opp_order)
        res = not mine           # 自分: どれか 1 つ（既定 False）／相手: 全部（既定 True）
        tried = 0
        for mv in moves:
            child = game.fork()
            try:
                apply(child, actor, mv)
            except Exception:    # noqa: BLE001  分岐した対話の不正な選択などは手として数えない
                self.errors += 1
                continue
            tried += 1
            nxt = None
            if mine and mv.get("action_type") == "ATTACH_DON" and not self.full_don:
                nxt = (mv.get("payload") or {}).get("uuid")
            r = self._solve(child, nxt, depth + 1)
            if mine and r:
                res = True
                self.best[key] = mv
                break
            if not mine and not r:
                res = False
                break
        if not mine and tried == 0:
            res = False
        if key is not None:
            self.tt[key] = res
        return res

    def line(self, game, limit: int = 40) -> List[str]:
        """勝ち筋の 1 本（自分は記録した手・相手は並びの先頭の応手）を文字にする。"""
        out, g, commit = [], game.fork(), None
        for _ in range(limit):
            if g.winner() is not None or int(g.turn_count) != self.turn:
                break
            pend = json.loads(g.pending_json())
            if not pend:
                break
            actor = pend["player_id"]
            if actor == self.me:
                mv = self.best.get(state_key(g) + (commit or ""))
                if mv is None:
                    break
                commit = ((mv.get("payload") or {}).get("uuid")
                          if mv.get("action_type") == "ATTACH_DON" and not self.full_don else None)
            else:
                moves = sorted(expand(g, actor, pend), key=_opp_order)
                mv = None
                for m in moves:
                    c = g.fork()
                    try:
                        apply(c, actor, m)
                    except Exception:  # noqa: BLE001
                        continue
                    mv = m
                    break
                if mv is None:
                    break
            out.append(f"{actor}:{describe(g, mv)}")
            apply(g, actor, mv)
        return out


def describe(game, mv: Dict[str, Any]) -> str:
    try:
        d = json.loads(game.describe_move_json(json.dumps(mv, ensure_ascii=False)))
        return json.dumps(d, ensure_ascii=False)[:160]
    except Exception:  # noqa: BLE001
        return json.dumps(mv, ensure_ascii=False)[:160]


def judge(game, me: str, seconds: float, max_nodes: int, memo: Optional[Dict[str, Any]] = None,
          char_attacks: bool = False) -> Dict[str, Any]:
    s = Solver(me, int(game.turn_count), seconds, max_nodes, memo, char_attacks)
    t0 = time.perf_counter()
    try:
        r: Optional[bool] = s.win(game.fork())
    except Unknown:
        r = None
    except Exception as exc:  # noqa: BLE001  判定器の例外で計測を止めない（判定不能として数える）
        r = None
        s.crash = f"{type(exc).__name__}: {exc}"[:200]
    out = {"res": r, "nodes": s.nodes, "sec": round(time.perf_counter() - t0, 3), "errors": s.errors}
    if s.too_deep:
        out["too_deep"] = True
    if s.crash:
        out["crash"] = s.crash
    if r:
        out["line"] = s.line(game)
    return out


# --- 乙: 相手の見えない札の配り直し --------------------------------------------------------

def redeal(hidden: Dict[str, Any], opp: str, mode: str, rng: random.Random) -> Optional[Dict[str, Any]]:
    """相手の手札＋山札＋裏向きのライフを 1 つの山にして、枚数を保って配り直した hidden。

    mode: "def"（守りの値の高い順に手札へ）／"def_ev"（カウンターイベントを先に）／"rand"（無作為）。
    表向きのライフはそのまま。"""
    h = json.loads(json.dumps(hidden))
    p = h["players"][opp]
    life = p["life"]
    down_idx = [i for i, c in enumerate(life) if not c.get("is_face_up")]
    pool = list(p["hand"]) + list(p["deck"]) + [life[i] for i in down_idx]
    nh, nd = len(p["hand"]), len(p["deck"])
    rng.shuffle(pool)
    if mode == "def":
        pool.sort(key=lambda c: -defense_value(c["card_id"]))
    elif mode == "def_ev":
        pool.sort(key=lambda c: (-(defense_value(c["card_id"]) % 1000 == 500), -defense_value(c["card_id"])))
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


def judge_otsu(game, me: str, seconds: float, max_nodes: int, n_rand: int, seed: int,
               char_attacks: bool = False) -> Dict[str, Any]:
    """乙: 配り直した全部で甲が成り立つか（1 つでも負け→False・判定不能が在れば None）。"""
    opp = "p2" if me == "p1" else "p1"
    hidden = json.loads(game.hidden_json())
    rng = random.Random(seed)
    modes = ["def", "def_ev"] + ["rand"] * n_rand
    res: Optional[bool] = True
    per = []
    for m in modes:
        h = redeal(hidden, opp, m, rng)
        try:
            g = E.engine().Game.from_hidden(json.dumps(h, ensure_ascii=False), "p1", "p2", None)
        except Exception as exc:  # noqa: BLE001
            per.append({"mode": m, "res": None, "crash": str(exc)[:200]})
            res = None
            continue
        j = judge(g, me, seconds, max_nodes, None, char_attacks)
        j.pop("line", None)
        j["mode"] = m
        per.append(j)
        if j["res"] is False:
            res = False
            break
        if j["res"] is None:
            res = None
    return {"res": res, "per": per}


# --- 盤面の要約（例の記録用）-----------------------------------------------------------------

def summarize_board(game, me: str) -> Dict[str, Any]:
    h = json.loads(game.hidden_json())
    out = {}
    for seat in ("p1", "p2"):
        p = h["players"][seat]
        don = p.get("don") or {}

        def cs(c):
            return f"{c['card_id']}{'(R)' if c.get('is_rest') else ''}{'+D' + str(c['attached_don']) if c.get('attached_don') else ''}"
        out["me" if seat == me else "opp"] = {
            "leader": cs(p["leader"]), "field": [cs(c) for c in p["field"]],
            "hand": [c["card_id"] for c in p["hand"]], "life": len(p["life"]), "deck": len(p["deck"]),
            "don_active": len(don.get("active") or []), "don_rested": len(don.get("rested") or []),
        }
    return out


# --- 対局と記録 -----------------------------------------------------------------------------

def play_one(args) -> List[Dict[str, Any]]:
    seed, a = args
    E.engine()
    db = D.load_db()
    kw = {"budget": a.budget}
    if a.worlds > 1:
        kw["worlds"] = a.worlds
    spec = E.SeatSpec(a.net, sims=a.sims, **kw)
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "synth_roles")
    points: List[Dict[str, Any]] = []
    memos: Dict[tuple, Dict[str, Any]] = {}

    def obs(game, name, turn, step, out, move):
        pend = json.loads(game.pending_json())
        if pend.get("action") != MAIN or game.turn_player() != name or game.has_interaction:
            return
        if turn < a.min_turn:
            return
        h = json.loads(game.hidden_json())
        opp = "p2" if name == "p1" else "p1"
        if len(h["players"][opp]["life"]) > a.max_life:
            return
        rec = {"seed": seed, "seat": name, "turn": turn, "step": step,
               "opp_life": len(h["players"][opp]["life"]),
               "opp_hand": len(h["players"][opp]["hand"]),
               "move": describe(game, move)}
        for k in [k for k in memos if k != (name, turn)]:
            del memos[k]                      # 前の手番の置換表は捨てる（メモリ）
        memo = memos.setdefault((name, turn), {})
        ko = judge(game, name, a.sec, a.max_nodes, memo, a.char_attacks)
        rec["ko"] = ko
        if ko["res"]:
            rec["otsu"] = judge_otsu(game, name, a.sec_otsu, a.max_nodes, a.rand_hands, seed * 1000 + step,
                                     a.char_attacks)
        rec["board"] = summarize_board(game, name)
        points.append(rec)

    t0 = time.perf_counter()
    try:
        r = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, observer=obs)
        winner, turns, void = r["winner"], r["turns"], r["winner"] is None
    except DR.GameAborted as exc:
        winner, turns, void = None, None, str(exc)
    for p in points:
        p["won_this_turn"] = (winner == p["seat"] and turns == p["turn"])
    return [{"game": {"seed": seed, "winner": winner, "turns": turns, "void": void,
                      "sec": round(time.perf_counter() - t0, 1), "points": len(points)}}] + points


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"points_{a.seed_base}_{a.games}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            if "game" in d:
                done.add(d["game"]["seed"])
    seeds = [a.seed_base + i for i in range(a.games) if a.seed_base + i not in done]
    print(f"todo {len(seeds)} (done {len(done)}) -> {path}", flush=True)
    import multiprocessing as mp
    with open(path, "a", encoding="utf-8") as f, mp.get_context("fork").Pool(a.workers) as pool:
        for i, recs in enumerate(pool.imap_unordered(play_one, [(s, a) for s in seeds])):
            # 局の行を最後に書く（途中で止めても、局の行がある局は点が揃っている）
            for r in recs[1:] + recs[:1]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            g = recs[0]["game"]
            print(f"[{i + 1}/{len(seeds)}] seed={g['seed']} pts={g['points']} sec={g['sec']} winner={g['winner']}",
                  flush=True)


def _rate(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None}


def cmd_summarize(a):
    path = a.points or os.path.join(a.out, f"points_{a.seed_base}_{a.games}.jsonl")
    games, pts = {}, []
    pending: List[Dict[str, Any]] = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if "game" in d:
            games[d["game"]["seed"]] = d["game"]
            pts.extend(p for p in pending if p["seed"] == d["game"]["seed"])
            pending = [p for p in pending if p["seed"] != d["game"]["seed"]]
        else:
            pending.append(d)
    n = len(pts)
    ko_t = [p for p in pts if p["ko"]["res"] is True]
    ko_u = [p for p in pts if p["ko"]["res"] is None]
    ot_t = [p for p in ko_t if (p.get("otsu") or {}).get("res") is True]
    ot_u = [p for p in ko_t if (p.get("otsu") or {}).get("res") is None]
    miss_ko = [p for p in ko_t if not p["won_this_turn"]]
    miss_ot = [p for p in ot_t if not p["won_this_turn"]]
    # 手番単位（同じ局・席・ターンの最初の対象点）
    turns: Dict[tuple, List[Dict[str, Any]]] = {}
    for p in pts:
        turns.setdefault((p["seed"], p["seat"], p["turn"]), []).append(p)
    first = [sorted(v, key=lambda x: x["step"])[0] for v in turns.values()]
    anyko = [v for v in turns.values() if any(x["ko"]["res"] is True for x in v)]
    anyot = [v for v in turns.values() if any((x.get("otsu") or {}).get("res") is True for x in v)]
    g_miss_ko = {p["seed"] for p in miss_ko}
    g_miss_ot = {p["seed"] for p in miss_ot}
    g_pts = {p["seed"] for p in pts}
    # 見逃した局が結局どうなったか（その席が負けたか）
    lost_after_ot = sum(1 for s, seat in {(p["seed"], p["seat"]) for p in miss_ot}
                        if games[s]["winner"] not in (None, seat))
    by_life = {}
    for L in (1, 2, 3):
        q = [p for p in pts if p["opp_life"] == L]
        by_life[str(L)] = {"points": len(q),
                           "ko": _rate(sum(1 for p in q if p["ko"]["res"] is True), len(q)),
                           "miss_ko": _rate(sum(1 for p in q if p["ko"]["res"] is True and not p["won_this_turn"]), len(q)),
                           "miss_otsu": _rate(sum(1 for p in q if (p.get("otsu") or {}).get("res") is True
                                                  and not p["won_this_turn"]), len(q))}
    secs = sorted(p["ko"]["sec"] for p in pts)
    summ = {
        "games": len(games), "void_games": sum(1 for g in games.values() if g["void"]),
        "games_with_points": len(g_pts), "points": n, "turns": len(turns),
        "ko": {"exists": _rate(len(ko_t), n), "unknown": _rate(len(ko_u), n)},
        "otsu": {"exists": _rate(len(ot_t), n), "unknown_among_ko": _rate(len(ot_u), len(ko_t))},
        "miss_per_point": {"ko": _rate(len(miss_ko), n), "otsu": _rate(len(miss_ot), n)},
        "miss_given_exists": {"ko": _rate(len(miss_ko), len(ko_t)), "otsu": _rate(len(miss_ot), len(ot_t))},
        "miss_per_turn": {
            "ko": _rate(sum(1 for v in anyko if not v[0]["won_this_turn"]), len(turns)),
            "otsu": _rate(sum(1 for v in anyot if not v[0]["won_this_turn"]), len(turns)),
            "ko_at_first_point": _rate(sum(1 for p in first if p["ko"]["res"] is True and not p["won_this_turn"]),
                                       len(first)),
            "otsu_at_first_point": _rate(sum(1 for p in first if (p.get("otsu") or {}).get("res") is True
                                             and not p["won_this_turn"]), len(first)),
            "turns_with_ko": len(anyko), "turns_with_otsu": len(anyot)},
        "miss_per_game": {"ko": _rate(len(g_miss_ko), len(games)), "otsu": _rate(len(g_miss_ot), len(games)),
                          "otsu_missed_then_lost": lost_after_ot},
        "by_opp_life": by_life,
        "solver_sec": {"mean": round(sum(secs) / n, 3) if n else None, "p50": secs[n // 2] if n else None,
                       "p99": secs[int(n * 0.99)] if n else None, "max": secs[-1] if n else None},
    }
    ex_src = sorted(miss_ot, key=lambda p: (p["seed"], p["step"])) + \
        sorted([p for p in miss_ko if p not in miss_ot], key=lambda p: (p["seed"], p["step"]))
    examples = []
    for p in ex_src[:10]:
        examples.append({"seed": p["seed"], "seat": p["seat"], "turn": p["turn"], "step": p["step"],
                         "kind": "otsu" if p in miss_ot else "ko_only", "opp_life": p["opp_life"],
                         "cpu_move": p["move"], "board": p["board"], "winning_line": p["ko"].get("line"),
                         "game_winner": games[p["seed"]]["winner"]})
    summ["examples"] = examples
    out = os.path.join(a.out, "summary.json")
    json.dump(summ, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "examples"}, ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("--seed-base", type=int, default=3011000)
        p.add_argument("--games", type=int, default=1000)
        p.add_argument("--out", default=OUT_DIR)
        if name == "run":
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--sims", type=int, default=E.PLAY_SIMS)
            p.add_argument("--budget", type=int, default=E.PLAY_BUDGET)
            p.add_argument("--worlds", type=int, default=1)
            p.add_argument("--net", default=None)
            p.add_argument("--min-turn", type=int, default=4)
            p.add_argument("--max-life", type=int, default=3)
            p.add_argument("--sec", type=float, default=5.0, help="甲の 1 点あたりの時間の上限")
            p.add_argument("--sec-otsu", type=float, default=3.0, help="乙の配り方 1 つあたりの時間の上限")
            p.add_argument("--max-nodes", type=int, default=200000)
            p.add_argument("--rand-hands", type=int, default=4)
            p.add_argument("--char-attacks", action="store_true", help="キャラへのアタックも手順に含める（既定は外す）")
        else:
            p.add_argument("--points", default=None)
    a = ap.parse_args(argv)
    (cmd_run if a.cmd == "run" else cmd_summarize)(a)


if __name__ == "__main__":
    main()
