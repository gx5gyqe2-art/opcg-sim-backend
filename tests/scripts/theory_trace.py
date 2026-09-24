#!/usr/bin/env python3
"""**理論の注釈つき棋譜**（ユーザ要望 2026-09-24「実際に理論と CPU の棋譜の違いを私の目で確かめたい」）。

CPU（探索）が実際に打った対局を歩き、**決定点ごとに**「盤面（両席・完全情報）」「探索が選んだ手」
「同じ候補の理論の価格」「理論の最善」「逸脱の大きさ `s`（手札 1 枚＝μ を単位に）」を JSON に落とす。
打ち回しには介入しない（影の判定＝T141 と同じ・`driver.run_game(observer=…, post=…)` の観測だけ）。
ビューアー（単体の HTML ページ）がこの JSON を読んで、盤面と候補表を人の目で追えるようにする。

**読みは `shadow_forbid` と同じ**（`live_theory.price_candidates` → `SEQ_MODE` の読み替え
`seq_prices` → 最善と `s`）。判定の式は 1 行も新しく書かない——`shadow_forbid.shadow_row` が返す
`forbidden`／`best_index`／`chosen_index`／`reread_src` と同じ量を、候補ごとの内訳つきで並べるだけ。

出力（`--json`）の形:

    {"meta": {...}, "games": [{"seed", "leaders": {"p1","p2"}, "winner", "turns",
                               "decisions": [{"i","turn","step","seat","kind","phase",
                                              "board": {"p1": side, "p2": side},
                                              "move": "打った手の説明", "events": [...],
                                              # kind == "main" のとき
                                              "cands": [{"d","fam","p","n","q","k","src"}],
                                              "chosen", "best", "s_mu", "forbidden"}]}]}

`side` = `compact_side`（リーダー・ライフ枚数・ドン・手札〔名前・コスト・パワー・カウンター〕・
場〔名前・パワー・レスト・付与ドン〕・ステージ・トラッシュ枚数・山札枚数）。

使い方:

    python tests/scripts/theory_trace.py --games 6 --seed-base 1300000 --decks synth_roles \
        --sims 64 --seq attack_any --json trace.json
"""

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.loop import decks as D  # noqa: E402
from opcg_sim.loop import driver as DR  # noqa: E402
from opcg_sim.loop import engine as E  # noqa: E402
import guard_afford as GA  # noqa: E402
import live_theory as LT  # noqa: E402
import shadow_forbid as SF  # noqa: E402
from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from theory_order import MU, THETA  # noqa: E402


# ---- 盤面 -----------------------------------------------------------------------------------------

_ZONES = ("field", "hand", "life", "trash", "stage")


def uuid_table(board):
    """`board_json` の両席から `{uuid: {"name", "card_id"}}`（手の説明でカード名を引くため）。
    ドン!!（`don_active`／`don_rested`）も含める。"""
    table = {}
    for side in (board.get("players") or {}).values():
        cards = [side.get("leader"), side.get("stage")]
        zones = side.get("zones") or {}
        for z in _ZONES:
            v = zones.get(z)
            cards.extend(v if isinstance(v, list) else [v])
        cards.extend(side.get("don_active") or [])
        cards.extend(side.get("don_rested") or [])
        for c in cards:
            if isinstance(c, dict) and c.get("uuid"):
                table[c["uuid"]] = {"name": c.get("name") or c.get("card_id") or "?",
                                    "card_id": c.get("card_id")}
    return table


def short(uuid):
    """uuid の先頭 8 文字（ビューアーが盤面のカードと手の主体・対象を突き合わせる鍵）。"""
    return str(uuid)[:8] if uuid else None


def _card(c, hand=False):
    if not isinstance(c, dict):
        return None
    d = {"name": c.get("name") or c.get("card_id") or "?", "card_id": c.get("card_id"),
         "cost": c.get("cost"), "power": c.get("power"), "u": short(c.get("uuid"))}
    if hand:
        d["counter"] = c.get("counter")
    else:
        d["rest"] = bool(c.get("is_rest"))
        d["don"] = int(c.get("attached_don") or 0)
        kw = c.get("keywords") or []
        if kw:
            d["kw"] = list(kw)
    return d


def compact_side(side, deck_count):
    """1 席ぶんの盤面を、ビューアーが要る動的な状態だけに絞る（完全情報＝手札の中身も持つ）。"""
    zones = side.get("zones") or {}
    stage = side.get("stage") or zones.get("stage")
    if isinstance(stage, list):
        stage = stage[0] if stage else None
    trash = zones.get("trash") or []
    return {
        "leader": _card(side.get("leader")),
        "life": int(side.get("life_count") or len(zones.get("life") or [])),
        "deck": int(deck_count or 0),
        "hand": [_card(c, hand=True) for c in zones.get("hand") or []],
        "field": [_card(c) for c in zones.get("field") or []],
        "stage": _card(stage) if stage else None,
        "trash": len(trash),
        # トラッシュの一番上（最後に置かれた札）——盤面の絵で山の表に出すため
        "trash_top": (trash[-1].get("card_id") if trash and isinstance(trash[-1], dict) else None),
        "don_active": sum(1 for x in side.get("don_active") or [] if not x.get("attached_to")),
        "don_rested": sum(1 for x in side.get("don_rested") or [] if not x.get("attached_to")),
        "don_deck": int(side.get("don_deck_count") or 0),
    }


def compact_board(board, deck_counts):
    players = board.get("players") or {}
    return {s: compact_side(players.get(s) or {}, (deck_counts or {}).get(s, 0)) for s in ("p1", "p2")}


# ---- 手の説明 ---------------------------------------------------------------------------------------

def _nm(table, uuid):
    if not uuid:
        return "?"
    e = table.get(uuid)
    return e["name"] if e else "?" + str(uuid)[:4]


def actors(move):
    """手の主体と対象の短い uuid `(su, tu)`（盤面の絵で枠を付けるため・無ければ `None`）。"""
    if not isinstance(move, dict):
        return None, None
    p = move.get("payload") or {}
    su = p.get("uuid") or move.get("card_uuid")
    tids = list(p.get("target_ids") or [])
    return short(su), short(tids[0]) if tids else None


def describe(move, table):
    """合法手（`legal` の要素・`swap` が受ける `move`）→ 人が読む 1 行。uuid はカード名に引く。"""
    if not isinstance(move, dict):
        return "?"
    at = move.get("action_type") or "?"
    p = move.get("payload") or {}
    u = p.get("uuid")
    tids = list(p.get("target_ids") or [])
    k = p.get("don_k")
    if at == "TURN_END":
        return "ターン終了"
    if at == "PLAY":
        return f"出す: {_nm(table, u)}"
    if at == "ATTACK":
        return f"攻撃: {_nm(table, u)} → {_nm(table, tids[0]) if tids else '?'}"
    if at == "ATTACH_DON":
        return f"付与1: {_nm(table, u)}"
    if at == "DON_BOX":
        kk = int(float(k or 0))
        if tids:
            return f"付与{kk}→攻撃: {_nm(table, u)} → {_nm(table, tids[0])}"
        return f"付与{kk}: {_nm(table, u)}"
    if at == "ACTIVATE_MAIN":
        return f"起動: {_nm(table, u)}"
    if at in ("MULLIGAN", "KEEP_HAND"):
        return "マリガン" if at == "MULLIGAN" else "手札キープ"
    if move.get("kind") == "battle":
        cu = move.get("card_uuid")
        return f"{at}: {_nm(table, cu)}" if cu else at
    if at == "RESOLVE_EFFECT_SELECTION":
        sel = p.get("selected_uuids") or p.get("selected") or []
        return "効果の選択: " + ("・".join(_nm(table, x) for x in sel) if sel else "（既定）")
    if u or tids:
        return f"{at}: {_nm(table, u)}" + (f" → {_nm(table, tids[0])}" if tids else "")
    return at


# ---- 決定点の注釈 ----------------------------------------------------------------------------------

def annotate_main(cands, priced, chosen, table, legal, groups, mu=MU):
    """main 行の候補表。`priced` は `seq_prices` を掛けた後の列（`price`／`src_index`）。
    戻り値は `(cands_out, best, s_mu, forbidden)`——`shadow_forbid.shadow_row` と同じ量
    （`s = played − max`・`forbidden = s < −TOL`）。値付けできない候補が混じっても壊れない。"""
    out = []
    scored = []
    for i, (c, p) in enumerate(zip(cands, priced)):
        mv = legal[groups[i]["rep"]] if i < len(groups) and groups[i].get("rep") is not None else None
        price = p.get("price")
        su, tu = actors(mv)
        out.append({"d": describe(mv, table), "fam": SF.move_family(c["sig"]), "su": su, "tu": tu,
                    "p": (round(float(price) / mu, 3) if price is not None else None),
                    "n": c.get("n"), "q": (round(float(c["q"]), 3) if c.get("q") is not None else None),
                    "k": c.get("k"), "src": p.get("src_index")})
        if price is not None:
            scored.append((i, float(price)))
    best, s_mu, forbidden = None, None, False
    if len(scored) >= 2 and chosen is not None:
        played = next((v for i, v in scored if i == chosen), None)
        bi, bv = max(scored, key=lambda iv: iv[1])
        best = bi
        if played is not None:
            s = played - bv
            s_mu = round(s / mu, 3)
            forbidden = bool(s < -SF.TOL)
    return out, best, s_mu, forbidden


# ---- 歩く ----------------------------------------------------------------------------------------

def collect(seeds, decks_mode, sims=64, net=None, dirichlet_eps=0.25, temp_turns=4, worlds=4,
            theta=THETA, mu=MU, max_steps=DR.DEFAULT_MAX_STEPS):
    """`seeds` を今ここに打ちながら（介入なし）決定点ごとの注釈を集める。"""
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    E.engine()
    search = {"worlds": int(worlds)} if worlds else {}
    spec = E.SeatSpec(net, sims=sims, dirichlet_eps=dirichlet_eps, temp_turns=temp_turns,
                      prune_futile=E.GEN_PRUNE_FUTILE, **search)
    db = D.load_db()
    games = []
    for seed in seeds:
        la, lb = D.leader_pair(db, seed, "random")
        p1, p2 = D.build_pair(db, la, lb, seed, decks_mode)
        decisions = []

        def observer(game, name, turn, step, out, move, _d=decisions):
            board = json.loads(game.board_json())
            table = uuid_table(board)
            info = board.get("turn_info") or {}
            rec = {"i": len(_d), "turn": int(turn), "step": int(step), "seat": name,
                   "kind": out.get("kind"), "phase": info.get("current_phase"),
                   "board": compact_board(board, json.loads(game.deck_counts_json())),
                   "move": describe(move, table), "events": []}
            rec["su"], rec["tu"] = actors(move)
            ab = board.get("active_battle") or None
            if ab:
                rec["battle"] = {"a": short(ab.get("attacker_uuid")), "t": short(ab.get("target_uuid"))}
            if out.get("kind") == "main":
                cands = LT.raw_candidates(game, name, out)
                groups = out.get("groups") or []
                legal = (out.get("stats") or {}).get("legal") or []
                if cands:
                    sc, tok, ci = LT.raw_row(game, name)
                    priced = LT.price_candidates(sc, tok, ci, cards, idx2cid, cands, theta, mu)
                    if SF.SEQ_MODE != "off":
                        priced, _r, _a = SF.seq_prices(cands, priced)
                    chosen = SF.find_chosen(cands, out, move)
                    cl, best, s_mu, forbidden = annotate_main(cands, priced, chosen, table, legal, groups, mu)
                    rec.update({"cands": cl, "chosen": chosen, "best": best, "s_mu": s_mu,
                                "forbidden": forbidden})
            _d.append(rec)

        def post(game, name, turn, step, move, events, _d=decisions):
            if name is None or not _d:
                return
            _d[-1]["events"] = [e.get("message") or e.get("type") for e in (events or [])
                                if isinstance(e, dict)][:12]

        res = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, observer=observer, post=post,
                          max_steps=max_steps)
        games.append({"seed": seed, "leaders": {"p1": p1[0], "p2": p2[0]}, "winner": res.get("winner"),
                      "turns": res.get("turns"), "decisions": decisions})
    return games


def summarise(games):
    """局ごとの行数と、main 行のうち理論が禁じた行の割合（ビューアーの見出し用）。"""
    n_main = sum(1 for g in games for d in g["decisions"] if d.get("kind") == "main")
    n_forb = sum(1 for g in games for d in g["decisions"] if d.get("forbidden"))
    return {"n_games": len(games), "n_decisions": sum(len(g["decisions"]) for g in games),
            "n_main": n_main, "n_forbidden": n_forb,
            "forbid_rate": round(n_forb / n_main, 4) if n_main else None}


def build_parser():
    ap = argparse.ArgumentParser(description="理論の注釈つき棋譜（ビューアー用 JSON）")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--decks", default="synth_roles",
                    choices=("singleton", "synth", "synth_dig", "synth_roles", "user"))
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--seq", default="attack_any", choices=SF.SEQ_MODES,
                    help="純付与の読み替え（T144／T155b・既定 attack_any＝T18 と同じ読み）")
    ap.add_argument("--json", default="")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    SF.set_seq_mode(a.seq)
    seeds = [a.seed_base + i for i in range(a.games)]
    games = collect(seeds, a.decks, sims=a.sims)
    out = {"meta": {"seq": a.seq, "decks": a.decks, "sims": a.sims, "mu": MU,
                    "seed_base": a.seed_base, "games": a.games, "summary": summarise(games)},
           "games": games}
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(json.dumps(out["meta"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
