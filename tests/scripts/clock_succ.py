#!/usr/bin/env python3
"""**候補 `OPCG_CLOCK_VALUE` の後続局面の表**（2026-10-10・`docs/reports/2026-10-10_clock_value.md` §1.2・§1.5）。

記録の局を記録と同じ設定で打ち直し（T140 `live_theory.replay_and_compare` と同じ経路）、自席ターンのメインの行
（候補 2 本以上）ごとに次を局ごとの npz に書く:

* 打つ前の盤面を**相手の側から**符号化した行（相手の手札の読み・鏡の手札に使う・§1.3 の 3）。
* 候補（`decide` の束の代表手）ごとに、真の盤面を複製（`Game.fork`）→ 探索の適用（`Game.apply_search_move`）→
  窓を探索の葉の規則で閉じる（`Game.resolve_windows`）→ 状態と、自分の側・相手の側の符号化した行。

状態: 0＝まだ自分のターンで自分がメインの判断をする番（読む）／1＝終局・自分の勝ち／2＝終局・自分の負け／
3＝`TURN_END` 以外の手でターンが渡った（値段なし）／4＝窓が閉じない・メインでない判断が残る（値段なし）／5＝適用できない（値段なし）／
6＝`TURN_END`（今の盤面からこのターンの残りの行動を手放した局面を読む・`end_of_turn`）。

打ち直した盤面は記録の行とビットで照合する（`match`＝その局の全部の行が一致・一致しない局は理論の側で使わない）。
行は記録と同じキャスト（float16／int16）で持つ（打つ前の行＝記録の行と同じ精度で前後を読む）。

使い方（局ごとに `<out>/<seed>.npz`・在る局は飛ばす＝途中から再開できる）:

    OPCG_LOG_SILENT=1 python tests/scripts/clock_succ.py --in <records_dir> --out <dir> [--games 10]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from opcg_sim.loop import decks as D  # noqa: E402
from opcg_sim.loop import driver as DR  # noqa: E402
from opcg_sim.loop import engine as E  # noqa: E402
from opcg_sim.loop import record_gen as RG  # noqa: E402
import live_theory as LT  # noqa: E402

ST_MAIN, ST_WON, ST_LOST, ST_TURN, ST_OPEN, ST_ERR, ST_END = 0, 1, 2, 3, 4, 5, 6
#: 符号化の列（`rust/opcg_engine/src/encode/{scalars,tokens}.rs`・`leaves_to.rs` の定数と同じ）
SC_MY_DON, S_CAN_ATTACK, S_PLAYABLE = 2, 5, 8
OWN_UNITS = range(0, 7)          # 自リーダー（0）と自分の場（2..7）・1 は相手のリーダー
HAND = range(12, 22)


def end_of_turn(me):
    """`TURN_END` の後続（§1.4 の訂正）: 今の盤面で、このターンに残った行動を全部手放した局面＝アクティブなドン 0・
    攻撃できる体なし・出せる札なし（規則: ターンを終えるとこのターンの行動は残らない）。相手の側の行は今のまま。"""
    sc, tok, ci = me[0].copy(), me[1].copy(), me[2].copy()
    sc[SC_MY_DON] = 0
    for s_ in OWN_UNITS:
        if s_ != 1:
            tok[s_, S_CAN_ATTACK] = 0
    for s_ in HAND:
        tok[s_, S_PLAYABLE] = 0
    return sc, tok, ci


def _meta(d):
    with open(os.path.join(d, "meta_n_record.json"), encoding="utf-8") as f:
        return json.load(f)


def record_steps(d, seeds=None):
    """記録の `{seed: {step: (sc, tok, ci)}}`（メインの行だけ・照合用）と局の順の seed の列。"""
    out, order = {}, []
    for game in PL.iter_games([d], row_cols=("who", "turn", "seed", "kind", "step", "pol_len"),
                              pol_cols=(), extra_fn=LT._extra):
        rows, _pol, ex, _L, _ptr, idx = game
        if not len(idx):
            continue
        s = int(rows["seed"][idx[0]])
        if seeds is not None and s not in seeds:
            continue
        order.append(s)
        m = {}
        for i in idx:
            if int(rows["kind"][i]) == 0:
                m[int(rows["step"][i])] = (ex["sc"][i], ex["tok"][i], ex["ci"][i])
        out[s] = m
    return out, order


def _cast(game, name):
    sc, tok, ci = LT.raw_row(game, name)
    c = LT.cast_row(sc, tok, ci)
    return c[0], c[1], c[2]


def successor(game, name, turn, mv, opts):
    """1 候補の後続（状態, 自分の側の行, 相手の側の行, 窓の手数）。"""
    opp = "p2" if name == "p1" else "p1"
    f = game.fork()
    try:
        f.apply_search_move(name, json.dumps(mv, ensure_ascii=False))
        r = json.loads(f.resolve_windows(opts))
    except Exception:                                       # noqa: BLE001  適用できない候補は値段なし
        return ST_ERR, None, None, 0
    w = f.winner()
    if w is not None:
        return (ST_WON if w == name else ST_LOST), None, None, r["plies"]
    if r["open"]:
        return ST_OPEN, None, None, r["plies"]
    pend = json.loads(f.pending_json()) or {}
    if int(f.turn_count or 0) != int(turn) or pend.get("player_id") != name:
        return ST_TURN, None, None, r["plies"]
    if pend.get("action") != "MAIN_ACTION":
        return ST_OPEN, None, None, r["plies"]
    return ST_MAIN, _cast(f, name), _cast(f, opp), r["plies"]


def replay_game(d, seed, meta, rec_rows):
    """1 局を打ち直して表を作る。戻り＝npz に書く dict。"""
    E.engine()
    search = {"worlds": int(meta["search"]["worlds"])} if (meta.get("search") or {}).get("worlds") else {}
    spec = E.SeatSpec(meta.get("net"), sims=int(meta["sims"]), dirichlet_eps=float(meta["dirichlet_eps"]),
                      temp_turns=int(meta["temp_turns"]), prune_futile=E.GEN_PRUNE_FUTILE, **search)
    db = D.load_db()
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, meta["decks"])
    rows = {"step": [], "who": [], "turn": [], "n": [], "ptr": [], "osc": [], "otok": [], "oci": []}
    cand = {"st": [], "plies": [], "msc": [], "mtok": [], "mci": [], "osc": [], "otok": [], "oci": []}
    zsc, ztok, zci = None, None, None
    chk = {"rows": 0, "bad": 0}

    def observer(game, name, turn, step, out, move):
        nonlocal zsc, ztok, zci
        w = 0 if name == "p1" else 1
        if out.get("kind") != "main":
            return
        mine = _cast(game, name)
        ref = rec_rows.get(int(step))
        chk["rows"] += 1
        if ref is None or not (np.array_equal(mine[0], ref[0]) and np.array_equal(mine[1], ref[1])
                               and np.array_equal(np.asarray(mine[2], np.int64), np.asarray(ref[2], np.int64))):
            chk["bad"] += 1
        groups = out.get("groups") or []
        if not PL.is_own_turn(w, int(turn)) or len(groups) < 2:
            return
        legal = (out.get("stats") or {}).get("legal") or []
        opp = "p2" if name == "p1" else "p1"
        o = _cast(game, opp)
        if zsc is None:
            zsc, ztok, zci = np.zeros_like(o[0]), np.zeros_like(o[1]), np.zeros_like(o[2])
        rows["step"].append(int(step)); rows["who"].append(w); rows["turn"].append(int(turn))
        rows["n"].append(len(groups)); rows["ptr"].append(len(cand["st"]))
        rows["osc"].append(o[0]); rows["otok"].append(o[1]); rows["oci"].append(o[2])
        opts = spec.decide_opts(seed, int(turn), name, None)
        for g in groups:
            mv = legal[g["rep"]]
            if mv.get("action_type") == "TURN_END":
                st, me, op, plies = ST_END, end_of_turn(mine), o, 0
            else:
                st, me, op, plies = successor(game, name, turn, mv, opts)
            cand["st"].append(st); cand["plies"].append(plies)
            me = me or (zsc, ztok, zci)
            op = op or (zsc, ztok, zci)
            cand["msc"].append(me[0]); cand["mtok"].append(me[1]); cand["mci"].append(me[2])
            cand["osc"].append(op[0]); cand["otok"].append(op[1]); cand["oci"].append(op[2])

    res = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, max_steps=RG.MAX_STEPS, observer=observer)
    out = {"seed": np.int64(seed), "match": np.int8(chk["bad"] == 0 and chk["rows"] == len(rec_rows)),
           "rows_checked": np.int64(chk["rows"]), "rows_bad": np.int64(chk["bad"]),
           "rows_recorded": np.int64(len(rec_rows)), "winner": str(res["winner"])}
    for k, v in rows.items():
        out["r_" + k] = np.asarray(v) if v else np.zeros(0)
    for k, v in cand.items():
        out["c_" + k] = np.asarray(v) if v else np.zeros(0)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", required=True, help="記録のディレクトリ（1 つ）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=0, help="先頭から何局（0＝全部）")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    meta = _meta(a.src)
    rec, order = record_steps(a.src)
    if a.games:
        order = order[:a.games]
    t0 = time.time()
    n_done = n_bad = 0
    for s in order:
        path = os.path.join(a.out, "%d.npz" % s)
        if os.path.exists(path):
            continue
        t1 = time.time()
        out = replay_game(a.src, s, meta, rec[s])
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, path)
        n_done += 1
        n_bad += int(out["match"]) == 0
        print("CLOCK_SUCC seed=%d rows=%d cands=%d match=%d %.1fs (total %.0fs)"
              % (s, len(out["r_step"]), len(out["c_st"]), int(out["match"]), time.time() - t1, time.time() - t0),
              flush=True)
    print("CLOCK_SUCC_DONE games=%d mismatched=%d %.0fs" % (n_done, n_bad, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
