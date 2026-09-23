#!/usr/bin/env python3
"""**T137a**（2026-09-23）: 局×席ごとの **理論の累積 `G(t)`** と **棋譜の累積 `R(t)`** を同じ通貨で並べる。

## 問い（ユーザの提案・2026-09-23）

「打った手を理論式で数字にして累積したもの」と「実際に起きた損害を累積したもの」を並べ、
**2 本の曲線が重なるか**で理論の完成を測る。これまでの 2 つの橋は**この形を作っていなかった**——
線形の橋は累積を**最終的な勝敗**にしか結びつけず（途中の形を見ない）、交点の橋は**しきい値への到達**を
見るだけで手ごとの累積そのものは出さない。本 T はその**器**を作る。

## 式（新定数ゼロ・既存の価格式の再利用のみ・新しい式は書かない）

* **`G(t)`** … その席が自席ターン `t` までに選んだ全ての決定（`TURN_END` 以外）の理論値
  （`theory_order.score_candidate`）を、帳簿の規約で読み直したもの
  （`theory_bridge.ledger_value`・T58 の `exercise` 変換＋T85 の付与ゼロ化）の累積。
  **`theory_bridge.collect` の `g_row` と同じ規約**——ただし本器は `search_ctx`（デッキ依存の探す能力の
  計画価格）を**渡さない**（`score_candidate` は `ctx.get("search_ctx")` で欠落を許容し、デッキ非依存の
  既定〔`sel(k)`〕に落ちる・T68）。**探す能力を持つ `PLAY` の価格だけ、理論値がやや粗い**——
  差は T137b で測る。
* **`R(t)`** … 同じ `t` までに、その席が相手に実際に与えた損害
  （`attack_response.parts`／`parts_mirror` の `opp_life + opp_hand + opp_body`・
  `crossing_bridge.harm_of` と同じ式）の累積。**T113 と同じブラケット規約**
  （決定行 → 次の自席の決定行、無ければそのターンに残っている最後の行を鏡で読む）。

## 配線の検算（独立実装での突き合わせ）

`crossing_bridge.py` は**別に**同じ `harm_of`／T113 の規約で `F_end`（勝った席が実際に与えた
損害の総量）を計算している。本器は**そのコードを 1 行も呼ばずに**同じ量を独立に組み立てる
——**勝った席の `R(最終自席ターン)` の平均**が `crossing_bridge.collect()` の `ledger.F_end_mean`
と一致すれば、2 つの別々に書いた実装が同じ答えに収束したことになる（`--verify` で自動実行）。

使い方:

    python tests/scripts/two_curves.py --in <records_dir> [--games N] [--verify] [--dump rows.json] [--json out.json]
"""

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
import attack_response as AR  # noqa: E402
import guard_afford as GA  # noqa: E402
import theory_bridge as TB  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra, _state_of, move_family  # noqa: E402
from theory_order import (MU, SC_MY_DON, SC_MY_LEADER_POWER, SC_MY_LIFE, SC_OPP_LEADER_POWER,  # noqa: E402
                          SC_OPP_LIFE, THETA, hand_ids_of, opp_bodies_of, own_attackers_of, score_candidate,
                          slot_power, theta_of)


def harm_of(p):
    """**`crossing_bridge.harm_of` と同じ式**（独立に書く——配線の検算の意味が無くなるので import しない）。"""
    return float(p["opp_life"] + p["opp_hand"] + p["opp_body"])


def _price_row(sc, tok, ci, cards, idx2cid, sig, cid, tcid, si, ti, k, theta, mu, theta_mode="const"):
    """1 決定行の理論値（帳簿の規約＝T58 `exercise` ＋ T85 付与ゼロ化）。`TURN_END` は 0・値付けできなければ `None`。
    **`theory_bridge.collect` の `g_row` と同じパイプライン**を、`search_ctx` を渡さずに簡略化して呼ぶ（docstring 参照）。"""
    sc_a, tok_a = np.asarray(sc), np.asarray(tok)
    olp = float(sc_a[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    mlp = float(sc_a[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
    th = theta_of(tok_a, float(sc_a[SC_MY_LIFE]), float(sc_a[SC_MY_DON]), mode=theta_mode, theta=theta)
    r = max(1.0, min(5.0, float(sc_a[SC_OPP_LIFE])))
    ctx = {"theta": th, "mu": mu, "opp_leader_power": olp, "my_leader_power": mlp, "r_turns": r, "don_k": 1,
           "attackers": own_attackers_of(tok_a, olp), "don_active": float(sc_a[SC_MY_DON]),
           "st": _state_of(sc, ci, idx2cid, tok=tok, cards=cards),
           # **T150f-2**: 見送った登場の価値（`misalloc_play`）に要る手札の card_id 列
           "hand": hand_ids_of(ci, idx2cid),
           "opp_bodies": opp_bodies_of(tok_a, mlp, r, th, mu, ci_row=ci, idx2cid=idx2cid)}

    def _score():
        return score_candidate(sig, cid, tcid, ctx, cards,
                                src_power=slot_power(tok_a, si), tgt_power=slot_power(tok_a, ti), don_k=k)

    played_v = _score()
    if played_v is None:
        return None
    g_v = TB.ledger_value(_score, played_v)
    # **T85**: 付与の増分は殴る行の価格に既に入っている（`ATTACH_LEDGER_MODE=in_attack`）ので、
    # 付与の行そのものは帳簿では 0 にする（移転は 1 回だけ数える）。
    if TB.ATTACH_LEDGER_MODE == "in_attack" and move_family(sig) == "attach":
        g_v = 0.0
    return float(g_v)


def two_curves_for_game(rows, pol, ex, L, ptr, idx, cards, idx2cid, theta=THETA, mu=MU):
    """1 局ぶんの `G(t)`／`R(t)`（局×席の累積系列）。戻り値: `{(w): {"turns": [...], "g": [...], "r": [...]}}`
    ＋ `winner`（0/1/None）。`turns` はその席の自席ターン番号の並び（昇順）。

    **T143**: `r_life`／`r_hand`／`r_body` は `R(t)` を `harm_of` の 3 項（相手のライフ・手札・体）に
    分けた累積——**足すと `r` に戻る**（新しい量ではなく、同じ差分を足す前に分けて持つだけ）。"""
    order = list(idx)
    by_seat = {}
    for n, i in enumerate(order):
        if int(rows["kind"][i]) == 0:
            by_seat.setdefault(int(rows["who"][i]), []).append(n)
    nxt = {}
    for _w, ns in by_seat.items():
        for a, b in zip(ns, ns[1:]):
            nxt[a] = b
    # **T113**: そのターンに残っている最後の行（席を問わない）——閉じる相手が同席に無い決定行のため
    last_of_turn = {}
    for n, i in enumerate(order):
        last_of_turn[int(rows["turn"][i])] = n

    turn_seq = {0: [], 1: []}
    g_turn = {}            # (w, t) -> 理論値の和
    r_turn = {}             # (w, t) -> 実現の損害の和
    g_fam_turn = {}        # (w, t) -> {型: 理論値の和}（T137b・g_turn の分割・和は g_turn と一致する）
    # (w, t) -> 実現の損害の 3 部品（T143・`harm_of` の 3 項をそのまま分けて積む＝和は r_turn と一致する）
    r_part_turn = {}
    z_of = {}
    for n, i in enumerate(order):
        w, t = int(rows["who"][i]), int(rows["turn"][i])
        z = float(rows["z"][i])
        if z != 0.0:
            z_of[w] = 1.0 if z > 0 else 0.0
        if t < 1 or not PL.is_own_turn(w, t) or int(rows["kind"][i]) != 0:
            continue
        k = int(L[i]); ch = int(rows["pol_chosen"][i])
        if k < 1 or ch < 0 or ch >= k:
            continue
        sc, tok, ci = ex["sc"][i], ex["tok"][i], ex["ci"][i]
        if (w, t) not in g_turn:
            g_turn[(w, t)] = 0.0; r_turn[(w, t)] = 0.0; g_fam_turn[(w, t)] = {}
            r_part_turn[(w, t)] = {"life": 0.0, "hand": 0.0, "body": 0.0}
            turn_seq[w].append(t)
        # ---- R(t): T113 のブラケット（次の行 → 実現の損害） ----
        j = nxt.get(n)
        mirror = False
        if j is not None and int(rows["turn"][order[j]]) == t:
            i2 = order[j]
        else:
            m = last_of_turn.get(t)
            i2 = order[m] if (m is not None and m > n) else None
            mirror = i2 is not None and int(rows["who"][i2]) != w
        if i2 is not None:
            p = (AR.parts_mirror if mirror else AR.parts)(sc, tok, ex["sc"][i2], ex["tok"][i2])
            r_turn[(w, t)] += harm_of(p)
            rp = r_part_turn[(w, t)]
            rp["life"] += float(p["opp_life"]); rp["hand"] += float(p["opp_hand"]); rp["body"] += float(p["opp_body"])
        # ---- G(t): 選んだ候補の理論値（帳簿の規約） ----
        b = int(ptr[i]) + ch
        sig = json.loads(pol["pol_sig"][b])
        tl = sig[2] if len(sig) > 2 else None
        g_v = _price_row(sc, tok, ci, cards, idx2cid, sig,
                         str(pol["pol_cid"][b]) or None, (str(pol["pol_tcid"][b]) or None) if tl else None,
                         int(pol["pol_si"][b]), int(pol["pol_ti"][b]), int(pol["pol_k"][b]), theta, mu)
        if g_v is not None:
            g_turn[(w, t)] += g_v
            fam = move_family(sig)
            g_fam_turn[(w, t)][fam] = g_fam_turn[(w, t)].get(fam, 0.0) + g_v
    winner = None
    for w, zz in z_of.items():
        if zz > 0.5:
            winner = w
    out = {}
    for w in (0, 1):
        ts = turn_seq[w]
        g_cum, r_cum, g_fam = [], [], []
        r_life, r_hand, r_body = [], [], []
        gs = rs = 0.0
        ls = hs = bs = 0.0
        for t in ts:
            gs += g_turn.get((w, t), 0.0); rs += r_turn.get((w, t), 0.0)
            g_cum.append(gs); r_cum.append(rs)
            g_fam.append(g_fam_turn.get((w, t), {}))     # **その 1 ターンの**（累積ではない）型別内訳
            rp = r_part_turn.get((w, t)) or {"life": 0.0, "hand": 0.0, "body": 0.0}
            ls += rp["life"]; hs += rp["hand"]; bs += rp["body"]
            r_life.append(ls); r_hand.append(hs); r_body.append(bs)
        out[w] = {"turns": ts, "g": g_cum, "r": r_cum, "g_fam": g_fam,
                  "r_life": r_life, "r_hand": r_hand, "r_body": r_body}
    return out, winner


def collect(dirs, limit_games=0, theta=THETA, mu=MU, dump=None):
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    games = 0
    winner_final_r = []          # 勝った席の R(最終自席ターン)（配線の検算に使う）
    end_g, end_r = [], []        # 各局・各席の終点（G, R）
    n_games = n_seats = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        seed_g = int(rows["seed"][idx[0]]) if len(idx) else -1
        curves, winner = two_curves_for_game(rows, pol, ex, L, ptr, idx, cards, idx2cid, theta, mu)
        n_games += 1
        for w in (0, 1):
            c = curves[w]
            if not c["turns"]:
                continue
            n_seats += 1
            end_g.append(c["g"][-1]); end_r.append(c["r"][-1])
            if winner == w:
                winner_final_r.append(c["r"][-1])
            if dump is not None:
                dump.append({"seed": seed_g, "w": w, "winner": winner, **c})
    end_g = np.asarray(end_g); end_r = np.asarray(end_r)
    out = {"games": n_games, "seats": n_seats,
           "end_g_mean": round(float(end_g.mean()), 4) if end_g.size else None,
           "end_r_mean": round(float(end_r.mean()), 4) if end_r.size else None,
           "winner_final_r_mean": round(float(np.mean(winner_final_r)), 4) if winner_final_r else None,
           "winner_final_r_n": len(winner_final_r)}
    return out


def verify_against_crossing_bridge(dirs, limit_games=0, theta=THETA, mu=MU):
    """**配線の検算**: 本器の `winner_final_r_mean`（独立実装）を `crossing_bridge.collect()` の
    `ledger.F_end_mean`（別の独立実装）と突き合わせる。一致すれば両方の実装が正しい可能性が高い
    （どちらも間違っていて同じ間違いをする、という経路は式が違うので考えにくい）。"""
    import crossing_bridge as CB
    mine = collect(dirs, limit_games, theta, mu)
    # `CB.collect` は `(rows_out, ledger, stats, turn_harm, theta_check)` の 5-tuple を返す
    # （`CB.main` と同じ呼び方・辞書ではない）。`F_end_mean` は `CB.summarise` の `ledger` 節に在る。
    rows_out, cb_ledger, _stats, turn_harm, theta_check = CB.collect(dirs, limit_games, theta, mu)
    cb_summary = CB.summarise(rows_out, cb_ledger, turn_harm, theta_check)
    cb_f_end = (cb_summary.get("ledger") or {}).get("F_end_mean")
    diff = None
    if mine["winner_final_r_mean"] is not None and cb_f_end is not None:
        diff = round(mine["winner_final_r_mean"] - cb_f_end, 4)
        rel = round(diff / max(1e-9, abs(cb_f_end)), 4)
    else:
        rel = None
    return {"two_curves_F_end_mean": mine["winner_final_r_mean"], "two_curves_winners_n": mine["winner_final_r_n"],
            "crossing_bridge_F_end_mean": cb_f_end,
            "crossing_bridge_winners_n": (cb_summary.get("ledger") or {}).get("winners"),
            "diff": diff, "rel_diff": rel}


def build_parser():
    ap = argparse.ArgumentParser(description="2 本の曲線（理論の累積 G(t)・棋譜の累積 R(t)）を作る（T137a）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="crossing_bridge の F_end_mean と独立実装で突き合わせる")
    ap.add_argument("--dump", default="", help="局×席ごとの累積系列を JSON に書く（T137b/c/d の材料）")
    ap.add_argument("--json", default="")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    dump = [] if a.dump else None
    out = collect(a.src, a.games, dump=dump)
    if a.verify:
        out["verify"] = verify_against_crossing_bridge(a.src, a.games)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
