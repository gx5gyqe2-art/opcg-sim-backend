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


**段 7（2026-10-07）**: 1 局ぶんの `G(t)`／`R(t)`（`_price_row`＝`score_candidate` を帳簿の規約で読み直したもの・
`attack_response.parts`／`parts_mirror` の損害・T113 のブラケット）は Rust の局の駆動
（`rust/opcg_engine/src/theory/core/drv_t18.rs` の `two_curves`）。`--verify` の相手（`crossing_bridge.collect`）も Rust で、
2 つは Rust の中で別々の経路（`drv_t18` の `ar_parts` の和 対 `drv_cb` の `F`）を通る。
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
import theory_rs as TR  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra  # noqa: E402
from theory_rs import MU, THETA  # noqa: E402


def two_curves_for_game(game, theta=THETA, mu=MU):
    """1 局ぶんの `G(t)`／`R(t)`（局×席の累積系列・Rust の局の駆動 `two_curves`）。戻り値:
    `{w: {"turns": [...], "g": [...], "r": [...], "g_fam": [...], "r_life": [...], "r_hand": [...], "r_body": [...]}}`
    ＋ `winner`（0/1/None）。`turns` はその席の自席ターン番号の並び（昇順）。**T143**: `r_life`／`r_hand`／`r_body` は
    `R(t)` を 3 項（相手のライフ・手札・体）に分けた累積——**足すと `r` に戻る**。"""
    res = TR.game_call("two_curves", game, {"cfg": TR.cfg(theta, mu), "in": {}, "stats": {}, "carry": {}})
    curves = res["curves"]
    return {0: curves[0], 1: curves[1]}, res["winner"]


def collect(dirs, limit_games=0, theta=THETA, mu=MU, dump=None):
    games = 0
    winner_final_r = []          # 勝った席の R(最終自席ターン)（配線の検算に使う）
    end_g, end_r = [], []        # 各局・各席の終点（G, R）
    n_games = n_seats = 0
    for game in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        seed_g = TR.seed_of(game)
        curves, winner = two_curves_for_game(game, theta, mu)
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
