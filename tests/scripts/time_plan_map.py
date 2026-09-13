"""時間 × 方針の地図——「自分の方が速いときに殴り、遅いときに盤面を触っているか」を測る
（計画 §20.11 の候補 H の続き・読み取り専用・訓練も Rust 変更もしない）。

`2026-09-13_time_head.md` で**盤面は時計を持っている**ことが分かった（決着までの自席ターン数を
誤差約 1.0 ターン・2／3 先のライフを誤差 0.8 枚で当てる）。そこで**時間の物差しで方針を割る**。
ユーザが言語化した中盤の 3 方針——(1) 盤面を展開しながらライフを殴り切る純粋な殴り合い・
(2) キャラ攻撃を混ぜて手札とライフのシーソー・(3) キャラ攻撃を増やして手札切れを待つ——の
使い分けは「どちらが速いか」の計算そのものなので、**レースの余裕（margin）と残りターン数で
方針の勝率を割れば、どの方針がどこで勝ちやすいかが出る**。

行の取り方は `plan_value_map.py` と同じ（自席ターンの最初の main 行／相手ターンの最初の自分の行・
holdout だけ）。**方針は打った方針**（`plan_labels` の分類）で、**時間は 2 通りで割る**:
  pred … ネットの時間軸ヘッド（`--net` に `--time-weight` で訓練したネットを渡す）の予測
  true … 記録から後付けした真値（`time_labels.label_game`・同じ行）
両方出すのは「ネットの読みが行動に使える精度か」を pred と true の差で見るため。

帯:
  margin … 3 つ先の自席ターン開始時の（自分のライフ − 相手のライフ）を丸めて
            `m<=-2 / m-1 / m0 / m+1 / m>=+2`（正＝自分が速い＝レースで有利）
  clock  … 決着までの自席ターン数 `t<=2 / t3-4 / t5+`

出すもの（`attack`＝自席ターン・`defense`＝相手ターン）:
  by_true_margin／by_pred_margin／by_true_clock … 帯ごとに
      share   … 打った方針の割合（face／board／mixed、守りは take／guard）
      winrate … **その帯でその方針を打った局の勝率**（どれが勝ちやすいかの実測）
      n       … 行数（帯ごと・方針ごと）
  spread … 帯をまたいだ share の振れ幅（**0 に近ければ時計で方針を変えていない**）
  time_err … この行集合での予測誤差（列ごとの平均絶対誤差・pred と true の差）

**交絡（必ず添えて読む）**: 勝率は「その方針が**選ばれた**局の勝率」で反事実ではない
（`plan_value_map.py` と同じ制約）。帯の中でも盤面は同じではないので、
「その方針にすれば勝てる」ではなく「その帯ではその方針を選んだ局が勝っている」までしか言えない。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/time_plan_map.py --net ~/nrel_r10.npz \\
    --in ~/n32_wave/w*/n_records --holdout-mod 7 --out ~/tpm_w32.json
"""
import argparse
import collections
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

from plan_value_map import _pad  # noqa: E402
from opcg_sim.learned import n_rel as NL  # noqa: E402
from opcg_sim.learned.train import n_rel_train as NT  # noqa: E402
from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from opcg_sim.learned.train import time_labels as TL  # noqa: E402
from opcg_sim.learned.train.n_eff_feat import build_eff_tables  # noqa: E402

ATTACK = ("face", "board", "mixed")
DEFENSE = ("take", "guard")
MARGIN_BANDS = ("m<=-2", "m-1", "m0", "m+1", "m>=+2")
CLOCK_BANDS = ("t<=2", "t3-4", "t5+")
#: 時間の列（`time_labels.TIME_COLS` と同じ並び）の index
C_TLEFT, C_MY2, C_OP2, C_MY3, C_OP3, C_MYE, C_OPE = range(TL.D_TIME)


def margin_band(m):
    """（自分のライフ − 相手のライフ）→ 帯。正＝自分が速い。"""
    k = int(round(m))
    if k <= -2:
        return "m<=-2"
    if k >= 2:
        return "m>=+2"
    return {-1: "m-1", 0: "m0", 1: "m+1"}[k]


def clock_band(t):
    """決着までの自席ターン数 → 帯。"""
    k = t
    return "t<=2" if k <= 2.5 else ("t3-4" if k <= 4.5 else "t5+")


def _extra(dd, n):
    """forward に要る列＋真値を作るためのライフ 2 列。"""
    sc, tok = _pad(np.asarray(dd["scalars"])[:n].astype(np.float32),
                   np.asarray(dd["tokens"])[:n].astype(np.float32))
    return {"sc": sc, "tok": tok,
            "ci": np.asarray(dd["card_idx"])[:n, :NL.N_TOK].astype(np.int64),
            "life0": np.asarray(dd["scalars"])[:n, 0].astype(np.float32),
            "lives": np.asarray(dd["scalars"])[:n, 0:2].astype(np.float32)}


def collect(net, rt, dirs, holdout_mod=7, limit_games=0, bs=512):
    """holdout の行 → 方針・勝敗・時間（予測と真値）の記録。"""
    cards = PL.Cards()
    att, dfn = [], []
    games = 0
    no_true = 0
    pend = []

    def flush():
        if not pend:
            return
        sc = np.stack([p[2] for p in pend]); ci = np.stack([p[3] for p in pend])
        tok = np.stack([p[4] for p in pend])
        _v, pu = net.value_with_time(sc, ci, tok, *NT.relations_or_zeros(net, ci, tok, rt))
        for (kind, rec, _s, _c, _t), q in zip(pend, pu):
            q = np.asarray(q, np.float32)
            rec["pred_margin"] = float(q[C_MY3] - q[C_OP3])
            rec["pred_clock"] = float(q[C_TLEFT] * TL.T_SCALE)
            rec["pred"] = [float(x) for x in q]
            (att if kind == "a" else dfn).append(rec)
        pend.clear()

    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, extra_fn=_extra):
        seed = int(rows["seed"][idx[0]])
        if holdout_mod and seed % holdout_mod != 0:
            continue
        games += 1
        if limit_games and games > limit_games:
            break
        labels, _unk = PL.label_game(rows, pol, ex["life0"], L, ptr, idx, cards)
        tm_true, tmask = TL.label_game(rows, ex["lives"], idx)
        zs = {int(rows["who"][i]): float(rows["z"][i]) for i in idx}
        seen_own, seen_opp = set(), set()
        for n, i in enumerate(idx):
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1 or int(labels[n]) < 0:
                continue
            if not tmask[n]:
                no_true += 1
                continue
            own = PL.is_own_turn(w, t)
            if own:
                if int(rows["kind"][i]) != 0 or (w, t) in seen_own:
                    continue
                seen_own.add((w, t))
            else:
                if (w, t) in seen_opp:
                    continue
                seen_opp.add((w, t))
            tt = tm_true[n]
            rec = {"turn": t, "z": zs.get(w, 0.0), "played": int(labels[n]),
                   "true_margin": float(tt[C_MY3] - tt[C_OP3]),
                   "true_clock": float(tt[C_TLEFT] * TL.T_SCALE),
                   "true": [float(x) for x in tt]}
            pend.append(("a" if own else "d", rec, ex["sc"][i], ex["ci"][i], ex["tok"][i]))
            if len(pend) >= bs:
                flush()
    flush()
    return att, dfn, games, no_true


def _cells(recs, classes, band_fn, bands):
    """帯 × 方針 → n／share／winrate。"""
    out = {}
    for b in bands:
        sub = [r for r in recs if band_fn(r) == b]
        if not sub:
            continue
        row = {"n": len(sub), "share": {}, "winrate": {}, "n_by_plan": {}}
        for c in classes:
            k = PL.PLAN_CLASSES.index(c)
            s = [r for r in sub if r["played"] == k]
            row["share"][c] = len(s) / len(sub)
            row["n_by_plan"][c] = len(s)
            row["winrate"][c] = (sum(1 for r in s if r["z"] > 0) / len(s)) if s else None
        row["winrate_all"] = sum(1 for r in sub if r["z"] > 0) / len(sub)
        best = [(v, c) for c, v in row["winrate"].items() if v is not None and row["n_by_plan"][c] >= 30]
        row["best_plan"] = max(best)[1] if best else None
        out[b] = row
    return out


def _spread(cells, classes):
    """帯をまたいだ share の振れ幅（max − min）＝時計で方針を変えているかの指標。"""
    out = {}
    for c in classes:
        xs = [v["share"][c] for v in cells.values() if v["n"] >= 100]
        out[c] = (max(xs) - min(xs)) if len(xs) >= 2 else None
    return out


def _time_err(recs):
    """この行集合での予測誤差（列ごとの平均絶対誤差）。"""
    if not recs or "pred" not in recs[0]:
        return None
    p = np.array([r["pred"] for r in recs], np.float32)
    t = np.array([r["true"] for r in recs], np.float32)
    mae = np.abs(p - t).mean(0)
    out = dict(zip(TL.TIME_COLS, [round(float(x), 4) for x in mae]))
    out["t_left_turns"] = round(float(np.abs(p[:, C_TLEFT] - t[:, C_TLEFT]).mean() * TL.T_SCALE), 3)
    out["margin"] = round(float(np.abs(
        (p[:, C_MY3] - p[:, C_OP3]) - (t[:, C_MY3] - t[:, C_OP3])).mean()), 3)
    return out


def block(recs, classes):
    out = {"rows": len(recs)}
    if not recs:
        return out
    out["by_true_margin"] = _cells(recs, classes, lambda r: margin_band(r["true_margin"]), MARGIN_BANDS)
    out["by_pred_margin"] = _cells(recs, classes, lambda r: margin_band(r["pred_margin"]), MARGIN_BANDS)
    out["by_true_clock"] = _cells(recs, classes, lambda r: clock_band(r["true_clock"]), CLOCK_BANDS)
    out["spread_true_margin"] = _spread(out["by_true_margin"], classes)
    out["spread_pred_margin"] = _spread(out["by_pred_margin"], classes)
    out["time_err"] = _time_err(recs)
    out["played"] = {c: sum(1 for r in recs if r["played"] == PL.PLAN_CLASSES.index(c))
                     for c in classes}
    return out


def _print(out):
    def cells(d, classes):
        for b, v in d.items():
            sh = " ".join(f"{c} {v['share'][c]:.2f}" for c in classes)
            wr = " ".join(f"{c} {v['winrate'][c]:.3f}" if v['winrate'][c] is not None else f"{c} -"
                          for c in classes)
            print(f"    {b:7s} n {v['n']:6d} | share {sh} | 勝率 {wr} | 最良 {v['best_plan']}")
    for name, classes in (("attack", ATTACK), ("defense", DEFENSE)):
        b = out.get(name) or {}
        print(f"{name}: 行 {b.get('rows', 0)}  打った内訳 {b.get('played')}")
        if not b.get("rows"):
            continue
        print("  真の余裕（3 ターン先のライフ差・正＝自分が速い）")
        cells(b["by_true_margin"], classes)
        print("  ネットの予測した余裕")
        cells(b["by_pred_margin"], classes)
        print("  真の残りターン数")
        cells(b["by_true_clock"], classes)
        print(f"  share の振れ幅 真 {b['spread_true_margin']} 予測 {b['spread_pred_margin']}")
        print(f"  予測誤差 {b['time_err']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True, help="時間軸ヘッド付きの npz（`--time-weight` で訓練）")
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--holdout-mod", type=int, default=7, help="seed%%N==0 だけ読む（0 で全部）")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON の書き出し先")
    args = ap.parse_args(argv)
    t0 = time.time()
    stats, ab, abm, pwr, isl, _vocab = build_eff_tables()
    net = NL.NRelNet.load(args.net, (stats, ab, abm, pwr, isl))
    if not getattr(net, "time", False):
        raise SystemExit(f"{args.net} に時間軸ヘッド（`time_*` の鍵）が無い")
    rt = None if "rel" in (net.ablate or ()) else (stats, ab, abm, pwr, isl)
    att, dfn, games, no_true = collect(net, rt, args.src, args.holdout_mod, args.limit_games)
    out = {"net": os.path.basename(args.net), "games": games, "src": list(args.src),
           "rows_without_truth": no_true,
           "attack": block(att, ATTACK), "defense": block(dfn, DEFENSE),
           "seconds": round(time.time() - t0, 1)}
    _print(out)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print(f"→ {args.out}（{out['seconds']}s・{games} 局）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
