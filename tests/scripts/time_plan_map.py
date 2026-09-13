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

**「しない」と「できない」を分ける**（ユーザ指摘 2026-09-13・本計器の主目的の 1 つ）:
「終盤に守らない」「終盤に盤面を触る」は、**選ばなかった**のではなく**選べなかった**
（手札にカウンターが無い・リーダーを殴る手が合法でない）だけかもしれない。判断点の記録には
**その時に合法だった手の一覧**（候補列）とカード ID が残っているので、**推測ではなく観測**で分けられる:
  face_avail   … 相手リーダーを殴る手が候補にあった
  board_avail  … 相手キャラを殴る／除去する手が候補にあった
  guard_avail  … **守る手段が在った**（手札のカウンター値の合計 > 0 か、自分の場に起動中のブロッカー）
  counter_sum  … **手札のカウンター値の合計**（power 単位）
  guard_enough … `counter_sum` が飛んでくる攻撃の超過パワー以上（ブロッカーが在れば無条件で真）

**守りの手段を候補一覧からは読めない**（2026-09-13 に実測で判った）: `record_gen` は
「窓・コミットは訪問を配らない＝候補を持たない」ので、カウンター窓の行の `pol_len` は 0。
そこで**その行の符号化そのもの**から読む——`n_rel_feat.S_COLS` の `counter_value`（列 7・手札の枠だけ・
`min(counter/2000, 2.5)`・**カウンターイベントも max で入っている**）を手札 10 枠で合計し、
`is_blocker_active`（列 6）を自分の場 5 枠で見る。攻めの側は main 行なので候補一覧が在る＝そのまま使う。
これで **`attack_both`（face も board も選べた行だけ）／`defense_can`（守れた行だけ）／
`defense_enough`（止めるだけのカウンターが在った行だけ）** を出す。絞っても差が残れば
**選択の誤り**、消えれば**能力の限界**＝手当ての場所が変わる。

**交絡（必ず添えて読む）**: 勝率は「その方針が**選ばれた**局の勝率」で反事実ではない
（`plan_value_map.py` と同じ制約）。帯の中でも盤面は同じではないので、
「その方針にすれば勝てる」ではなく「その帯ではその方針を選んだ局が勝っている」までしか言えない。
`guard_enough` も近似（カウンターは複数の攻撃に振り分けられる・ここは**そのターン最初の窓**で測る）。

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
from race_state import _extra as _race_extra, _incoming  # noqa: E402
from opcg_sim.learned import n_rel as NL  # noqa: E402
from opcg_sim.learned.train import n_rel_train as NT  # noqa: E402
from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from opcg_sim.learned.train import time_labels as TL  # noqa: E402
from opcg_sim.learned.train.n_eff_feat import build_eff_tables  # noqa: E402

ATTACK = ("face", "board", "mixed")
DEFENSE = ("take", "guard")
MARGIN_BANDS = ("m<=-2", "m-1", "m0", "m+1", "m>=+2")
#: 守りの候補に出る action_type（`rules/legal.rs`・**窓の行は候補を持たないので実際には出てこない**）
GUARD_ATS = ("SELECT_COUNTER", "SELECT_BLOCKER")
#: 守る手段を符号化から読む（`n_rel_feat.S_COLS` の index・`encode/tokens.rs` の表）
S_BLOCKER, S_COUNTER = 6, 7
SLOT_OWN_FIELD, SLOT_HAND = slice(2, 7), slice(12, 22)
COUNTER_SCALE = 2000.0          # `counter_value` は `min(counter/2000, 2.5)`
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
    """forward に要る列＋真値を作るためのライフ 2 列＋飛んでくる攻撃を測る `race_state` の列。"""
    sc, tok = _pad(np.asarray(dd["scalars"])[:n].astype(np.float32),
                   np.asarray(dd["tokens"])[:n].astype(np.float32))
    _sc_race, tk_race = _race_extra(dd, n)
    return {"sc": sc, "tok": tok, "race_tk": tk_race,
            "ci": np.asarray(dd["card_idx"])[:n, :NL.N_TOK].astype(np.int64),
            "life0": np.asarray(dd["scalars"])[:n, 0].astype(np.float32),
            "lives": np.asarray(dd["scalars"])[:n, 0:2].astype(np.float32)}


def avail(pol, L, ptr, i, u2c, cards):
    """行 i の**候補一覧**から「何が選べたか」（ユーザ指摘 2026-09-13・「しない」と「できない」）。

    攻めは `plan_labels.move_class` と同じ分類を候補 1 本ずつに当てる（＝打った手の分類と同じ規則）。
    守りは `SELECT_COUNTER`／`SELECT_BLOCKER` の有無と、候補に出ているカウンター値の合計。
    """
    k = int(L[i])
    base = int(ptr[i])
    out = {"face_avail": False, "board_avail": False, "guard_avail": False,
           "blocker_avail": False, "counter_sum": 0}
    for j in range(base, base + k):
        sj = json.loads(pol["pol_sig"][j])
        at = sj[0]
        if at in GUARD_ATS:
            out["guard_avail"] = True
            if at == "SELECT_BLOCKER":
                out["blocker_avail"] = True
            else:
                cid = str(pol["pol_cid"][j] or "")
                inf = cards.info(cid) if cid else None
                if inf:
                    out["counter_sum"] += int(inf.get("counter") or 0)
            continue
        cl = _cls(sj, u2c, cards)
        if cl == "face":
            out["face_avail"] = True
        elif cl == "board":
            out["board_avail"] = True
    return out


def guard_means(tok_row):
    """行の符号化 → (手札のカウンター値の合計, 起動中のブロッカーが居るか)。

    窓の行は候補一覧を持たない（`record_gen`「窓・コミットは訪問を配らない」）ので、
    「守れたのか」はここから読む。`counter_value` は手札の枠だけに入り、**カウンターイベントも
    max で含む**（`encode/tokens.rs` の表）。1 枚あたり 2.5（＝5000）で飽和する近似。
    """
    hand = np.asarray(tok_row)[SLOT_HAND]
    field = np.asarray(tok_row)[SLOT_OWN_FIELD]
    csum = float(hand[:, S_COUNTER].sum()) * COUNTER_SCALE
    blocker = bool((field[:, S_BLOCKER] > 0.5).any())
    return csum, blocker


def _cls(sj, u2c, cards):
    return PL.move_class(sj, u2c, cards)


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
        u2c = PL.uuid_map(pol, L, ptr, idx)          # 候補の uuid → カード ID（availability 用）
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
            av = avail(pol, L, ptr, i, u2c, cards)
            if not own:
                # 窓の行は候補を持たないので、守る手段は符号化から読む（2026-09-13）
                csum, blocker = guard_means(ex["tok"][i])
                over = _incoming(ex["race_tk"][i])    # 飛んでくる攻撃の超過パワー（None＝攻撃なし）
                av.update(counter_sum=csum, blocker_avail=blocker,
                          guard_avail=bool(csum > 0.0 or blocker), atk_over=over,
                          guard_enough=bool(blocker or (over is not None and csum >= over)
                                            or (over is not None and over <= 0.0)))
            rec = {"turn": t, "z": zs.get(w, 0.0), "played": int(labels[n]), **av,
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


def capability(recs, classes, keys):
    """「選べたのか」の内訳（帯 × 打った方針 × 選べたか）＝**しない／できないの会計**。"""
    out = {}
    for b in CLOCK_BANDS:
        sub = [r for r in recs if clock_band(r["true_clock"]) == b]
        if not sub:
            continue
        row = {"n": len(sub)}
        for k in keys:
            row[k] = sum(1 for r in sub if r.get(k)) / len(sub)
        for c in classes:
            kk = PL.PLAN_CLASSES.index(c)
            s_ = [r for r in sub if r["played"] == kk]
            if not s_:
                continue
            row[f"played_{c}"] = {"n": len(s_),
                                  **{k: sum(1 for r in s_ if r.get(k)) / len(s_) for k in keys}}
        out[b] = row
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
    for name, classes in (("attack", ATTACK), ("attack_both", ATTACK), ("defense", DEFENSE),
                          ("defense_can", DEFENSE), ("defense_enough", DEFENSE)):
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
    cap = out.get("capability") or {}
    for name, d in cap.items():
        print(f"選べたか（{name}）")
        for band, row in d.items():
            keys = {k: round(v, 3) for k, v in row.items() if isinstance(v, float)}
            print(f"    {band:7s} n {row['n']:6d} {keys}")
            for k, v in row.items():
                if k.startswith("played_"):
                    print(f"      {k}: n {v['n']:6d} "
                          f"{ {kk: round(vv, 3) for kk, vv in v.items() if kk != 'n'} }")


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
    att_both = [r for r in att if r.get("face_avail") and r.get("board_avail")]
    dfn_can = [r for r in dfn if r.get("guard_avail")]
    dfn_enough = [r for r in dfn if r.get("guard_enough")]
    out = {"net": os.path.basename(args.net), "games": games, "src": list(args.src),
           "rows_without_truth": no_true,
           "attack": block(att, ATTACK), "defense": block(dfn, DEFENSE),
           # 「しない」と「できない」を分けたもの（ユーザ指摘 2026-09-13）
           "attack_both": block(att_both, ATTACK),
           "defense_can": block(dfn_can, DEFENSE),
           "defense_enough": block(dfn_enough, DEFENSE),
           "capability": {
               "attack": capability(att, ATTACK, ("face_avail", "board_avail")),
               "defense": capability(dfn, DEFENSE,
                                     ("guard_avail", "blocker_avail", "guard_enough"))},
           "seconds": round(time.time() - t0, 1)}
    _print(out)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print(f"→ {args.out}（{out['seconds']}s・{games} 局）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
