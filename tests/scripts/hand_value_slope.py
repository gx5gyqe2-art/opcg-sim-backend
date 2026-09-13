"""手札 1 枚の価値を V は正しい向きに学んでいるか（`docs/measurement.md` §12・読み取り専用）。

**なぜ要るか**（ユーザ指摘 2026-07-30「手札の価値をどう正しく判断するか」＋`encoder.py` v6 の注釈）:
v5 までの符号化では value が**「手札が減った＝勝者の相貌」という逆向きの相関**を学んでいた
（遮蔽帰属: 手札枚数 +0.084・手札ID +0.165 が誤着を押し上げた）。守るのは札を使う行為・
受けるのは札が増える行為なので、**V が手札を負に評価しているなら `V_guard − V_take` は上振れする**
＝`2026-09-13_time_plan_map.md` と `2026-09-13_can_or_wont.md` の「守った方が勝ちやすい」は
偏りで説明できてしまう。本計器はその偏りの**符号**を測る。

測る量（`measurement.md` §12）:

```
slope_true = ∂P(win) / ∂(手札枚数)     同じ盤面帯の中での実測勝率の傾き
slope_V    = ∂V̂     / ∂(手札枚数)     同じ帯・同じ行での V の傾き
```

**符号が違えば V は手札の価値を逆に学んでいる**。大きさが違うだけなら較正の問題。

### 帯（交絡の除き方）

手札の枚数は盤面と強く相関する（負けている側は殴られて手札が増え、勝っている側は使い切る）。
そこで**帯の中だけで傾きを取る**（within 推定＝帯を固定効果として落とす）:

```
slope = Σ_b Σ_i (h_ib − h̄_b)(y_ib − ȳ_b) / Σ_b Σ_i (h_ib − h̄_b)²
```

帯 `b` は `(自ライフ, 相手ライフ, ターン帯, 自場のキャラ数)`。比較のため**帯を無視した
傾き（`pooled`）も併記する**——両者の差がそのまま交絡の大きさ。

### モデルの応答（観測ではなく「ネットに聞く」）

観測の傾きは「その手札枚数になった局」の勝率なので、反事実ではない。そこで**同じ行の入力を
1 枚ぶん動かして V を引き直す**（モデル内の因果）:

- `slope_V_count` … scalars の**手札枚数の列だけ**（index 6）を ±1 して差を取る。
  `measurement.md` §12 が問題にした「枚数の列」そのものの応答＝曖昧さがない。
- `slope_V_slot` … 手札の枠（トークン 12..22）のうち**カウンター値が最小の埋まった枠を 0 にし**、
  枚数の列も 1 減らす＝「1 枚失う」に近い操作。ただし v6 の手札集約（scalars 55..60）と
  EXTRA の `guard_per_card` は**据え置き**なので、応答は**過小評価側**（下限）。

3 つとも同じ行集合で出すので直接比べられる。

### 行の取り方

`time_plan_map.py` と同じ（自席ターンの最初の main 行／相手ターンの最初の行・holdout だけ）。
`z` はその席の勝敗（+1/−1・引き分けは 0 なので落とす）。**自席ターンと相手ターンを分けて出す**
——手札の意味が違う（展開の資源／守りの資源）。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/hand_value_slope.py --net ~/nrel_r5.npz \\
    --in ~/n32_wave/w*/n_records --holdout-mod 7 --out ~/hvs_w32.json
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

from plan_value_map import _pad  # noqa: E402
from opcg_sim.learned import n_rel as NL  # noqa: E402
from opcg_sim.learned.train import n_rel_train as NT  # noqa: E402
from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from opcg_sim.learned.train.n_eff_feat import build_eff_tables  # noqa: E402

#: scalars の列（`rust/opcg_engine/src/encode/scalars.rs` の対応表・**生の枚数**）
SC_MY_LIFE, SC_OPP_LIFE = 0, 1
SC_MY_HAND, SC_OPP_HAND = 6, 7
SC_MY_FIELD, SC_OPP_FIELD = 8, 9
SC_TURN = 10
#: トークンの手札 10 枠と `counter_value` の列（`n_rel_feat.S_COLS`）
SLOT_HAND = slice(12, 22)
S_COUNTER = 7
TURN_BANDS = ("T<=4", "T5-8", "T9+")


def turn_band(t):
    return "T<=4" if t <= 4 else ("T5-8" if t <= 8 else "T9+")


def band_key(sc_row):
    """帯＝(自ライフ, 相手ライフ, ターン帯, 自場のキャラ数)。**手札は入れない**（説明変数）。"""
    return (int(round(float(sc_row[SC_MY_LIFE]))),
            int(round(float(sc_row[SC_OPP_LIFE]))),
            turn_band(int(round(float(sc_row[SC_TURN])))),
            int(round(float(sc_row[SC_MY_FIELD]))))


def _extra(dd, n):
    """forward に要る列（現行の形へ 0 埋めしてから積む）。"""
    sc, tok = _pad(np.asarray(dd["scalars"])[:n].astype(np.float32),
                   np.asarray(dd["tokens"])[:n].astype(np.float32))
    return {"sc": sc, "tok": tok,
            "ci": np.asarray(dd["card_idx"])[:n, :NL.N_TOK].astype(np.int64)}


def drop_one_hand(tok_row):
    """手札の枠のうち**カウンター値が最小の埋まった枠**を 0 にした写し（と、落とせたか）。

    「埋まった枠」は S 列のどれかが非 0 であること（空の枠は全 0）。カウンター値で選ぶのは
    「一番安い札を失う」＝予算モデル（`life_budget.md` §3）の `min c` に対応させるため。
    """
    tok = np.array(tok_row, np.float32, copy=True)
    hand = tok[SLOT_HAND]
    occupied = [j for j in range(hand.shape[0]) if float(np.abs(hand[j]).sum()) > 0.0]
    if not occupied:
        return tok, False
    j = min(occupied, key=lambda k: float(hand[k, S_COUNTER]))
    tok[SLOT_HAND.start + j] = 0.0
    return tok, True


def within_slope(rows, y_key, h_key="hand", band="band", min_n=2):
    """帯の中だけの傾き（within 推定）と、帯を無視した傾き（pooled）。"""
    by = {}
    for r in rows:
        by.setdefault(r[band], []).append(r)
    num = den = 0.0
    used_bands = used_rows = 0
    for _b, sub in by.items():
        if len(sub) < min_n:
            continue
        h = np.array([r[h_key] for r in sub], np.float64)
        y = np.array([r[y_key] for r in sub], np.float64)
        if float(h.var()) <= 0.0:
            continue
        hd = h - h.mean(); yd = y - y.mean()
        num += float((hd * yd).sum()); den += float((hd * hd).sum())
        used_bands += 1; used_rows += len(sub)
    h = np.array([r[h_key] for r in rows], np.float64)
    y = np.array([r[y_key] for r in rows], np.float64)
    hd = h - h.mean(); yd = y - y.mean()
    pooled = float((hd * yd).sum() / (hd * hd).sum()) if float((hd * hd).sum()) > 0 else None
    # within 推定の標準誤差（帯ごとの残差から・帯の固定効果ぶん自由度を引く）
    se = None
    if den > 0 and used_rows > used_bands + 1:
        beta = num / den
        ss = 0.0
        for _b, sub in by.items():
            if len(sub) < min_n:
                continue
            hh = np.array([r[h_key] for r in sub], np.float64)
            yy = np.array([r[y_key] for r in sub], np.float64)
            if float(hh.var()) <= 0.0:
                continue
            res = (yy - yy.mean()) - beta * (hh - hh.mean())
            ss += float((res * res).sum())
        se = float(np.sqrt(ss / max(used_rows - used_bands - 1, 1) / den))
    return {"within": (num / den) if den > 0 else None, "within_se": se,
            "pooled": pooled, "bands_used": used_bands, "rows_used": used_rows,
            "n": len(rows)}


def collect(net, rt, dirs, holdout_mod=7, limit_games=0, bs=512, do_slot=True):
    """holdout の行 → (自席ターンの記録, 相手ターンの記録, 局数)。"""
    own, opp = [], []
    games = 0
    pend = []

    def flush():
        if not pend:
            return
        sc = np.stack([p[1] for p in pend])
        ci = np.stack([p[2] for p in pend])
        tok = np.stack([p[3] for p in pend])
        rel = NT.relations_or_zeros(net, ci, tok, rt)
        v0 = np.asarray(net.value(sc, ci, tok, *rel), np.float32).reshape(-1)
        # 枚数の列だけ −1（下限は 0）
        sc_m = np.array(sc, copy=True)
        sc_m[:, SC_MY_HAND] = np.maximum(sc_m[:, SC_MY_HAND] - 1.0, 0.0)
        v_m = np.asarray(net.value(sc_m, ci, tok, *rel), np.float32).reshape(-1)
        if do_slot:
            tok_s = np.stack([p[4] for p in pend])
            ok = np.array([p[5] for p in pend], bool)
            rel_s = NT.relations_or_zeros(net, ci, tok_s, rt)
            v_s = np.asarray(net.value(sc_m, ci, tok_s, *rel_s), np.float32).reshape(-1)
        for k, (rec, _s, _c, _t, _ts, slot_ok) in enumerate(pend):
            rec["v"] = float(v0[k])
            # −1 枚にしたときの差を「+1 枚あたり」に直す（符号を揃える）
            rec["dv_count"] = float(v0[k] - v_m[k])
            if do_slot and slot_ok:
                rec["dv_slot"] = float(v0[k] - v_s[k])
            (own if rec["own"] else opp).append(rec)
        pend.clear()

    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, extra_fn=_extra):
        seed = int(rows["seed"][idx[0]])
        if holdout_mod and seed % holdout_mod != 0:
            continue
        games += 1
        if limit_games and games > limit_games:
            break
        zs = {int(rows["who"][i]): float(rows["z"][i]) for i in idx}
        seen_own, seen_opp = set(), set()
        for i in idx:
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1:
                continue
            z = zs.get(w, 0.0)
            if z == 0.0:                       # 引き分け＝勝敗の傾きが定義できない
                continue
            is_own = PL.is_own_turn(w, t)
            if is_own:
                if int(rows["kind"][i]) != 0 or (w, t) in seen_own:
                    continue
                seen_own.add((w, t))
            else:
                if (w, t) in seen_opp:
                    continue
                seen_opp.add((w, t))
            sc_row = ex["sc"][i]
            rec = {"own": is_own, "z": 1.0 if z > 0 else 0.0,
                   "hand": float(sc_row[SC_MY_HAND]),
                   "opp_hand": float(sc_row[SC_OPP_HAND]),
                   "turn": t, "band": "|".join(str(x) for x in band_key(sc_row))}
            tok_s, slot_ok = drop_one_hand(ex["tok"][i]) if do_slot else (None, False)
            pend.append((rec, sc_row, ex["ci"][i], ex["tok"][i], tok_s, slot_ok))
            if len(pend) >= bs:
                flush()
    flush()
    return own, opp, games


def _half(x):
    return None if x is None else x / 2.0


def _r(x, nd=5):
    return None if x is None else round(float(x), nd)


def _sign_agrees(true_slope, *model_slopes):
    """**本計器の判定**: 実測の勝率の傾きと、ネットの 3 通りの傾きの符号が全部揃っているか。

    揃っていなければ V は手札の価値を逆に学んでいる＝`V_guard − V_take` の読みは
    その分だけ上振れしている（`docs/measurement.md` §12）。
    """
    if true_slope is None or true_slope == 0.0:
        return None
    want = 1.0 if true_slope > 0 else -1.0
    out = {}
    for name, x in zip(("slope_v_obs", "dv_count", "dv_slot"), model_slopes):
        out[name] = None if x is None or x == 0.0 else bool((1.0 if x > 0 else -1.0) == want)
    out["all"] = all(v for v in out.values() if v is not None) if any(
        v is not None for v in out.values()) else None
    return out


def summarize(recs):
    """1 つの行集合の要約（傾き 3 本＋手札枚数の分布）。"""
    if not recs:
        return None
    out = {"n": len(recs),
           "hand_mean": round(float(np.mean([r["hand"] for r in recs])), 3),
           "hand_sd": round(float(np.std([r["hand"] for r in recs])), 3),
           "winrate": round(float(np.mean([r["z"] for r in recs])), 4),
           "v_mean": round(float(np.mean([r["v"] for r in recs])), 4),
           "slope_true": within_slope(recs, "z"),
           "slope_v_obs": within_slope(recs, "v"),
           "dv_count_mean": round(float(np.mean([r["dv_count"] for r in recs])), 5)}
    slot = [r["dv_slot"] for r in recs if "dv_slot" in r]
    out["dv_slot_mean"] = round(float(np.mean(slot)), 5) if slot else None
    out["dv_slot_n"] = len(slot)
    # **単位を揃える**: V は tanh 出力で (−1, 1)＝おおよそ `2·P(win) − 1` なので、
    # 勝率の傾きと比べるときは半分にする（符号は変わらない・大きさの比較のため）。
    out["in_pwin"] = {
        "slope_true": _r(out["slope_true"]["within"]),
        "slope_v_obs": _r(_half(out["slope_v_obs"]["within"])),
        "dv_count": _r(_half(out["dv_count_mean"])),
        "dv_slot": _r(_half(out["dv_slot_mean"])),
    }
    out["sign_agrees"] = _sign_agrees(out["slope_true"]["within"], out["slope_v_obs"]["within"],
                                     out["dv_count_mean"], out["dv_slot_mean"])
    # 帯ごとの傾き（大きい帯だけ・均質性の確認用）
    by = {}
    for r in recs:
        by.setdefault(r["band"], []).append(r)
    big = sorted(by.items(), key=lambda kv: -len(kv[1]))[:8]
    out["top_bands"] = [
        {"band": b, "n": len(sub),
         "slope_true": within_slope(sub, "z")["pooled"],
         "slope_v": within_slope(sub, "v")["pooled"]}
        for b, sub in big if len(sub) >= 30]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", required=True, help="NRel の npz（value だけ使う）")
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--holdout-mod", type=int, default=7, help="seed %% mod == 0 の局だけ（0=全部）")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--no-slot", action="store_true", help="枠を落とす応答を測らない（速い）")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    t0 = time.time()
    stats, ab, abm, pwr, isl, _vocab = build_eff_tables()
    net = NL.NRelNet.load(args.net, (stats, ab, abm, pwr, isl))
    rt = None if "rel" in (net.ablate or ()) else (stats, ab, abm, pwr, isl)
    own, opp, games = collect(net, rt, args.src, args.holdout_mod, args.limit_games,
                              args.batch, do_slot=not args.no_slot)
    out = {"net": os.path.basename(args.net), "games": games,
           "holdout_mod": args.holdout_mod,
           "own_turn": summarize(own), "opp_turn": summarize(opp),
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
