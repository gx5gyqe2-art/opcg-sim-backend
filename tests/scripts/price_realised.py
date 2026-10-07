"""**手の型ごとに「価格」と「実現した価値」を並べる**（T41・2026-09-15・読み取り専用・記録だけ）。

`docs/cpu_theory_gap.md` §0.1 の「目的の一致」——完成＝`v(手)` が手の良し悪しを説明する＝
恒等式 `z − 0.5 ≈ ΔG`（傾き 1）。T40 で `ΔG` が実デッキで反転し（型別では攻撃 0.87・効果 0.16）、
**攻撃と効果・登場の価格の比が現実と合っていない**と判った。本器はその比を**型ごとに直接測る**。

## 何を並べるか

```
価格   v(打った手)                              … 理論の値付け（`theory_order.score_candidate`）
実現   S_meas(次の自分の行) − S_meas(この行)      … 実測の価格で評価した盤面の変化
S_meas = λ·(自ライフ − 相手ライフ) + μ·(自手札 − 相手手札) + δ·(自総ドン − 相手総ドン)
         + Σ ν_meas(自分の体) − Σ ν_meas(相手の体)     ν_meas は帯ごとの実測値（式ではない）
```

**ドンは「総在庫」で数える**（`δ·(自分の総ドン − 相手の総ドン)`・総ドン＝アクティブ＋レスト＋
リーダー付与＋キャラ付与）。初版はアクティブだけを数えて付与の実現が −δ になった（2026-09-15 訂正）——
付与・登場で減ったアクティブは**次のターンに全部戻る**ので在庫の損ではなく、総在庫なら動かない。
**`RAMP_DON`（ドンデッキから追加）だけが総在庫を増やす**＝そこは +δ で実現する。理論の価格が引く
`費用·δ` は**機会費用**（同じターンに他に使えなかった分）で在庫の差分には現れない＝**登場・イベントの
「価格」は機会費用ぶん実現より低く出る**のが正しい読み（`gross`＝足し戻した総額を併記する）。

**同じターンの中の連続する自分の判断点（main 行・kind 0）**で挟む——その間に起きるのは**その手の解決だけ**
（攻撃なら相手の守りの窓・トリガー・自分のアタック時効果の選択）。**自分の選択の行（kind 1/2）は判断点ではない**ので
挟む相手にしない（初版はそこで挟んでいて攻撃の実現が半分消えていた・2026-09-16 訂正）。ターン最後の行（次の自分の行が次のターン）は**相手のターンが
丸ごと挟まる**ので外す。**守りの窓は比べない**（2026-09-15 訂正）——窓は複数行（ブロッカー→カウンター）なので「次の自分の行」が
同じ攻撃の途中になり、しかも**守りの結果は攻撃側の実現（相手が奪ったライフ・使わせた札）に既に入っている**
（攻撃の価格は守り手の最適応答を織り込んだ `min`）＝守りの質は「攻撃側の価格 − 実現」の側に現れる。

**回帰しない**——価格の側と実現の側を別々に出して**比**を見るだけ。実現の側の価格は全部 A 層の実測
（`λ`・`μ`・`δ`・帯ごとの `ν`）で、式（`nu_of`）は通らない＝**価格の欠陥が実現の側に混ざらない**。

## 読み方（事前登録）

- 型ごとの `実現 / 価格` の比が **1 から離れた型が、`ΔG` の反転を作っている型**。
- **攻撃の比が 1 より大きく、効果・登場の比が 1 より小さい**なら、T40 の読み（攻撃が安すぎる／
  効果・登場が高すぎる）が支持される。
- 局ごとの `Σ実現` は `S_meas(終局) − S_meas(開始)` に（外したターン境界を除いて）畳まれる＝
  **`Σ実現 − Σ実現(相手)` が勝敗を傾き ≈ 1 で説明する**なら、実測の価格の水準そのものは正しい。
  そこが外れれば `λ`・`μ`・`ν_meas` の水準の問題で、型の比の問題ではない。

**限界**: 実現は「次の自分の行まで」の変化なので、**その手が将来に残す価値**（体の残りの仕事・
サーチの選択の利得）は `ν_meas`・手札の枚数を通してしか入らない。付与（DON）の実現は攻撃に混ざる。


**段 7（2026-10-07）**: 価格（`score_candidate`）・実現（`S_meas`・入った札の質）・1 局ぶんの行の読みは Rust の局の駆動
（`rust/opcg_engine/src/theory/core/drv_pr.rs`）。Python に残るのは記録の読み・デッキの作り直し・集計と JSON。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/price_realised.py --in ~/w41 --out ~/pr_w41.json
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
import theory_rs as TR  # noqa: E402
from theory_bridge import MOVE_FAMILIES, POL_COLS, ROW_COLS, _extra  # noqa: E402
from theory_rs import LAM, MU, THETA  # noqa: E402

#: 実測の価格（`game_theory.md` §18）——**式ではなく実測**。`δ` は `theory_gate` と同じ値
DELTA = 0.0277
#: **帯ごとの実測 `ν`**（Phase 1a・`2026-09-13_nu_measure` 系・§18）。式 `nu_of` は通さない
NU_MEAS = {"lt_leader": 0.0690, "leader_to_sat": 0.1503, "over_sat": 0.2112}
#: **後で効く効果**（T53）——次の判断点には出ず、同じターンの後の行（攻撃）に実現が出る動作の型
FLOW_ACTS = frozenset({"ACTIVE_DON", "ATTACH_DON", "GRANT_KEYWORD", "BUFF", "BP_BUFF", "REST"})


def collect(dirs, limit_games=0, theta=THETA, mu=MU, theta_mode="const"):
    """(局, 席) ごとに、手の型ごとの価格と実現を足す（1 局ぶんは Rust の局の駆動 `price_realised`）。"""
    from theory_bridge import _seat_decks   # T68
    import deck_refill as DR
    idx2cid = TR.idx2cid()
    rec_decks = DR.record_decks(dirs)
    per = {}
    stats = {"games": 0, "own_rows": 0, "scored": 0, "no_next": 0, "silent": 0,
             "search_price": "plan", "search_deck_ok": 0, "search_deck_bad": 0,
             # **T69**: 物差しの手札の項の規約・入った札の数・補正の和（Σ(gain − μ)）・入った札の gain の平均
             "hand_meas": "quality", "hand_added": 0, "hand_quality_sum": 0.0, "hand_gain_sum": 0.0}
    games = 0
    for game in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        rows, _pol, ex, _L, _ptr, idx = game
        stats["games"] += 1                                       # `_seat_decks` が `stats` に数える前
        seed = int(rows["seed"][idx[0]])
        decks = _seat_decks(rec_decks, seed, rows, ex, idx, idx2cid, stats)   # T68
        pin = {"decks": [None if decks.get(w) is None else list(decks.get(w)) for w in (0, 1)]}
        c = TR.cfg(theta, mu, theta_mode, F_PRICING_FIX=True)
        res = TR.game_call("price_realised", game, {"cfg": c, "in": pin, "stats": stats, "carry": {}})
        for key, rec in res["per"]:
            for tk in (rec.get("turns") or {}).values():
                if "acts" in tk:
                    tk["acts"] = set(tk["acts"])
            key = tuple(key)
            if key in per:
                raise RuntimeError("price_realised: per の鍵 %r が 2 局に出た" % (key,))
            per[key] = rec
        new = res["stats"]
        stats.clear()
        stats.update(new)
    return per, stats


def _slope(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if len(x) < 10 or float(x.var()) <= 0:
        return None
    xd = x - x.mean()
    return float((xd * (y - y.mean())).sum() / (xd * xd).sum())




def _auc(scores, labels):
    from theory_bridge import auc
    return auc(scores, labels)


def _boot_slope(x, y, reps, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y); vals = []
    for _ in range(int(reps)):
        k = rng.integers(0, len(x), len(x))
        v = _slope(x[k], y[k])
        if v is not None:
            vals.append(v)
    return ([round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]
            if len(vals) >= 10 else [None, None])


def summarise(per, reps=200, seed=0):
    fams = MOVE_FAMILIES
    out = {"by_family": {}}
    allrows = [r for rec in per.values() for r in rec["rows"]]
    def block(rs):
        p = np.array([r["price"] for r in rs]); q = np.array([r["real"] for r in rs])
        g = np.array([r["gross"] for r in rs])
        te = np.array([r.get("real_te", r["real"]) for r in rs])
        return {"n": len(rs), "price_mean": round(float(p.mean()), 5), "real_mean": round(float(q.mean()), 5),
                "gross_mean": round(float(g.mean()), 5),
                # **行 → ターン末の実現**（T53・後の行の実現と重なるので参考値）
                "real_turn_end_mean": round(float(te.mean()), 5),
                "price_p10_p50_p90": [round(float(np.percentile(p, k)), 4) for k in (10, 50, 90)],
                "real_p10_p50_p90": [round(float(np.percentile(q, k)), 4) for k in (10, 50, 90)],
                "ratio_real_over_price": (round(float(q.mean() / p.mean()), 3)
                                          if abs(float(p.mean())) > 1e-9 else None),
                "ratio_real_over_gross": (round(float(q.mean() / g.mean()), 3)
                                          if abs(float(g.mean())) > 1e-9 else None),
                "corr": round(float(np.corrcoef(p, q)[0, 1]), 4) if p.std() > 0 and q.std() > 0 else None,
                "slope_real_on_price": (round(_slope(p, q), 4) if _slope(p, q) is not None else None)}
    for f in fams:
        rs = [r for r in allrows if r["fam"] == f]
        if len(rs) < 10:
            continue
        out["by_family"][f] = block(rs)
    # 効果の型の内訳（最初の動作の型ごと）
    eff = [r for r in allrows if r["fam"] == "effect"]
    acts = {}
    for r in eff:
        acts.setdefault(r["act"], []).append(r)
    out["effect_by_action"] = {a: block(rs) for a, rs in sorted(acts.items(), key=lambda kv: -len(kv[1]))
                               if len(rs) >= 20}
    # **登場時効果の型の内訳**（T54）——登場の行を登場時能力の最初の動作の型で切る（`?` は能力なし／読めない）
    pacts = {}
    for r in [r for r in allrows if r["fam"] == "play"]:
        pacts.setdefault(r.get("act") or "?", []).append(r)
    out["play_by_onplay_action"] = {a: block(rs) for a, rs in sorted(pacts.items(), key=lambda kv: -len(kv[1]))
                                    if len(rs) >= 20}
    # **T53 (a) ターン単位の恒等式を「後で効く効果が在るターン」と無いターンで分ける**——
    # 流れの効果の価格が正しければ両群の Σ価格/実現 は同じになる（重ね数えをせずに検める）
    grp = {"with_flow_effect": [], "without": []}
    for rec in per.values():
        for tk in rec["turns"].values():
            if tk["first"] is None or tk["last"] is None:
                continue
            key = "with_flow_effect" if (tk.get("acts") or set()) & FLOW_ACTS else "without"
            grp[key].append((tk["price"], tk["last"] - tk["first"]))
    out["turns_by_flow_effect"] = {}
    for key, xs in grp.items():
        if len(xs) >= 20:
            p = np.array([a for a, _b in xs]); q = np.array([b for _a, b in xs])
            out["turns_by_flow_effect"][key] = {"turns": len(xs), "price_mean": round(float(p.mean()), 5),
                                                "real_mean": round(float(q.mean()), 5),
                                                "ratio_real_over_price": round(float(q.mean() / p.mean()), 3) if abs(p.mean()) > 1e-9 else None}
    # **T53 (b) 登場の内訳**——価格の部品と実現の部品の平均
    pp = [r["play_parts"] for r in allrows if r.get("play_parts")]
    if len(pp) >= 20:
        out["play_breakdown"] = {"n": len(pp), **{k: round(float(np.mean([x[k] for x in pp])), 5) for k in pp[0]}}
    # 局ごとの突き合わせ: Σ価格 と Σ実現 が勝敗をどれだけ説明するか（傾き 1 が理想）
    by = {}
    for (sd, w), rec in per.items():
        by.setdefault(sd, {})[w] = rec
    pairs = []
    for sd, seats in by.items():
        a, b = seats.get(0), seats.get(1)
        if a is None or b is None or a["z"] is None or b["z"] is None:
            continue
        na, nb = sum(a["n"].values()), sum(b["n"].values())
        if na < 1 or nb < 1:
            continue
        def turn_sum(r, key):
            return sum((tk["last"] - tk["first"]) if key == "real" else tk["price"]
                       for tk in r["turns"].values() if tk["first"] is not None and tk["last"] is not None)
        pairs.append({"z": a["z"],
                      "dPrice": sum(a["price"].values()) - sum(b["price"].values()),
                      "dReal": sum(a["real"].values()) - sum(b["real"].values()),
                      "dPriceTurn": turn_sum(a, "price") - turn_sum(b, "price"),
                      "dRealTurn": turn_sum(a, "real") - turn_sum(b, "real"),
                      "dPrice_fam": {f: a["price"][f] - b["price"][f] for f in fams},
                      "dReal_fam": {f: a["real"][f] - b["real"][f] for f in fams}})
    out["games"] = len(pairs)
    if len(pairs) >= 10:
        y = [p["z"] for p in pairs]
        for key in ("dPrice", "dReal", "dPriceTurn", "dRealTurn"):
            x = [p[key] for p in pairs]
            out[key] = {"auc": round(_auc(x, y), 4), "slope": (round(_slope(x, y), 4)
                                                             if _slope(x, y) is not None else None),
                        "slope_ci95": _boot_slope(x, y, reps, seed),
                        "mean_abs": round(float(np.mean(np.abs(x))), 4)}
        out["by_family_games"] = {}
        for f in fams:
            xp = [p["dPrice_fam"][f] for p in pairs]; xr = [p["dReal_fam"][f] for p in pairs]
            if float(np.var(xp)) <= 0 and float(np.var(xr)) <= 0:
                continue
            out["by_family_games"][f] = {
                "auc_price": round(_auc(xp, y), 4) if float(np.var(xp)) > 0 else None,
                "auc_real": round(_auc(xr, y), 4) if float(np.var(xr)) > 0 else None,
                "slope_price": (round(_slope(xp, y), 3) if _slope(xp, y) is not None else None),
                "slope_real": (round(_slope(xr, y), 3) if _slope(xr, y) is not None else None)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--theta-mode", default="const", choices=("const", "board", "max"))
    ap.add_argument("--boot-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flow-pricing", default=None, choices=TR.FLOW_PRICING_MODES,
                    help="**T54** 後で効く効果を付与の行で数える（`option`・既定）か、使った行で数える（`exercise`＝付与の行は 0）か")
    ap.add_argument("--search-value", default=None, choices=TR.SWITCH_VALUES["SEARCH_VALUE_MODE"],
                    help="**N-4** 足した札の値: `legacy`（既定・`max(ΔH, ΔG)`）／`joint`（1 枚 1 役の手札の価値の増え・残す候補）")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.flow_pricing is not None:
        TR.set_switch("FLOW_PRICING", a.flow_pricing)
    if a.search_value is not None:
        TR.set_switch("SEARCH_VALUE_MODE", a.search_value)
    t0 = time.time()
    TR.reset_cond_stats()
    per, stats = collect(a.src, a.limit_games, a.theta, MU, a.theta_mode)
    stats["cond"] = dict(TR.COND_STATS)                     # T72: 条件の判定（真／偽／判らない）の数
    stats["f_pricing_fixes"] = TR.F_PRICING_FIXES_LABEL     # 値付けの直し（`all`・定数）
    res = {"nu_mode": TR.SW["NU_MODE"], "surv_mode": TR.SW["SURV_MODE"], "flow_pricing": TR.RUN["FLOW_PRICING"],
           "search_price": "plan", "hand_meas": "quality",
           "play_now": "hand", "cost_afford": TR.SW["COST_AFFORD_MODE"], "pricing_fixes": TR.PRICING_FIXES_LABEL,
           "decision_rows": "main",
           "inflow": "on", "cond_clock": "on", "stats": stats,
           "frozen": {"lambda": LAM, "mu": MU, "delta": DELTA, "nu_meas": NU_MEAS, "theta": a.theta},
           "summary": summarise(per, a.boot_reps, a.seed), "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
