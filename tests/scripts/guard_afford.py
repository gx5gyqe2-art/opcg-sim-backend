"""守りは「持っていた」のか「払えた」のか——カウンターの **DON コスト**を入れ直す
（`docs/life_budget.md` §5 の宿題・読み取り専用・ネットを使わない）。

**なぜ要るか**: `2026-09-13_can_or_wont.md` は「受けた行の **92.6%** は守る手段を持っていた・
**71.0%** は止めるだけの量を持っていた」と書いたが、これは**上限（過大）**だった。
符号化の `counter_value` は **`max(印字カウンター, 【カウンター】イベントの上げ幅)`**
（`encode/tokens.rs` 列 7）で、**イベント側は DON を払わないと打てない**
（`rules/battle.rs::apply_counter`＝カードが EVENT なら `pay_cost` → 能力解決 → トラッシュ／
キャラなら印字カウンターを足してトラッシュ＝**無料**）。つまり手札に大きな数字が見えていても、
ドンが足りなければ**払えない**。

本計器は同じ行集合で**払えるかどうかを入れた会計**を出し直す:

```
free   = Σ（手札のうち EVENT でない枠の印字カウンター）              … 無料
paid   = { (cost_k, value_k) }（手札の【カウンター】イベント）        … DON を払う
budget = 自分のアクティブなドン（scalars 列 2）
best(budget) = free + max{ Σ value_k | Σ cost_k ≤ budget }           … 0/1 ナップサック（DP）
```

`guard_enough_don = ブロッカーが居る or best(budget) ≥ 飛んでくる超過パワー`。
ブロッカーは**レストするだけ＝無料**なので予算に関係しない（`apply_counter` の前段）。

出すもの（`take`＝受けた行・`guard`＝守った行。帯は決着までの自席ターン数）:
  means_naive / means_don   … 守る手段が在った割合（旧＝符号化そのまま／新＝払える手段だけ）
  enough_naive / enough_don … 飛んでくる攻撃を止めるだけの量が在った割合（同）
  blocked_by_don            … **在るのに払えなかった**割合（差の正体）
  budget / free / paid       … ドン・無料カウンター・有料カウンターの分布（読みの材料）

**旧との差がそのまま「上限の緩み」**。差が小さければ `can_or_wont` の結論（守りは選択の誤り）は
そのまま残る。大きければ「払えなかった」＝能力の限界の側へ動く＝手当ての場所が変わる。

**残る近似**（読むときに必ず添える）:
- カウンターは**複数の攻撃に振り分けられる**。ここは `can_or_wont` と同じく
  **そのターン最初の窓**で測る＝1 回の攻撃に対する会計。
- 【カウンター】イベントの上げ幅は符号化の `counter_value`（1 枚 5000 で飽和）から読む＝
  **飽和は過小評価側**。印字カウンターはマスターの実値（飽和しない）。
- ドンは「その行の時点で起きているドン」。相手ターン中に自分のドンが減る要因（相手の効果）は
  行ごとに見ているので入っている。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/guard_afford.py \\
    --in ~/n32_wave/w*/n_records --holdout-mod 7 --out ~/afford_w32.json
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
from race_state import _extra as _race_extra, _incoming  # noqa: E402
from opcg_sim.learned import n_rel as NL  # noqa: E402
from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from opcg_sim.learned.train import time_labels as TL  # noqa: E402

#: scalars（`encode/scalars.rs` の対応表・生の枚数）
SC_MY_DON_ACTIVE = 2
#: tokens の列（`n_rel_feat.S_COLS`）と枠の並び
S_BLOCKER, S_COUNTER, S_IS_EVENT = 6, 7, 19
SLOT_OWN_FIELD, SLOT_HAND = slice(2, 7), slice(12, 22)
COUNTER_SCALE = 2000.0          # `counter_value` は `min(counter/2000, 2.5)`
CLOCK_BANDS = ("t<=2", "t3-4", "t5+")
C_TLEFT = 0                     # `time_labels.TIME_COLS` の先頭


def clock_band(t):
    return "t<=2" if t <= 2.5 else ("t3-4" if t <= 4.5 else "t5+")


def _extra(dd, n):
    sc, tok = _pad(np.asarray(dd["scalars"])[:n].astype(np.float32),
                   np.asarray(dd["tokens"])[:n].astype(np.float32))
    _sc_race, tk_race = _race_extra(dd, n)
    return {"sc": sc, "tok": tok, "race_tk": tk_race,
            "ci": np.asarray(dd["card_idx"])[:n, :NL.N_TOK].astype(np.int64),
            "lives": np.asarray(dd["scalars"])[:n, 0:2].astype(np.float32)}


def knapsack(items, budget):
    """0/1 ナップサック（`items`＝(cost, value)・コストは整数のドン）。戻り＝最大の value。"""
    b = int(max(budget, 0))
    best = [0.0] * (b + 1)
    for cost, value in items:
        c = int(cost)
        if c > b:
            continue
        for k in range(b, c - 1, -1):
            cand = best[k - c] + float(value)
            if cand > best[k]:
                best[k] = cand
    return best[b] if b >= 0 else 0.0


def hand_counters(tok_row, ci_row, idx2cid, cards):
    """手札の枠 → (無料のカウンター合計, 有料イベントの [(cost, value)], 枠ごとの内訳)。

    **無料か有料かはカードの種別で決まる**（`apply_counter`）: EVENT は `pay_cost` が要る・
    キャラ（とイベント以外）は印字カウンターをそのまま足せる。`counter_value` は両方の max
    なので、種別と印字値をマスターから引いて**分解する**。
    """
    tok = np.asarray(tok_row)
    hand_tok = tok[SLOT_HAND]
    hand_ci = np.asarray(ci_row)[SLOT_HAND]
    free = 0.0
    paid = []
    slots = 0
    for j in range(hand_tok.shape[0]):
        cv = float(hand_tok[j, S_COUNTER]) * COUNTER_SCALE
        if float(np.abs(hand_tok[j]).sum()) <= 0.0:
            continue                                    # 空の枠
        slots += 1
        if cv <= 0.0:
            continue
        cid = idx2cid.get(int(hand_ci[j]))
        inf = cards.info(cid) if cid else None
        printed = float((inf or {}).get("counter") or 0.0)
        is_event = bool((inf or {}).get("event")) if inf else bool(hand_tok[j, S_IS_EVENT] > 0.5)
        cost = float((inf or {}).get("cost") or 0.0)
        if printed > 0.0:
            free += printed                             # 無料で払える分（キャラでもイベントでも）
        if is_event and cv > printed:
            # 【カウンター】イベントの上げ幅ぶんは印字を超える＝その差はコストを払って得る
            paid.append((cost, cv - printed))
    return free, paid, slots


def collect(dirs, holdout_mod=7, limit_games=0):
    """holdout の**相手ターンの最初の行**→ 会計の記録（ネット不要）。"""
    cards = PL.Cards()
    idx2cid = {i: c for c, i in _vocab().items()}
    recs = []
    games = 0
    no_true = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, extra_fn=_extra):
        seed = int(rows["seed"][idx[0]])
        if holdout_mod and seed % holdout_mod != 0:
            continue
        games += 1
        if limit_games and games > limit_games:
            break
        life0 = ex["sc"][:, 0]
        labels, _unk = PL.label_game(rows, pol, life0, L, ptr, idx, cards)
        tm_true, tmask = TL.label_game(rows, ex["lives"], idx)
        zs = {int(rows["who"][i]): float(rows["z"][i]) for i in idx}
        seen = set()
        for n, i in enumerate(idx):
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1 or int(labels[n]) < 0 or PL.is_own_turn(w, t):
                continue
            if (w, t) in seen:
                continue
            seen.add((w, t))
            if not tmask[n]:
                no_true += 1
                continue
            tok = ex["tok"][i]
            blocker = bool((np.asarray(tok)[SLOT_OWN_FIELD][:, S_BLOCKER] > 0.5).any())
            naive_sum = float(np.asarray(tok)[SLOT_HAND][:, S_COUNTER].sum()) * COUNTER_SCALE
            free, paid, slots = hand_counters(tok, ex["ci"][i], idx2cid, cards)
            budget = int(round(float(ex["sc"][i][SC_MY_DON_ACTIVE])))
            best = free + knapsack(paid, budget)
            over = _incoming(ex["race_tk"][i])
            recs.append({
                "played": PL.PLAN_CLASSES[int(labels[n])],
                "z": 1.0 if zs.get(w, 0.0) > 0 else 0.0,
                "clock": clock_band(float(tm_true[n][C_TLEFT]) * TL.T_SCALE),
                "over": over, "blocker": blocker, "budget": budget,
                "hand_slots": slots, "free": free, "paid_n": len(paid),
                "paid_cost_min": min((c for c, _v in paid), default=None),
                "naive_sum": naive_sum, "best_don": best,
                "means_naive": bool(blocker or naive_sum > 0.0),
                "means_don": bool(blocker or best > 0.0),
                "enough_naive": bool(blocker or (over is not None and naive_sum >= over)
                                     or (over is not None and over <= 0.0)),
                "enough_don": bool(blocker or (over is not None and best >= over)
                                   or (over is not None and over <= 0.0)),
            })
    return recs, games, no_true


def _vocab():
    from opcg_sim.learned.vocab import shared_vocab
    return shared_vocab()


def _rate(sub, key):
    return round(sum(1 for r in sub if r[key]) / len(sub), 4) if sub else None


def block(recs):
    """1 つの行集合 → 旧／新の会計と、差の正体（払えなかった割合）。"""
    if not recs:
        return None
    out = {"n": len(recs),
           "means_naive": _rate(recs, "means_naive"), "means_don": _rate(recs, "means_don"),
           "enough_naive": _rate(recs, "enough_naive"), "enough_don": _rate(recs, "enough_don"),
           "blocker": _rate(recs, "blocker"),
           "budget_mean": round(float(np.mean([r["budget"] for r in recs])), 2),
           "free_mean": round(float(np.mean([r["free"] for r in recs])), 1),
           "paid_n_mean": round(float(np.mean([r["paid_n"] for r in recs])), 3),
           "naive_sum_mean": round(float(np.mean([r["naive_sum"] for r in recs])), 1),
           "best_don_mean": round(float(np.mean([r["best_don"] for r in recs])), 1)}
    # **差の正体**: 旧では足りていたのに、払えないので足りない行
    lost = [r for r in recs if r["enough_naive"] and not r["enough_don"]]
    out["blocked_by_don"] = round(len(lost) / len(recs), 4)
    out["blocked_by_don_n"] = len(lost)
    over = [r["over"] for r in recs if r["over"] is not None]
    out["over_mean"] = round(float(np.mean(over)), 1) if over else None
    out["no_attack"] = round(sum(1 for r in recs if r["over"] is None) / len(recs), 4)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--holdout-mod", type=int, default=7, help="seed%%N==0 だけ読む（0 で全部）")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    t0 = time.time()
    recs, games, no_true = collect(args.src, args.holdout_mod, args.limit_games)
    out = {"games": games, "rows": len(recs), "rows_without_truth": no_true,
           "src": list(args.src), "all": block(recs)}
    for band in CLOCK_BANDS:
        sub = [r for r in recs if r["clock"] == band]
        out[band] = {"all": block(sub),
                     "take": block([r for r in sub if r["played"] == "take"]),
                     "guard": block([r for r in sub if r["played"] == "guard"])}
    out["take"] = block([r for r in recs if r["played"] == "take"])
    out["guard"] = block([r for r in recs if r["played"] == "guard"])
    out["seconds"] = round(time.time() - t0, 1)
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
