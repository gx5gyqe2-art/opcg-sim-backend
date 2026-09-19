#!/usr/bin/env python3
"""**終局の偏りを 2 項に厳密分解する**（T112・2026-09-19・ユーザ指示「T18 以外で検証できる範囲を検証」）。

## 何を測るか

交点の橋の**終局の偏り**（勝った席の `τ` − その席の実際の残りターン・既定で **+2.56／+2.60**）は
**1 つの数に潰れていて、どこを直せばよいか言わない**。T95〜T111 の改善で 1 度も縮まなかったのは
**場所が分からないまま項を足していた**からである。

本器は偏りを**恒等式として** 2 つに割る:

```
τ_theory − t_left = ( τ_theory − τ_harm ) + ( τ_harm − t_left )
                      ^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^
                      A の軌跡の誤り         的（Θ）と要（実際の損害の総額）の差
```

`τ_harm` は**実際にその席がそのターン与えた損害の列**が `Θ` に届くまでのターン数（端数の配分は
`crossing_bridge.tau_grow` と同じ）。`Σ_{i<t_left} harm_i = 要` なので**第 2 項は `Θ`/要 を時間の単位で
書いたもの**＝T96 以来ずっと見てきた量。**第 1 項はこれまで 1 度も分離されていなかった**。

## 2 つの母数を両方出す（どちらも偏りだが数が違う）

- `turns`＝**勝った席の自席ターン**（`theta_check` の行・1 ターン 1 行）。位相 `j` の分布に偏りが無く、
  **理論の形を読むのはこちら**。
- `rows`＝**判断行**（`rows_out` の行・`crossing_bridge.summarise` の `bias` の母数そのもの）。
  **台帳の数と突き合わせるのはこちら**。負けた席の行は**勝った席の 1 つ前のターン**を見るので、
  残りターンが `t_left − 1`（そのターンは既に終わっている）＝**同じ `τ` を 1 ターン短い物差しで測る**。
  その +1 のぶんだけ `rows` の偏りは `turns` より大きく出る（本器は `own`/`opp` に分けて出す）。

## 成り立つ条件（既定で成り立つ・破れたら器が黙らずに落ちる）

既定の `RACE_MODE=static`・`THETA_HAND_PLACE=stock`・`THETA_RETURN_MODE=off`・`RATE_DECAY_MODE=off` では
`tau_grow` の的は**定数 `Θ`**（`need = theta + r·j + 盾 + 段差` の後ろ 3 項が 0）。
このとき**速さの列だけを差し替えた歩き**が作れる＝分解が恒等式になる。
的が動く構成（`race`／`shield`／`untap`／`decay`）では第 2 項の意味が変わるので、
`--require-static` を既定で有効にし、**既定以外の構成では明示的に降りる**。

## 3 つの出口を突き合わせる鍵（T112 で `collect` に足した `g`/`who`）

`collect` は `turn_harm`（両席の自席ターン）・`theta_check`（勝った席だけ）・`rows_out`（判断行）の
3 つを返す。本器は **`g`（局の序数）と `who`（席）で突き合わせる**——並び順に頼らない。
突き合わせたあと `j` と `t_left` の一致、そして **`τ` が 2 つの経路で同じ値**（T101 で 1 本にした
`tau_theory_of` を両方が通っているか）を**毎回検算する**（食い違えば落ちる）。

使い方:

    python tests/scripts/bias_budget.py --in <records_dir> [--games N] [--json out.json]
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

import crossing_bridge as CB  # noqa: E402
from theory_order import MU, THETA  # noqa: E402

#: 届かない歩きの打ち切り（**正本は `crossing_bridge.RACE_CAP`**・別の値を持たない）
CAP = CB.RACE_CAP


def static_target():
    """**的が定数か**（分解が恒等式になる条件）。既定の構成では真。"""
    return (CB.RACE_MODE == "static" and CB.THETA_HAND_PLACE == "stock"
            and CB.THETA_RETURN_MODE == "off" and CB.RATE_DECAY_MODE == "off")


def tau_of_sequence(theta, adds, tail, cap=CAP):
    """**列 `adds` が `Θ` に届くまでのターン数**（端数の配分は `tau_grow` と同じ・届かねば `tail` で延長）。

    `adds` が尽きたら `tail`（その席の 1 自席ターンあたりの平均の損害）で埋める——
    **記録は勝った席の最後のターンで終わる**ので、`Θ` が大きい行では列が足りない。
    `tail <= 0` なら永遠に届かないので `cap` を返す。"""
    theta = float(theta)
    if theta <= 0.0:
        return 0.0
    f = 0.0
    n = len(adds)
    for j in range(1, int(cap) + 1):
        add = float(adds[j - 1]) if j - 1 < n else float(tail)
        if f + add >= theta:
            short = max(0.0, theta - f)
            return float(j - 1) + (short / add if add > 1e-9 else 1.0)
        f += add
    return float(cap)


def winner_blocks(turn_harm, theta_check, cap=CAP):
    """**勝った席の自席ターンの列**を局ごとに切り、`theta_check` と**組にして**返す。

    返すのは `{(g, who): [(harm 行, theta_check 行), ...]}`（`j` の昇順）。
    **突き合わせは鍵（`g`・`who`・`j`）で行い、並び順には頼らない**。
    `j`/`t_left` が食い違えば `AssertionError`（黙って別の行を組にしない）。"""
    hw = {}
    for r in turn_harm:
        if bool(r.get("won")):
            hw[(int(r["g"]), int(r["who"]), int(r["j"]))] = r
    assert len(hw) == len(theta_check), (
        "turn_harm(won) と theta_check の本数が違う: %d 対 %d" % (len(hw), len(theta_check)))
    out = {}
    for tc in theta_check:
        key = (int(tc["g"]), int(tc["who"]), int(tc["j"]))
        h = hw.get(key)
        assert h is not None, "theta_check の行に対応する turn_harm が無い: %s" % (key,)
        assert int(h["t_left"]) == int(tc["t_left"]), (
            "残りターンが食い違う: harm %s 対 check %s（%s）" % (h["t_left"], tc["t_left"], key))
        out.setdefault(key[:2], []).append((int(tc["j"]), h, tc))
    blocks = {}
    for key, items in out.items():
        items.sort(key=lambda x: x[0])
        assert [x[0] for x in items] == list(range(len(items))), (
            "自席ターン番号が連番でない（%s）: %s" % (key, [x[0] for x in items]))
        harm = [float(h["harm"]) for _j, h, _tc in items]
        n = len(harm)
        blocks[key] = {
            "pairs": [(h, tc) for _j, h, tc in items],
            "harm": harm,
            # 届かない行の延長に使う 1 自席ターンあたりの平均
            "tail": float(np.mean(harm)) if harm else 0.0,
            # **最後のターンを外した平均**でも測る——勝った席の最後の自席ターンは
            # 相手のライフが 0 になった瞬間に終わるので短い（T107）＝延長の材料には偏る。
            "tail_exlast": float(np.mean(harm[:-1])) if n >= 2 else (float(np.mean(harm)) if harm else 0.0),
        }
    return blocks


def _split(theta, tau, t_act, adds, tail, tail_ex, cap=CAP):
    """1 行ぶんの分解（`res == res_rate + res_target` は構成上厳密）。"""
    tau = min(float(tau), cap)
    th_all = tau_of_sequence(theta, adds, tail, cap)
    th_ex = tau_of_sequence(theta, adds, tail_ex, cap)
    return {"tau": tau, "tau_harm": th_all, "tau_harm_exlast": th_ex,
            "res": tau - t_act,                 # 偏り（`summarise` と同じ量）
            "res_rate": tau - th_all,           # **A の軌跡の誤り**
            "res_target": th_all - t_act,       # **的（Θ 対 要）の差**
            "res_target_exlast": th_ex - t_act}


def decompose_turns(blocks, cap=CAP):
    """**母数＝勝った席の自席ターン**（`theta_check` の行）で分解する。"""
    rows = []
    for (g, who), b in sorted(blocks.items()):
        for k, (h, tc) in enumerate(b["pairs"]):
            d = _split(float(tc["theta"]), float(tc["tau"]), float(tc["t_left"]),
                       b["harm"][k:], b["tail"], b["tail_exlast"], cap)
            d.update({"g": g, "who": who, "j": int(tc["j"]), "t_left": float(tc["t_left"]),
                      "theta": float(tc["theta"]), "need": float(tc["need"]), "own": True,
                      "capped": bool(float(tc["tau"]) >= cap),
                      "harm": float(h["harm"]), "slope_theory": float(h.get("slope_theory") or 0.0)})
            rows.append(d)
    return rows


def decompose_rows(rows_out, blocks, cap=CAP):
    """**母数＝判断行**（`crossing_bridge.summarise` の `bias` と同じ）で分解する。

    勝った席の行は自分の `τ`／残りターン、負けた席の行は**相手（＝勝った席）の 1 つ前のターン**の
    `τ` と `t_opp_act`（そのターンを除いた残り）を読む——`summarise` と同じ読み方。"""
    rows = []
    for rec in rows_out:
        won = bool(rec["won"])
        wseat = int(rec["who"]) if won else 1 - int(rec["who"])
        j = int(rec["j_me"]) if won else int(rec["j_opp"])
        b = blocks.get((int(rec["g"]), wseat))
        if b is None or j >= len(b["harm"]):
            continue
        h, tc = b["pairs"][j]
        theta = float(rec["theta_me"]) if won else float(rec["theta_opp"])
        tau = float(rec["tau_me_theory"]) if won else float(rec["tau_opp_theory"])
        t_act = float(rec["t_me_act"]) if won else float(rec["t_opp_act"])
        # **2 つの経路の τ が同じか**（T101 で `tau_theory_of` に 1 本化した式を両方が通っているか）
        assert abs(min(tau, cap) - min(float(tc["tau"]), cap)) < 1e-6, (
            "τ が 2 つの経路で違う: rows_out %s 対 theta_check %s" % (tau, tc["tau"]))
        assert abs(theta - float(tc["theta"])) < 1e-6, (
            "Θ が 2 つの経路で違う: rows_out %s 対 theta_check %s" % (theta, tc["theta"]))
        d = _split(theta, tau, t_act, b["harm"][j:], b["tail"], b["tail_exlast"], cap)
        d.update({"g": int(rec["g"]), "who": wseat, "j": j, "t_left": t_act,
                  "theta": theta, "need": float(tc["need"]), "own": won,
                  "capped": bool(tau >= cap),
                  "harm": float(h["harm"]), "slope_theory": float(h.get("slope_theory") or 0.0)})
        rows.append(d)
    return rows


def _band(rows, key, labels=5):
    by = {}
    for r in rows:
        v = int(r[key])
        by.setdefault(str(v) if v <= labels else "%d+" % (labels + 1), []).append(r)
    return {k: {"n": len(g),
                "res": round(float(np.mean([x["res"] for x in g])), 3),
                "res_median": round(float(np.median([x["res"] for x in g])), 3),
                "rate": round(float(np.mean([x["res_rate"] for x in g])), 3),
                "target": round(float(np.mean([x["res_target"] for x in g])), 3),
                "theta": round(float(np.mean([x["theta"] for x in g])), 4),
                "need": round(float(np.mean([x["need"] for x in g])), 4)}
            for k, g in sorted(by.items(), key=lambda kv: (kv[0].endswith("+"), kv[0]))}


def summarise_budget(rows):
    """行ごとの分解を 1 枚の表にする（**恒等式の検算を必ず出す**）。"""
    if not rows:
        return {"n": 0}
    res = np.array([r["res"] for r in rows])
    rr = np.array([r["res_rate"] for r in rows])
    rt = np.array([r["res_target"] for r in rows])
    rtx = np.array([r["res_target_exlast"] for r in rows])
    srt = np.sort(res)[::-1]
    k = max(1, int(round(0.1 * len(srt))))
    tot = float(res.mean())
    out = {
        "n": len(rows),
        # **恒等式の検算**: 行ごとに `res == res_rate + res_target` が成り立つ（浮動小数の誤差だけ）
        "identity_max_abs_error": float(np.abs(rr + rt - res).max()),
        "total_mean": round(tot, 4), "total_median": round(float(np.median(res)), 4),
        "p25": round(float(np.percentile(res, 25)), 3),
        "p75": round(float(np.percentile(res, 75)), 3),
        "p90": round(float(np.percentile(res, 90)), 3),
        "sigma": round(float(res.std()), 3), "mae": round(float(np.abs(res).mean()), 3),
        # **平均が裾の統計かどうか**——上位 10% の行が総和の何割を出しているか
        "top10pct_contrib": round(float(srt[:k].sum() / res.sum()), 4) if res.sum() else None,
        "capped_share": round(float(np.mean([r["capped"] for r in rows])), 4),
        "rate_mean": round(float(rr.mean()), 4), "rate_median": round(float(np.median(rr)), 4),
        "target_mean": round(float(rt.mean()), 4), "target_median": round(float(np.median(rt)), 4),
        "target_mean_exlast": round(float(rtx.mean()), 4),
        "share_rate": round(float(rr.mean() / tot), 4) if abs(tot) > 1e-9 else None,
        "share_target": round(float(rt.mean() / tot), 4) if abs(tot) > 1e-9 else None,
        "theta_over_need": round(float(np.mean([r["theta"] for r in rows])
                                      / max(1e-9, np.mean([r["need"] for r in rows]))), 4),
        # **位相（自席ターン番号 j）別**が本器の主産物——偏りがどの位相に住んでいるか
        "by_j": _band(rows, "j"),
        "by_t_left": _band(rows, "t_left"),
    }
    own = [r for r in rows if r["own"]]
    opp = [r for r in rows if not r["own"]]
    if own and opp:
        # **物差しの +1**（負けた席の行は勝った席のそのターンを除いた残りと比べる）を見えるようにする
        out["by_side"] = {"own": {"n": len(own), "res": round(float(np.mean([r["res"] for r in own])), 4)},
                          "opp": {"n": len(opp), "res": round(float(np.mean([r["res"] for r in opp])), 4)}}
        # **同じターン**（同じ `g`・席・`j`）を両側から見た行だけを組にすると、差は**厳密に 1**
        # ——`τ` も `Θ` も同じ値なので、違うのは物差し（`t_left` 対 `t_left − 1`）だけ。
        d_own = {(r["g"], r["who"], r["j"]): r for r in own}
        diffs = [r["res"] - d_own[(r["g"], r["who"], r["j"])]["res"]
                 for r in opp if (r["g"], r["who"], r["j"]) in d_own]
        if diffs:
            out["by_side"]["paired_n"] = len(diffs)
            out["by_side"]["paired_offset_mean"] = round(float(np.mean(diffs)), 6)
            out["by_side"]["paired_offset_max_abs_error"] = float(np.abs(np.array(diffs) - 1.0).max())
    return out


def collect_budget(dirs, limit_games=0, theta=None, mu=None, require_static=True):
    """記録を 1 度読んで**両方の母数**で分解する（`crossing_bridge.collect` を 1 回呼ぶだけ）。"""
    if require_static and not static_target():
        raise SystemExit(
            "的が動く構成では分解が恒等式にならない（RACE_MODE=%s THETA_HAND_PLACE=%s "
            "THETA_RETURN_MODE=%s RATE_DECAY_MODE=%s）。--no-require-static で強行できる。"
            % (CB.RACE_MODE, CB.THETA_HAND_PLACE, CB.THETA_RETURN_MODE, CB.RATE_DECAY_MODE))
    theta = THETA if theta is None else theta
    mu = MU if mu is None else mu
    rows_out, _ledger, stats, turn_harm, theta_check = CB.collect(dirs, limit_games, theta, mu, "const")
    blocks = winner_blocks(turn_harm, theta_check)
    by_turn = decompose_turns(blocks)
    by_row = decompose_rows(rows_out, blocks)
    # **台帳そのものと突き合わせる**（`rows` の母数は `summarise` の `bias` と同じ行集合のはず）
    led = CB.summarise(rows_out, _ledger, turn_harm, theta_check)
    led_bias = ((led.get("by_slope") or {}).get("theory") or {}).get("bias")
    out = {"turns": summarise_budget(by_turn), "rows": summarise_budget(by_row),
           "ledger_bias": led_bias, "ledger_sigma_T": ((led.get("by_slope") or {}).get("theory") or {}).get("sigma_T"),
           "static_target": static_target(),
           "modes": {"race": CB.RACE_MODE, "theta_hand_place": CB.THETA_HAND_PLACE,
                     "theta_return": CB.THETA_RETURN_MODE, "rate_decay": CB.RATE_DECAY_MODE,
                     "rate_walk": CB.RATE_WALK_MODE, "don_purse": CB.DON_PURSE_MODE,
                     "theta_don": CB.THETA_DON_MODE, "theta_hand": CB.THETA_HAND_MODE,
                     "theta_hand_window": CB.THETA_HAND_WINDOW, "rate_don": CB.RATE_DON_MODE,
                     "rate_don_pay": CB.RATE_DON_PAY, "rate_ramp": CB.RATE_RAMP},
           "games": stats.get("games"), "rows_out": len(rows_out), "blocks": len(blocks)}
    return out, {"turns": by_turn, "rows": by_row}


def main(argv=None):
    ap = argparse.ArgumentParser(description="終局の偏りを (A の軌跡, 的) に厳密分解する（T112）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--json", default="")
    ap.add_argument("--no-require-static", dest="require_static", action="store_false",
                    help="的が動く構成でも強行する（分解は恒等式でなくなる）")
    # **T114／T116**: 切替ごとに分解を測り直せるようにする（**報告の数を再現する唯一の道**）
    ap.add_argument("--rate-don", default=CB.RATE_DON_MODE, choices=CB.RATE_DON_MODES)
    ap.add_argument("--rate-don-pay", default="on", choices=("on", "off"))
    ap.add_argument("--rate-ramp", type=float, default=CB.RATE_RAMP)
    ap.add_argument("--theta-hand", default=CB.THETA_HAND_MODE, choices=CB.THETA_HAND_MODES)
    ap.add_argument("--theta-hand-window", default=CB.THETA_HAND_WINDOW, choices=CB.THETA_HAND_WINDOWS)
    a = ap.parse_args(argv)
    CB.set_theta_hand_mode(a.theta_hand)
    CB.set_theta_hand_window(a.theta_hand_window)
    CB.set_rate_don_mode(a.rate_don, pay=(a.rate_don_pay == "on"), ramp=a.rate_ramp)
    out, _rows = collect_budget(a.src, a.games, require_static=a.require_static)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
